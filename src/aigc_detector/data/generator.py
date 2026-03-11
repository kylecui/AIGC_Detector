"""Multi-model AI text generator for dataset construction.

Loads quantized LLMs one at a time (to fit RTX 3060 12GB VRAM), generates diverse
text using prompt templates from configs/prompts.yaml, and writes JSONL output.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aigc_detector.config import settings

# Prefer cached models — fall back to download if not available
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

console = Console()

# Models that need special loading (GPTQ vs BnB)
GPTQ_MODELS = {"mistral-7b-gptq", "llama-3-8b-gptq"}


def load_prompts(path: str = "configs/prompts.yaml") -> dict:
    """Load prompt templates from YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_bnb_config() -> BitsAndBytesConfig:
    """Build 4-bit quantization config for BitsAndBytes models."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_tokenizer(
    hf_id: str,
    model_key: str,
    device: str = "cuda",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a single model + tokenizer. Handles GPTQ vs BnB loading."""
    console.print(f"[bold blue]Loading model:[/] {hf_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_key in GPTQ_MODELS:
        # GPTQ models — load directly (pre-quantized weights)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        # BitsAndBytes 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="auto",
            quantization_config=_build_bnb_config(),
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    model.eval()
    console.print(f"[bold green]Loaded:[/] {hf_id} on {device}")
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Unload model from GPU and free VRAM."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[dim]Model unloaded, VRAM freed.[/]")


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate text using chat template. Returns the model's response only."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Use chat template if available, otherwise fall back to basic formatting
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for models without chat template
        input_text = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (skip input)
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def _resolve_model_source_name(model_key: str) -> str:
    """Map model registry key to a short source name for JSONL records."""
    mapping = {
        "mistral-7b-gptq": "mistral7b",
        "qwen2.5-7b": "qwen2.5",
        "chatglm3-6b": "chatglm3",
        "yi-6b-chat": "yi6b",
        "llama-3-8b-gptq": "llama3",
    }
    return mapping.get(model_key, model_key)


def _select_prompts_for_model(
    prompt_config: dict,
    model_langs: list[str],
) -> list[dict[str, Any]]:
    """Build a flat list of generation tasks from prompt config for the given languages.

    Returns list of dicts: {domain, style, lang, system_prompt, user_prompt}
    """
    tasks: list[dict[str, Any]] = []
    system_prompts = prompt_config["system_prompts"]
    prompts = prompt_config["prompts"]

    for domain_key, styles in prompts.items():
        for style_key, lang_dict in styles.items():
            sys_prompts = system_prompts[style_key]
            for lang in model_langs:
                if lang not in lang_dict:
                    continue
                user_prompt_list = lang_dict[lang]
                sys_prompt = sys_prompts[lang]
                for user_prompt in user_prompt_list:
                    tasks.append(
                        {
                            "domain": domain_key,
                            "style": style_key,
                            "lang": lang,
                            "system_prompt": sys_prompt,
                            "user_prompt": user_prompt,
                        }
                    )
    return tasks


def generate_for_model(
    model_key: str,
    hf_id: str,
    model_langs: list[str],
    prompt_config: dict,
    output_path: Path,
    num_per_prompt: int = 50,
    device: str = "cuda",
) -> dict[str, int]:
    """Generate AI text using a single model across all applicable prompts.

    Loads model → generates → unloads. Writes JSONL records to output_path (append mode).

    Args:
        model_key: Registry key (e.g. "qwen2.5-7b")
        hf_id: HuggingFace model ID
        model_langs: Languages this model supports (["zh"], ["en"], or ["zh", "en"])
        prompt_config: Loaded prompts.yaml dict
        output_path: Path to output JSONL file (appended)
        num_per_prompt: Number of generations per unique prompt template
        device: Target device

    Returns:
        Stats dict with generation counts.
    """
    source_name = _resolve_model_source_name(model_key)
    gen_config = prompt_config.get("generation", {})
    temperatures = gen_config.get("temperatures", [0.7, 1.0])
    top_p_values = gen_config.get("top_p_values", [0.9])
    length_ranges = gen_config.get("length_ranges", {"zh": [500], "en": [400]})
    max_new_tokens = gen_config.get("max_new_tokens", 2048)
    rep_penalty = gen_config.get("repetition_penalty", 1.1)

    tasks = _select_prompts_for_model(prompt_config, model_langs)
    if not tasks:
        console.print(f"[yellow]No prompts found for {model_key} (langs={model_langs}), skipping.[/]")
        return {"model": model_key, "generated": 0, "skipped": 0}

    # Load model
    model, tokenizer = load_model_and_tokenizer(hf_id, model_key, device)

    generated = 0
    skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "a", encoding="utf-8") as f:
            total_iterations = len(tasks) * num_per_prompt
            pbar = tqdm(total=total_iterations, desc=f"Generating [{model_key}]", unit="text")

            for task_info in tasks:
                lang = task_info["lang"]
                lengths = length_ranges.get(lang, [500])

                for i in range(num_per_prompt):
                    # Cycle through temperatures, top_p, and lengths for diversity
                    temp = temperatures[i % len(temperatures)]
                    tp = top_p_values[i % len(top_p_values)]
                    target_length = lengths[i % len(lengths)]

                    user_prompt = task_info["user_prompt"].replace("{length}", str(target_length))

                    try:
                        response = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            system_prompt=task_info["system_prompt"],
                            user_prompt=user_prompt,
                            temperature=temp,
                            top_p=tp,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=rep_penalty,
                        )

                        if len(response) < 100:
                            skipped += 1
                            pbar.update(1)
                            continue

                        record = {
                            "id": f"a_{uuid.uuid4().hex[:8]}",
                            "text": response,
                            "label": "ai",
                            "lang": lang,
                            "source": source_name,
                            "domain": task_info["domain"],
                            "gen_params": {
                                "temperature": temp,
                                "top_p": tp,
                                "style": task_info["style"],
                            },
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        generated += 1

                    except Exception as e:
                        console.print(f"[red]Generation error:[/] {e}")
                        skipped += 1

                    pbar.update(1)

            pbar.close()
    finally:
        unload_model(model, tokenizer)

    stats = {"model": model_key, "generated": generated, "skipped": skipped}
    console.print(f"[bold green]{model_key}:[/] generated={generated}, skipped={skipped}")
    return stats


def generate_all(
    output_dir: Path | None = None,
    num_per_prompt: int = 50,
    models: list[str] | None = None,
    prompt_config_path: str = "configs/prompts.yaml",
) -> Path:
    """Run generation across all configured models sequentially (one at a time for VRAM).

    Args:
        output_dir: Directory for output JSONL. Defaults to settings.dataset_dir / "raw".
        num_per_prompt: Number of generations per unique prompt template.
        models: Specific model keys to use. None = all generation models from registry.
        prompt_config_path: Path to prompts YAML config.

    Returns:
        Path to the output JSONL file.
    """
    from aigc_detector.models.registry import get_registry

    if output_dir is None:
        output_dir = settings.dataset_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ai_raw.jsonl"

    prompt_config = load_prompts(prompt_config_path)
    registry = get_registry()

    # Determine which models to use
    if models is not None:
        model_keys = models
    else:
        model_keys = [k for k, v in registry.items() if v.purpose == "generation"]

    # Language mapping per model
    lang_map = {
        "mistral-7b-gptq": ["en"],
        "qwen2.5-7b": ["zh", "en"],
        "chatglm3-6b": ["zh"],
        "yi-6b-chat": ["zh", "en"],
        "llama-3-8b-gptq": ["en"],
    }

    all_stats = []

    console.rule("[bold]AI Text Generation Pipeline")
    console.print(f"Models: {model_keys}")
    console.print(f"Output: {output_path}")
    console.print(f"Generations per prompt: {num_per_prompt}")

    for model_key in model_keys:
        if model_key not in registry:
            console.print(f"[yellow]Model {model_key} not in registry, skipping.[/]")
            continue

        model_info = registry[model_key]
        model_langs = lang_map.get(model_key, ["en"])

        stats = generate_for_model(
            model_key=model_key,
            hf_id=model_info.hf_id,
            model_langs=model_langs,
            prompt_config=prompt_config,
            output_path=output_path,
            num_per_prompt=num_per_prompt,
            device=settings.device,
        )
        all_stats.append(stats)

    # Summary
    console.rule("[bold]Generation Summary")
    total_gen = sum(s["generated"] for s in all_stats)
    total_skip = sum(s["skipped"] for s in all_stats)
    console.print(f"Total generated: {total_gen}")
    console.print(f"Total skipped: {total_skip}")
    console.print(f"Output file: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI text dataset using local LLMs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: dataset/raw)")
    parser.add_argument("--num-per-prompt", type=int, default=50, help="Generations per prompt template")
    parser.add_argument("--models", nargs="*", default=None, help="Specific model keys to use")
    parser.add_argument("--prompts", default="configs/prompts.yaml", help="Path to prompt config YAML")
    args = parser.parse_args()

    generate_all(
        output_dir=args.output_dir,
        num_per_prompt=args.num_per_prompt,
        models=args.models,
        prompt_config_path=args.prompts,
    )
