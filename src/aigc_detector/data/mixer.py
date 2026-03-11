"""Mixed text construction: AI Completion and Sentence Insertion methods.

Constructs mixed (human + AI) text samples from existing human text by:
- AI Completion: Taking the first portion of human text and using an LLM to continue it
- Sentence Insertion: Inserting AI-generated sentences into human text at random positions

These mixed samples are used to train the detector on partially AI-written content.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

import torch
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aigc_detector.config import settings
from aigc_detector.utils.text import clean_text, is_chinese, split_sentences_bilingual

console = Console()


def _generate_continuation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """Generate a text continuation using the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def ai_completion(
    human_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    split_ratio: float = 0.5,
    lang: str | None = None,
) -> dict | None:
    """Construct mixed text via AI completion of human text prefix.

    Takes the first `split_ratio` portion of the human text (by sentences),
    then uses the LLM to continue writing in the same style.

    Args:
        human_text: The original human-written text.
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        split_ratio: Proportion of human text to keep (default: 0.5).
        lang: Language code. Auto-detected if None.

    Returns:
        Mixed text record dict, or None if generation fails.
    """
    if lang is None:
        lang = "zh" if is_chinese(human_text) else "en"

    sentences = split_sentences_bilingual(human_text)
    if len(sentences) < 4:
        return None  # Too few sentences to split meaningfully

    split_point = max(2, int(len(sentences) * split_ratio))
    human_part_sentences = sentences[:split_point]

    if lang == "zh":
        human_part = "".join(human_part_sentences)
        prompt = f"请继续以下文本，保持相同的风格和主题：\n\n{human_part}"
    else:
        human_part = " ".join(human_part_sentences)
        prompt = f"Continue the following text, maintaining the same style and topic:\n\n{human_part}"

    # Estimate how much to generate based on remaining text length
    remaining_sentences = sentences[split_point:]
    if lang == "zh":
        remaining_len = len("".join(remaining_sentences))
    else:
        remaining_len = len(" ".join(remaining_sentences).split())

    max_new_tokens = min(max(remaining_len * 2, 128), 1024)

    try:
        ai_part = _generate_continuation(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    except Exception as e:
        console.print(f"[red]Completion generation error:[/] {e}")
        return None

    if len(ai_part) < 50:
        return None  # Too short, discard

    mixed_text = human_part + ai_part
    if len(mixed_text) < 200:
        return None

    return {
        "id": f"m_{uuid.uuid4().hex[:8]}",
        "text": mixed_text,
        "label": "mixed",
        "lang": lang,
        "method": "ai_completion",
        "boundary_char_idx": len(human_part),
        "human_ratio": len(human_part) / len(mixed_text),
    }


def sentence_insertion(
    human_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    insert_ratio: float = 0.3,
    lang: str | None = None,
) -> dict | None:
    """Construct mixed text by inserting AI-generated sentences into human text.

    Randomly selects positions in the human text and generates contextually
    appropriate AI sentences to insert at those positions.

    Args:
        human_text: The original human-written text.
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        insert_ratio: Proportion of sentences to insert (default: 0.3).
        lang: Language code. Auto-detected if None.

    Returns:
        Mixed text record dict, or None if generation fails.
    """
    if lang is None:
        lang = "zh" if is_chinese(human_text) else "en"

    sentences = split_sentences_bilingual(human_text)
    if len(sentences) < 5:
        return None  # Too few sentences for meaningful insertion

    num_insertions = max(1, int(len(sentences) * insert_ratio))
    # Choose insertion positions (indices after which to insert)
    insert_positions = sorted(random.sample(range(len(sentences)), min(num_insertions, len(sentences))))

    sentence_labels = ["human"] * len(sentences)
    offset = 0

    for pos in insert_positions:
        actual_pos = pos + offset
        # Build context from surrounding sentences
        context_start = max(0, actual_pos - 1)
        context_end = min(len(sentences), actual_pos + 2)
        context_sentences = sentences[context_start:context_end]

        if lang == "zh":
            context = "".join(context_sentences)
            prompt = f"根据以下文本的上下文，生成一个自然衔接的句子：\n\n{context}\n\n请只输出一个句子："
        else:
            context = " ".join(context_sentences)
            prompt = (
                f"Based on the following context, generate one sentence that fits naturally:"
                f"\n\n{context}\n\nGenerate only one sentence:"
            )

        try:
            ai_sentence = _generate_continuation(model, tokenizer, prompt, max_new_tokens=128, temperature=0.9)
        except Exception:
            continue

        # Clean and validate the generated sentence
        ai_sentence = clean_text(ai_sentence)
        # Take only the first sentence if multiple were generated
        ai_sents = split_sentences_bilingual(ai_sentence)
        if not ai_sents:
            continue
        ai_sentence = ai_sents[0]

        if len(ai_sentence) < 10:
            continue

        # Insert after the current position
        sentences.insert(actual_pos + 1, ai_sentence)
        sentence_labels.insert(actual_pos + 1, "ai")
        offset += 1

    # Check that we actually inserted something
    ai_count = sum(1 for label in sentence_labels if label == "ai")
    if ai_count == 0:
        return None

    if lang == "zh":
        mixed_text = "".join(sentences)
    else:
        mixed_text = " ".join(sentences)

    if len(mixed_text) < 200:
        return None

    return {
        "id": f"m_{uuid.uuid4().hex[:8]}",
        "text": mixed_text,
        "label": "mixed",
        "lang": lang,
        "method": "sentence_insertion",
        "sentence_labels": sentence_labels,
        "human_ratio": sum(1 for lbl in sentence_labels if lbl == "human") / len(sentence_labels),
    }


def generate_mixed_texts(
    human_jsonl_path: Path,
    output_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_completion: int = 5000,
    num_insertion: int = 5000,
    split_ratios: list[float] | None = None,
    insert_ratios: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """Generate mixed text samples from a human text JSONL file.

    Args:
        human_jsonl_path: Path to JSONL file with human text records.
        output_path: Path to write mixed text JSONL output.
        model: Loaded language model for generation.
        tokenizer: Corresponding tokenizer.
        num_completion: Number of AI completion samples to generate.
        num_insertion: Number of sentence insertion samples to generate.
        split_ratios: List of split ratios for AI completion (cycled). Default: [0.3, 0.5, 0.7].
        insert_ratios: List of insert ratios for sentence insertion (cycled). Default: [0.2, 0.3, 0.4].
        seed: Random seed for reproducibility.

    Returns:
        Stats dict with generation summary.
    """
    random.seed(seed)

    if split_ratios is None:
        split_ratios = [0.3, 0.5, 0.7]
    if insert_ratios is None:
        insert_ratios = [0.2, 0.3, 0.4]

    # Read human texts
    human_records: list[dict] = []
    with open(human_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                human_records.append(json.loads(line))

    if not human_records:
        console.print("[red]No human records found.[/]")
        return {"completion_generated": 0, "insertion_generated": 0}

    random.shuffle(human_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completion_count = 0
    insertion_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        # AI Completion
        console.print(f"[bold blue]Generating AI completion samples:[/] target={num_completion}")
        pbar = tqdm(total=num_completion, desc="AI Completion", unit="sample")

        for i, record in enumerate(human_records):
            if completion_count >= num_completion:
                break

            ratio = split_ratios[i % len(split_ratios)]
            result = ai_completion(
                human_text=record["text"],
                model=model,
                tokenizer=tokenizer,
                split_ratio=ratio,
                lang=record.get("lang"),
            )

            if result is not None:
                result["source_id"] = record.get("id", "")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                completion_count += 1
                pbar.update(1)

        pbar.close()

        # Sentence Insertion — use a different portion of human records
        console.print(f"[bold blue]Generating sentence insertion samples:[/] target={num_insertion}")
        remaining = human_records[num_completion:]
        if len(remaining) < num_insertion:
            # Wrap around if we don't have enough
            remaining = human_records

        pbar = tqdm(total=num_insertion, desc="Sentence Insertion", unit="sample")

        for i, record in enumerate(remaining):
            if insertion_count >= num_insertion:
                break

            ratio = insert_ratios[i % len(insert_ratios)]
            result = sentence_insertion(
                human_text=record["text"],
                model=model,
                tokenizer=tokenizer,
                insert_ratio=ratio,
                lang=record.get("lang"),
            )

            if result is not None:
                result["source_id"] = record.get("id", "")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                insertion_count += 1
                pbar.update(1)

        pbar.close()

    stats = {
        "completion_generated": completion_count,
        "insertion_generated": insertion_count,
        "total_mixed": completion_count + insertion_count,
    }

    console.rule("[bold]Mixed Text Generation Summary")
    console.print(f"AI Completion: {completion_count}")
    console.print(f"Sentence Insertion: {insertion_count}")
    console.print(f"Total: {completion_count + insertion_count}")
    console.print(f"Output: {output_path}")

    return stats


def generate_mixed_all(
    human_jsonl_path: Path | None = None,
    output_dir: Path | None = None,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    num_per_method_per_lang: int = 5000,
    seed: int = 42,
) -> Path:
    """End-to-end mixed text generation: load model, process per-language, unload.

    Args:
        human_jsonl_path: Path to processed human text JSONL. Default: dataset/raw/human_processed.jsonl.
        output_dir: Output directory. Default: dataset/raw.
        model_id: HuggingFace model ID for generation.
        num_per_method_per_lang: Samples per method per language.
        seed: Random seed.

    Returns:
        Path to the output JSONL file.
    """
    from aigc_detector.data.generator import load_model_and_tokenizer, unload_model

    if human_jsonl_path is None:
        human_jsonl_path = settings.dataset_dir / "raw" / "human_processed.jsonl"
    if output_dir is None:
        output_dir = settings.dataset_dir / "raw"

    output_path = output_dir / "mixed_raw.jsonl"

    console.rule("[bold]Mixed Text Generation Pipeline")
    console.print(f"Model: {model_id}")
    console.print(f"Input: {human_jsonl_path}")
    console.print(f"Output: {output_path}")

    model, tokenizer = load_model_and_tokenizer(model_id, model_id, settings.device)

    try:
        generate_mixed_texts(
            human_jsonl_path=human_jsonl_path,
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            num_completion=num_per_method_per_lang,
            num_insertion=num_per_method_per_lang,
            seed=seed,
        )
    finally:
        unload_model(model, tokenizer)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mixed (human+AI) text samples")
    parser.add_argument("--input", type=Path, default=None, help="Input JSONL with human texts")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: dataset/raw)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-per-method", type=int, default=5000, help="Samples per method (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_mixed_all(
        human_jsonl_path=args.input,
        output_dir=args.output_dir,
        model_id=args.model,
        num_per_method_per_lang=args.num_per_method,
        seed=args.seed,
    )
