#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Hugging Face 自动下载（或复用缓存）Llama-3-8B-Instruct，
再批量生成 AI 样本，写入 dataset/ai 并同步记录 meta.jsonl
"""

import os, json, uuid, pathlib, random, torch, argparse
from tqdm import tqdm
from huggingface_hub import snapshot_download, HfFolder
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)

# =============================================================
# 0️⃣  CLI 参数
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="Hugging Face 仓库 ID")
parser.add_argument("--model_dir", default="models/llama3-8b",
                    help="本地模型存放路径")
parser.add_argument("--quant", action="store_true",
                    help="是否 4-bit 量化加载 (bitsandbytes)")
parser.add_argument("--device", default="auto",
                    help="'auto' | cpu | 0,1,2... (多 GPU 逗号分隔)")
args = parser.parse_args()

REPO_ID   = args.repo
MODEL_DIR = pathlib.Path(args.model_dir).expanduser()
QUANT     = args.quant
DEVICE    = args.device

# =============================================================
# 1️⃣  下载或复用本地模型
# -------------------------------------------------------------
if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.bin")):
    print(f"⏬ 模型 {REPO_ID} 未检测到本地权重，开始下载...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 如果你已在 shell 登陆过 HF CLI，可省略 token
    HF_TOKEN = os.getenv("HF_TOKEN") or HfFolder.get_token()

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODEL_DIR),
        resume_download=True,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    print("✅ 模型下载完成！")
else:
    print(f"✅ 检测到本地模型文件，路径：{MODEL_DIR}")

# =============================================================
# 2️⃣  加载 tokenizer & model
# -------------------------------------------------------------
print("🚀 正在加载模型到内存...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

if QUANT:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map=DEVICE,          # auto 会按显存分配；CPU 则填 "cpu"
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map=DEVICE,
        torch_dtype=torch.float16,
    )

pipe = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=-1 if DEVICE == "cpu" else 0,   # transformers 只接收 int / -1
)

print("✅ 模型加载完成！\n")

# =============================================================
# 3️⃣  样本生成
# -------------------------------------------------------------
OUTPUT_DIR = pathlib.Path("dataset/ai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
meta_fp = open("dataset/meta.jsonl", "a", encoding="utf-8")

prompts = [
    "Explain the core ideas of zero-trust architecture to a CTO.",
    "Write a short story (300 words) about an AI detective.",
    "Summarize the advantages of IPv6 in 5 bullet points.",
    "Generate a marketing email for a new cybersecurity SaaS.",
    "Translate the following Chinese paragraph into native English:\n零信任架构是一种…",
    "Draft a legal disclaimer for a tech blog.",
    "Produce a poem using iambic pentameter on data privacy.",
    "Compare LSTM and Transformer models in a classroom tone.",
    "Give me 10 interview questions about cloud security.",
    "Describe the future of quantum computing for non-experts.",
]

temperatures = (0.7, 1.0)

for prompt in tqdm(prompts, desc="prompts"):
    for temp in temperatures:
        out = pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"].strip()

        uid = f"a_{uuid.uuid4().hex[:8]}"
        (OUTPUT_DIR / f"{uid}.txt").write_text(out, encoding="utf-8")
        meta_fp.write(json.dumps({
            "id": uid,
            "split": "train",
            "source": f"{MODEL_DIR.name}{'-4bit' if QUANT else ''}",
            "prompt": prompt,
            "temperature": temp,
            "label": "A",
            "lang": "en"
        }) + "\n")
meta_fp.close()

print("\n🎉 样本生成完毕！生成文件数：",
      len(list(OUTPUT_DIR.glob('a_*.txt'))))
