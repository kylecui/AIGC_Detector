#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate AI samples with Meta-Llama-3-8B-Instruct
• 自动检测/下载 → 离线加载
• Accelerate 自动 GPU/CPU offload
"""

import os, json, uuid, argparse, pathlib, shutil
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from dotenv import load_dotenv

# AI_Detector API Key
# load api key from .env file via load_dotenv()
load_dotenv()

REPO_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_DIR = pathlib.Path("models/llama3")        # ↓下载到这里
GPU_MEM   = "10GiB"                              # 调整显存上限
CPU_MEM   = "16GiB"
MAX_NEW_TOKENS = 256                             # 生成长度

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true",
                    help="完全离线模式，不尝试联网")
args = parser.parse_args()

HF_TOKEN = os.getenv("HF_TOKEN")  # 推荐 export HF_TOKEN=hf_xxx
if not HF_TOKEN and not args.offline:
    print("⚠️  未检测到 HF_TOKEN，若仓库 gated 会 401。")

# 1️⃣ 下载模型（若本地不存在 & 非离线）
if not LOCAL_DIR.exists() or not any(LOCAL_DIR.glob("*.bin")):
    if args.offline:
        raise RuntimeError(f"离线模式找不到本地模型 {LOCAL_DIR}")
    print(f"⏬ Downloading {REPO_ID} → {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,

        token=HF_TOKEN,
    )
    # 可选：清理不必要的 tokenizer_checkpoints
    for junk in LOCAL_DIR.glob("*.safetensors.index.json"):
        junk.unlink()
    print("✅ Download finished\n")

# 2️⃣ 离线加载
print("🚀 Loading model (GPU+CPU offload)…")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    torch_dtype="auto",
    device_map="auto",
    max_memory={0: GPU_MEM, "cpu": CPU_MEM},
    local_files_only=True,
)

gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompts = [
    "Explain the core ideas of zero-trust architecture to a CTO.",
    "Write a short story (300 words) about an AI detective.",
    "Summarize the advantages of IPv6 in 5 bullet points.",
    "Generate a marketing email for a new cybersecurity SaaS.",
    "Translate the following Chinese paragraph into native English:\n零信任是一种全新的安全架构……",
    "Draft a legal disclaimer for a tech blog.",
    "Produce a poem using iambic pentameter on data privacy.",
    "Compare LSTM and Transformer models in a classroom tone.",
    "Give me 10 interview questions about cloud security.",
    "Describe the future of quantum computing for non-experts.",
]
temps = (0.7, 1.0)

out_dir = pathlib.Path("dataset/ai"); out_dir.mkdir(parents=True, exist_ok=True)
meta_fp = open("dataset/meta.jsonl", "a", encoding="utf-8")

for p in tqdm(prompts, desc="prompts"):
    for t in temps:
        txt = gen(
            p,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=t,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"].strip()

        uid = f"a_{uuid.uuid4().hex[:8]}"
        (out_dir / f"{uid}.txt").write_text(txt, encoding="utf-8")
        meta_fp.write(json.dumps({
            "id": uid,
            "split": "train",
            "source": "llama3-8b-fp16-offload",
            "prompt": p,
            "temperature": t,
            "label": "A",
            "lang": "en"
        }) + "\n")

meta_fp.close()
print("🎉 生成完成！共写入", len(list(out_dir.glob('a_*.txt'))), "条样本")
