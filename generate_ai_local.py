import os, json, uuid, pathlib, random, torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from tqdm import tqdm

REPO_ID   = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
MODEL_DIR = pathlib.Path("models/mistral7b-gptq")
DEVICE    = "auto"          # 自动分配到 GPU0
TOKEN     = None            # 公开仓库，无需 token；若网络不稳可加代理环境变量

# ── 1. 下载（断点续传） ──────────────────────────────────
if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.safetensors")):
    snapshot_download(
        repo_id   = REPO_ID,
        local_dir = str(MODEL_DIR),
        resume_download       = True,
        local_dir_use_symlinks= False,
        token=TOKEN,
    )

# ── 2. 加载 4-bit 模型 ─────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    # MODEL_DIR,
    # device_map=DEVICE,
    # torch_dtype=torch.float16,
    # trust_remote_code=True,   # GPTQ 权重需要
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map=DEVICE,            # Accelerate 自动放到 cuda:0
    # load_in_4bit=True,            # ← bitsandbytes 4-bit 开关
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_quant_type="nf4",
)

pipe = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    # device=0,
)

# ── 3. 批量生成样本 ─────────────────────────────────
OUTPUT = pathlib.Path("dataset/ai"); OUTPUT.mkdir(parents=True, exist_ok=True)
meta   = open("dataset/meta.jsonl", "a", encoding="utf-8")

prompts = [
    "Explain the core ideas of zero-trust architecture to a CTO.",
    "Write a short story (300 words) about an AI detective.",
    "Summarize the advantages of IPv6 in 5 bullet points.",
    "Generate a marketing email for a new cybersecurity SaaS.",
    "Translate the following Chinese paragraph into native English: ...",
    "Draft a legal disclaimer for a tech blog.",
    "Produce a poem using iambic pentameter on data privacy.",
    "Compare LSTM and Transformer models in a classroom tone.",
    "Give me 10 interview questions about cloud security.",
    "Describe the future of quantum computing for non-experts."
]
temps = (0.7, 1.0)

for p in tqdm(prompts, desc="prompts"):
    for t in temps:
        out = pipe(
            p, max_new_tokens=512, do_sample=True,
            temperature=t, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"].strip()

        uid = f"a_{uuid.uuid4().hex[:8]}"
        (OUTPUT/f"{uid}.txt").write_text(out, encoding="utf-8")
        meta.write(json.dumps({
            "id": uid, "split": "train", "source": REPO_ID,
            "prompt": p, "temperature": t, "label": "A", "lang": "en"
        })+"\n")

meta.close()
print("✅ All done!")
