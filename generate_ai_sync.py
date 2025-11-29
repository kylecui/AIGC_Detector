"""
generate_ai_sync.py
按 prompts × 温度 批量生成 A 类样本，写入 dataset/ai 并记录 meta.jsonl
"""
import pathlib, uuid, json, time, random
from tqdm import tqdm
from init_openai import client   # 同步 client
# -------------------------------

BASE = pathlib.Path("dataset/ai")
BASE.mkdir(parents=True, exist_ok=True)
meta_fp = open("dataset/meta.jsonl", "a", encoding="utf-8")

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

models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]

def generate_one(model: str, prompt: str, temp: float) -> str:
    """同步调用；必要时捕获 openai.RateLimitError 重试"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

for p in tqdm(prompts, desc="prompts"):
    for temp in (0.7, 1.0):
        mdl = random.choice(models)
        txt = generate_one(mdl, p, temp)
        uid = f"a_{uuid.uuid4().hex[:8]}"
        (BASE / f"{uid}.txt").write_text(txt, encoding="utf-8")
        meta_fp.write(json.dumps({
            "id": uid,
            "split": "train",
            "source": mdl,
            "prompt": p,
            "temperature": temp,
            "label": "A",
            "lang": "en"
        }) + "\n")
        time.sleep(1)        # 轻量限速，避免 429
meta_fp.close()
print("✅ 同步生成完成！")
