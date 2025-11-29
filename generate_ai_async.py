"""
generate_ai_async.py
异步并发批量生成，提高速率
"""
import asyncio, pathlib, uuid, json, random, tqdm.asyncio as tq_asyncio
from init_openai import aclient                       # 异步 client
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

sem = asyncio.Semaphore(5)        # 并发窗口，可视配额调大

async def gen_task(prompt, temp):
    async with sem:
        mdl = random.choice(models)
        resp = await aclient.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=512,
        )
        txt = resp.choices[0].message.content.strip()
        uid = f"a_{uuid.uuid4().hex[:8]}"
        (BASE / f"{uid}.txt").write_text(txt, encoding="utf-8")
        meta_fp.write(json.dumps({
            "id": uid,
            "split": "train",
            "source": mdl,
            "prompt": prompt,
            "temperature": temp,
            "label": "A",
            "lang": "en"
        }) + "\n")

async def main():
    tasks = [
        gen_task(p, t)
        for p in prompts
        for t in (0.7, 1.0)
    ]
    for f in tq_asyncio.tqdm.as_completed(tasks):
        await f

asyncio.run(main())
meta_fp.close()
print("✅ 异步批量生成完成！")
