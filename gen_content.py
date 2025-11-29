import openai, json, pathlib, uuid, random, time

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


BASE = pathlib.Path("dataset/ai"); BASE.mkdir(parents=True, exist_ok=True)
meta_fp = open("dataset/meta.jsonl", "a", encoding="utf-8")
# AI_Detector API Key
# load api key from .env file via load_dotenv()
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate(model, prompt, temperature):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
for p in prompts:
    for temp in (0.7, 1.0):
        mdl = random.choice(models)
        txt = generate(mdl, p, temp)
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
        time.sleep(1)          # 限速防封
meta_fp.close()
