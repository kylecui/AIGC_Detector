import requests, json, pathlib, uuid
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = pathlib.Path("dataset/human")
BASE.mkdir(parents=True, exist_ok=True)
meta_fp = open("dataset/meta.jsonl", "a", encoding="utf-8")

URLS = [
    "https://en.wikipedia.org/wiki/Zero_trust_security_model",
    "https://www.theverge.com/tech",   # 列表页示例
]

def crawl(url):
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "lxml")
    # 简陋正文提取：只拿 <p>
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(paras).strip()
    return text if len(text.split()) > 100 else None

for url in tqdm(URLS):
    txt = crawl(url)
    if txt:
        uid = f"h_{uuid.uuid4().hex[:8]}"
        (BASE / f"{uid}.txt").write_text(txt, encoding="utf-8")
        meta_fp.write(json.dumps({
            "id": uid,
            "split": "train",
            "source": url,
            "label": "H",
            "lang": "en"
        }) + "\n")

meta_fp.close()
