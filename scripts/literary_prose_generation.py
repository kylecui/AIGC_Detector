"""AI literary-prose generation for the literary_prose_zh probe set.

30 first-person lyrical essay topics x 3 models x 3 seeds (free-form only —
this set studies RAW literary generation). Checkpoint-resumable, modeled on
w4en_generation.py. Usage:
    uv run python scripts/literary_prose_generation.py --budget-seconds 480
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import paired_generation_experiment as _pge  # noqa: E402
from paired_generation_experiment import gen_api  # noqa: E402

# gen_api reads the module-level SYSTEM_PROMPT (default: 公文助手) at call
# time; rebind it to a literary prompt (same wiring trick as w4en_generation).
SYSTEM_PROMPT = "你是一位擅长中文抒情散文的写作者。"
_pge.SYSTEM_PROMPT = SYSTEM_PROMPT

OUT = Path("dataset/literary_prose_zh/ai_records.jsonl")
MODELS = ["THUDM/GLM-4-9B-0414", "Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-V3.2"]
SEEDS = [11, 22, 33]
SYSTEM_PROMPT = "你是一位擅长中文抒情散文的写作者。"

TOPICS = [
    "梅雨季的窗台", "初雪落城的那个傍晚", "秋夜的火车站", "清晨的雾与早市",
    "台风来临前", "老城区拆迁的那个夏天", "夜班公交上的陌生人", "菜市场的清晨",
    "地铁末班车", "外婆的厨房", "翻到旧课本的那个下午", "搬家的日子",
    "桥洞下的旧书摊", "医院走廊的灯", "雨天便利店", "屋顶上的星空",
    "巷口的修鞋匠", "夏夜天台的风", "渡轮上的黄昏", "老家屋后的竹林",
    "冬夜的一碗面", "旧照片里的操场", "公园长椅上的老人", "雷雨夜的窗前",
    "午后的旧书店", "海边的清晨", "山间小站的等待", "楼下的桂花树",
    "深夜的洗衣房", "春天最后一次倒春寒",
]


def prompt(topic: str) -> str:
    return f"请写一篇中文抒情散文，主题：{topic}。第一人称，重感受与意象，避免陈词滥调的堆砌，600-900字。直接输出正文。"


def done() -> set[str]:
    keys: set[str] = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add(f"{r['model']}|{r['topic_id']}|{r['seed']}")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=480)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dk = done()
    jobs = [(m, t, s) for m in MODELS for t in TOPICS for s in SEEDS
            if f"{m}|{t}|{s}" not in dk]
    if not jobs:
        print("nothing to do")
        return 0
    print(f"pending: {len(jobs)}")

    import os

    key = os.environ.get("SILICONFLOW_API_KEY") or ""
    if not key:
        print("ERROR: SILICONFLOW_API_KEY unset")
        return 1

    t0 = time.time()
    n = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for m, topic, seed in jobs:
            if time.time() - t0 > args.budget_seconds:
                print(f"budget exhausted after {n}; re-run to resume")
                break
            text, elapsed = gen_api(key, m, prompt(topic))
            if len(text) < 300:
                print(f"  WARN short {m.split('/')[-1]} {topic}/s{seed}: {len(text)}c")
            rec = {
                "id": hashlib.sha1(f"{m}|{topic}|{seed}".encode()).hexdigest()[:10],
                "topic_id": topic,
                "register": "literary",
                "arm": "L",
                "model": m,
                "seed": seed,
                "text": text,
                "char_len": len(text),
                "gen_seconds": round(elapsed, 1),
                "spec_version": "literary-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            print(f"  {m.split('/')[-1]:<16} {topic}/s{seed} {len(text)}c {elapsed:.0f}s")
    print(f"done: {n} new -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
