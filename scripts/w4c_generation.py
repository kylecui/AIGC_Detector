"""W4c: the decisive 2x2 — register (formal/casual) x prompt (free/contract).

Open question #3 (plan v2.1 §6): is the W4 capability gate CONTRACT-specific
or FORMALITY-specific? The missing cell answers it:

    casual x contract (cell D): strict output-contract on CASUAL content
    (social-media style posts with structure/emoji/length/tone constraints).
    - If D shows no evasion vs C -> formality is a necessary co-factor
      (Perplexity Trap C1-C3 interpretation holds)
    - If D shows evasion like B  -> the contract alone drives statistical
      collapse (register-independent)

Design: 20 casual topics x {C free, D contract} x 5 models x 5 seeds.
Models (harness-validated capability ladder):
  Qwen2.5-7B (below-floor, local) / Qwen3-8B / GLM-4-9B / Qwen3-14B /
  DeepSeek-V3.2 (SiliconFlow).
Formal arms (A/B) get seed-replication on the same 5 models over replication
topics t41-t60 for the dose-response analysis.

Checkpoint key: model|topic|arm|seed  (resume-safe; appends to same JSONL).
Usage:
  uv run python scripts/w4c_generation.py --part api-casual   [--budget-seconds 500]
  uv run python scripts/w4c_generation.py --part api-formal   [--budget-seconds 500]
  uv run python scripts/w4c_generation.py --part local        [--budget-seconds 500]
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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from paired_generation_experiment import (  # noqa: E402
    SYSTEM_PROMPT,
    gen_api,
    gen_local,
    load_local_model,
)

OUT_DIR = Path("dataset/paired_generation_v1")
RECORDS = OUT_DIR / "w4c_records.jsonl"
LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
API_MODELS = ["Qwen/Qwen3-8B", "THUDM/GLM-4-9B-0414", "Qwen/Qwen3-14B",
              "deepseek-ai/DeepSeek-V3.2"]
SEEDS = [11, 22, 33, 44, 55]

# Formal replication topics (reuse t41-t60 from scripts/topics_replication.json
# — loaded at runtime; fallback error if missing).

CASUAL_TOPICS = [
    {"id": "c01", "platform": "小红书", "subject": "第一次一个人租房的踩坑经验", "tag": "生活分享"},
    {"id": "c02", "platform": "微博", "subject": "吐槽早高峰地铁被挤成沙丁鱼", "tag": "日常吐槽"},
    {"id": "c03", "platform": "大众点评", "subject": "一家新开的川菜馆探店体验", "tag": "探店测评"},
    {"id": "c04", "platform": "小红书", "subject": "百元以内提升幸福感的好物推荐", "tag": "好物分享"},
    {"id": "c05", "platform": "知乎", "subject": "回答「坚持晨跑半年是什么体验」", "tag": "经验回答"},
    {"id": "c06", "platform": "豆瓣小组", "subject": "安利一部冷门但后劲很大的纪录片", "tag": "影视安利"},
    {"id": "c07", "platform": "微博", "subject": "看完一场livehouse演出后的激动碎碎念", "tag": "现场感受"},
    {"id": "c08", "platform": "小红书", "subject": "通勤半小时内的快速早餐方案", "tag": "生活妙招"},
    {"id": "c09", "platform": "虎扑", "subject": "聊聊昨晚那场绝杀球赛的心路历程", "tag": "体育讨论"},
    {"id": "c10", "platform": "知乎", "subject": "回答「养猫之后生活发生了哪些变化」", "tag": "养宠体验"},
    {"id": "c11", "platform": "小红书", "subject": "敏感肌换季护肤踩雷与自救", "tag": "护肤心得"},
    {"id": "c12", "platform": "大众点评", "subject": "一家老字号早餐店的怀旧味道", "tag": "探店怀旧"},
    {"id": "c13", "platform": "微博", "subject": "加班到深夜回家路上的心情记录", "tag": "情绪随笔"},
    {"id": "c14", "platform": "知乎", "subject": "回答「毕业三年你的同学都混得怎么样」", "tag": "人生观察"},
    {"id": "c15", "platform": "小红书", "subject": "周末城市周边两日游路线分享", "tag": "旅行攻略"},
    {"id": "c16", "platform": "豆瓣", "subject": "吐槽一部高开低走的电视剧大结局", "tag": "影视吐槽"},
    {"id": "c17", "platform": "虎扑", "subject": "安利一个坚持多年的小众爱好", "tag": "爱好安利"},
    {"id": "c18", "platform": "微博", "subject": "第一次尝试露营翻车全记录", "tag": "翻车实录"},
    {"id": "c19", "platform": "知乎", "subject": "回答「有哪些越早知道越好的人生道理」", "tag": "人生感悟"},
    {"id": "c20", "platform": "小红书", "subject": "打工人的工位改造与桌面好物", "tag": "工位改造"},
]


def casual_free_prompt(t: dict) -> str:
    return (
        f"请写一篇{t['platform']}帖子，话题：{t['subject']}。"
        "语气自然随性，像真实用户随手发的，300字左右，直接输出正文。"
    )


def casual_contract_prompt(t: dict) -> str:
    return f"""请严格按照以下规格撰写一篇{t['platform']}帖子，话题：{t['subject']}。

【输出结构——按以下顺序，各项缺一不可】
1. 开头钩子：一句话点题，必须引发好奇或共鸣（禁止直接报话题名）
2. 分点列表：以"①②③④"编号，共4点，每点开头配一个emoji，每点不超过40字
3. 转折段：以"说实话"开头，写一段50字以内的真实感受或小吐槽
4. 互动结尾：以一个提问句收尾，引导评论

【语言约束】
- 必须使用口语化网络表达，出现下列至少2处："真的""绝了""谁懂啊""救命""狠狠""拿捏"
- 禁止书面语套话（"综上所述""首先其次"）；禁止感叹号超过2个
- 必须包含至少1处话题标签，格式为 #标签#

【长度约束】全文（含标签）在280至330字之间。

【格式禁止】不使用Markdown标题/加粗/分割线；直接输出正文。"""


FORMAL_CONTRACT_PROMPT_FALLBACK = None  # formal prompts loaded from replication runner


def formal_prompts():
    """Import formal prompts + topics from the main experiment module."""
    import paired_generation_experiment as pge

    return pge.freeform_prompt, pge.contract_prompt


def load_formal_topics() -> list[dict]:
    data = json.loads(Path("scripts/topics_replication.json").read_text(encoding="utf-8"))
    return [t for t in data if t["id"].startswith("t4") or t["id"].startswith("t5") or t["id"].startswith("t6")][:20]


def done_keys() -> set[str]:
    keys: set[str] = set()
    if RECORDS.exists():
        for line in RECORDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add(f"{r['model']}|{r['topic_id']}|{r['arm']}|{r['seed']}")
    return keys


def build_jobs(part: str) -> list[tuple[str, dict, str, int]]:
    dk = done_keys()
    jobs: list[tuple[str, dict, str, int]] = []

    def add(model: str, topics: list[dict], arms: dict[str, str]):
        for t in topics:
            for arm in arms:
                for seed in SEEDS:
                    if f"{model}|{t['id']}|{arm}|{seed}" not in dk:
                        jobs.append((model, t, arm, seed))

    if part in ("api-casual", "all"):
        for m in API_MODELS:
            add(m, CASUAL_TOPICS, {"C": "casual-free", "D": "casual-contract"})
    if part in ("api-formal", "all"):
        ft = load_formal_topics()
        for m in API_MODELS:
            add(m, ft, {"A": "formal-free", "B": "formal-contract"})
    if part in ("local", "all"):
        add(LOCAL_MODEL, CASUAL_TOPICS, {"C": "casual-free", "D": "casual-contract"})
        ft = load_formal_topics()
        add(LOCAL_MODEL, ft, {"A": "formal-free", "B": "formal-contract"})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["api-casual", "api-formal", "local", "all"])
    ap.add_argument("--budget-seconds", type=int, default=500)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.part)
    if not jobs:
        print("nothing to do")
        return 0
    print(f"pending: {len(jobs)} generations ({args.part})")

    local_model = local_tok = None
    if any(m == LOCAL_MODEL for m, _, _, _ in jobs):
        local_model, local_tok = load_local_model()
        print("local model loaded")

    api_key = ""
    if any(m != LOCAL_MODEL for m, _, _, _ in jobs):
        import os

        api_key = os.environ.get("SILICONFLOW_API_KEY") or ""
        if not api_key:
            print("ERROR: SILICONFLOW_API_KEY unset")
            return 1

    ff_prompt, fc_prompt = formal_prompts()
    t0 = time.time()
    n_done = 0
    with RECORDS.open("a", encoding="utf-8") as fh:
        for model_id, t, arm, seed in jobs:
            if time.time() - t0 > args.budget_seconds:
                print(f"budget exhausted after {n_done}; re-run to resume")
                break
            if arm in ("A", "B"):
                prompt = ff_prompt(t) if arm == "A" else fc_prompt(t)
            else:
                prompt = casual_free_prompt(t) if arm == "C" else casual_contract_prompt(t)
            if model_id == LOCAL_MODEL:
                import torch

                torch.manual_seed(seed)
                text, elapsed = gen_local(local_model, local_tok, prompt)
            else:
                text, elapsed = gen_api(api_key, model_id, prompt)  # server-side sampling
            if len(text) < 120:
                print(f"  WARN short {model_id.split('/')[-1]} {t['id']}/{arm}/s{seed}: {len(text)}c")
            rec = {
                "id": hashlib.sha1(
                    f"{t['id']}|{arm}|{model_id}|{seed}".encode()
                ).hexdigest()[:10],
                "topic_id": t["id"],
                "register": "formal" if arm in ("A", "B") else "casual",
                "arm": arm,
                "model": model_id,
                "seed": seed,
                "backend": "local" if model_id == LOCAL_MODEL else "siliconflow",
                "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10],
                "text": text,
                "char_len": len(text),
                "gen_seconds": round(elapsed, 1),
                "spec_version": "w4c-2x2-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            print(f"  {model_id.split('/')[-1]:<18} {t['id']}/{arm}/s{seed} {len(text)}c {elapsed:.0f}s")
    print(f"done: {n_done} new -> {RECORDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
