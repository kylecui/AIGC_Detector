"""W4-EN: English mirror of the W4c 2x2 (register x prompt x 5 models x 5 seeds).

Answers: do the specification-slack findings (contract evasion on formal
register, bidirectional register shortcut) transfer to English — through a
DETECTION PIPELINE that is architecturally different on the EN path
(ensemble weights: linguistic .85 / statistical .15, encoder/binoculars 0)?

EN-native topics (no translations). Formal types mirror the zh families
(correction/apology/clarification/commitment/incident report/announcement);
casual = reddit/yelp/x-style posts.

Checkpoint: model|topic|arm|seed in dataset/paired_generation_v1/w4en_records.jsonl
Usage mirrors w4c_generation.py: --part api-formal | api-casual | local | all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import paired_generation_experiment as _pge  # noqa: E402
from paired_generation_experiment import (  # noqa: E402
    gen_api,
    gen_local,
    load_local_model,
)

OUT_DIR = Path("dataset/paired_generation_v1")
RECORDS = OUT_DIR / "w4en_records.jsonl"
LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
API_MODELS = ["Qwen/Qwen3-8B", "THUDM/GLM-4-9B-0414", "Qwen/Qwen3-14B",
              "deepseek-ai/DeepSeek-V3.2"]
SEEDS = [11, 22, 33, 44, 55]

EN_SYSTEM_PROMPT = (
    "You are a professional writing assistant familiar with Anglo-American "
    "business and institutional document conventions."
)

# W4-EN mechanical wiring: gen_api/gen_local read the module-level zh system
# prompt at call time; rebind it to the authored EN prompt so every EN record
# is generated under English instructions (fixes zh-text contamination).
_pge.SYSTEM_PROMPT = EN_SYSTEM_PROMPT

FORMAL_TOPICS = [
    {"id": "f01", "doc_type": "customer apology statement",
     "subject": "a nationwide shipping delay affecting holiday orders", "org": "a consumer electronics retailer"},
    {"id": "f02", "doc_type": "product recall notice",
     "subject": "a safety defect in a children's building-block toy line", "org": "a toy manufacturer"},
    {"id": "f03", "doc_type": "correction announcement",
     "subject": "errors in previously released quarterly revenue figures", "org": "a logistics company"},
    {"id": "f04", "doc_type": "clarification statement",
     "subject": "media reports alleging a customer data breach", "org": "a regional bank"},
    {"id": "f05", "doc_type": "privacy commitment letter",
     "subject": "new data-handling safeguards for a mobile application", "org": "a software company"},
    {"id": "f06", "doc_type": "service incident report",
     "subject": "a six-hour payment-platform outage and its remediation", "org": "a fintech provider"},
    {"id": "f07", "doc_type": "supplier compliance commitment",
     "subject": "labor and safety standards across a garment supply chain", "org": "an apparel brand"},
    {"id": "f08", "doc_type": "laboratory safety incident report",
     "subject": "a minor chemical spill and revised protocols", "org": "a university chemistry department"},
    {"id": "f09", "doc_type": "environmental compliance pledge",
     "subject": "emission-reduction targets for a processing plant", "org": "an industrial manufacturer"},
    {"id": "f10", "doc_type": "academic integrity commitment",
     "subject": "examination conduct for a professional certification program", "org": "a certification body"},
    {"id": "f11", "doc_type": "public event disclaimer",
     "subject": "participation risks for a city marathon", "org": "an event management company"},
    {"id": "f12", "doc_type": "trademark authorization statement",
     "subject": "licensed use of a registered brand mark by an affiliate", "org": "a hospitality group"},
    {"id": "f13", "doc_type": "office relocation notice",
     "subject": "headquarters move and business-continuity arrangements", "org": "an insurance company"},
    {"id": "f14", "doc_type": "refund policy announcement",
     "subject": "compensation for a cancelled subscription service", "org": "a streaming platform"},
    {"id": "f15", "doc_type": "security disclosure statement",
     "subject": "a responsibly disclosed vulnerability and its patch timeline",
     "org": "an open-source software foundation"},
    {"id": "f16", "doc_type": "partnership termination statement",
     "subject": "ending a co-branding collaboration with an external agency", "org": "a beverage company"},
    {"id": "f17", "doc_type": "quality assurance commitment",
     "subject": "warranty terms for an infrastructure project", "org": "a construction firm"},
    {"id": "f18", "doc_type": "volunteer code-of-conduct pledge",
     "subject": "conduct and confidentiality at an international expo", "org": "a volunteer association"},
    {"id": "f19", "doc_type": "service disruption explanation",
     "subject": "repeated clinic appointment system failures and remedies", "org": "a healthcare network"},
    {"id": "f20", "doc_type": "investor relations clarification",
     "subject": "misreadings of forward guidance in press coverage", "org": "a renewable energy company"},
]

CASUAL_TOPICS = [
    {"id": "g01", "platform": "reddit r/LifeProTips", "subject": "a cheap trick that keeps a tiny apartment clean"},
    {"id": "g02", "platform": "Yelp review", "subject": "a new ramen place with absurd portions"},
    {"id": "g03", "platform": "X/Twitter", "subject": "the commute from hell this morning"},
    {"id": "g04", "platform": "reddit r/AskReddit", "subject": "which hobby actually stuck past the honeymoon phase"},
    {"id": "g05", "platform": "Amazon review", "subject": "a budget mechanical keyboard after three months"},
    {"id": "g06", "platform": "reddit r/movies", "subject": "an underrated documentary everyone slept on"},
    {"id": "g07", "platform": "X/Twitter", "subject": "a software update that wrecked a workflow"},
    {"id": "g08", "platform": "reddit r/CasualConversation", "subject": "the neighbor who keeps rescuing stray cats"},
    {"id": "g09", "platform": "personal blog", "subject": "first-season balcony gardening disasters"},
    {"id": "g10", "platform": "reddit r/Fitness", "subject": "gym anxiety as a total beginner"},
    {"id": "g11", "platform": "Yelp review", "subject": "a brunch spot with a two-hour wait"},
    {"id": "g12", "platform": "reddit r/Gaming", "subject": "the shame of a 200-game backlog"},
    {"id": "g13", "platform": "X/Twitter", "subject": "email culture needs to chill"},
    {"id": "g14", "platform": "reddit r/badroommates", "subject": "the great thermostat war"},
    {"id": "g15", "platform": "reddit r/travel", "subject": "a hidden-gem day trip two hours from the city"},
    {"id": "g16", "platform": "reddit r/Cooking", "subject": "a five-ingredient dinner that slaps"},
    {"id": "g17", "platform": "X/Twitter", "subject": "adopting the world's most chaotic cat"},
    {"id": "g18", "platform": "reddit r/photography", "subject": "phone photos that rival real cameras"},
    {"id": "g19", "platform": "Google review", "subject": "a bookstore-cafe hybrid with perfect vibes"},
    {"id": "g20", "platform": "reddit r/productivity", "subject": "the two-minute habit that fixed a morning"},
]


def formal_free_prompt(t: dict) -> str:
    return (
        f"Write a {t['doc_type']} issued by {t['org']} regarding {t['subject']}. "
        "Complete, formal professional register, around 550 words. Output the document text directly."
    )


def formal_contract_prompt(t: dict) -> str:
    return (
        f"Write a {t['doc_type']} for {t['org']} regarding {t['subject']}, "
        "following this specification EXACTLY.\n"
        "\n"
        "[STRUCTURE — in this order, every element required]\n"
        '1. Title line: "REGARDING: <short subject phrase>" (all caps prefix)\n'
        '2. Addressee line beginning "To:"\n'
        '3. Opening paragraph beginning "We are writing to"\n'
        "4. An information block of at least 4 lines, each formatted "
        '"Field: value" (e.g. Issuer / Scope / Effective date / Contact)\n'
        "5. Body clauses numbered with Roman numerals (I., II., III., IV.), "
        'each beginning with a topic keyword (e.g. "I. Commitments:")\n'
        '6. A closing paragraph that is exactly "We hereby affirm the foregoing." '
        "followed by one more sentence\n"
        '7. Signature block: organization name, then "Date: ____"\n'
        "\n"
        "[LANGUAGE CONSTRAINTS]\n"
        "- Formal institutional register; include at least 3 of: "
        '"pursuant to", "hereby", "in accordance with", "shall", "undertake"\n'
        "- No contractions, no exclamations, no colloquialisms, no meta-commentary\n"
        "\n"
        "[LENGTH] 500-650 words total including the signature block.\n"
        "\n"
        "[FORMAT] No Markdown symbols (#, *, -); plain text only; output the document directly."
    )


def casual_free_prompt(t: dict) -> str:
    return (
        f"Write a {t['platform']} post about {t['subject']}. "
        "Natural, casual, like a real user dashing it off, around 200 words. Output the post text directly."
    )


def casual_contract_prompt(t: dict) -> str:
    return f"""Write a {t['platform']} post about {t['subject']}, following this specification EXACTLY.

[STRUCTURE — in this order, every element required]
1. Hook opening: one sentence that creates curiosity or relatability (never states the topic flatly)
2. A list of exactly 4 items, each starting with an emoji, each under 25 words
3. A transition paragraph starting "Real talk:" (max 30 words)
4. A closing question that invites replies

[LANGUAGE CONSTRAINTS]
- Casual internet voice; include at least 2 of: "ngl", "tbh", "lowkey", "literally", "I can't"
- No corporate tone, no "firstly/secondly", at most 2 exclamation marks total

[LENGTH] 130-180 words.

[FORMAT] No Markdown headers or bold; plain text; output the post directly."""


def done_keys() -> set[str]:
    keys: set[str] = set()
    if RECORDS.exists():
        for line in RECORDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add(f"{r['model']}|{r['topic_id']}|{r['arm']}|{r['seed']}")
    return keys


def build_jobs(
    part: str, models: list[str] | None = None, seeds: list[int] | None = None
) -> list[tuple[str, dict, str, int]]:
    dk = done_keys()
    use_models = models or (API_MODELS + [LOCAL_MODEL])
    use_seeds = seeds or SEEDS
    jobs: list[tuple[str, dict, str, int]] = []

    def add(model: str, topics: list[dict], arms: list[str]):
        for t in topics:
            for arm in arms:
                for seed in use_seeds:
                    if f"{model}|{t['id']}|{arm}|{seed}" not in dk:
                        jobs.append((model, t, arm, seed))

    if part in ("api-formal", "all"):
        for m in use_models:
            if m != LOCAL_MODEL:
                add(m, FORMAL_TOPICS, ["A", "B"])
    if part in ("api-casual", "all"):
        for m in use_models:
            if m != LOCAL_MODEL:
                add(m, CASUAL_TOPICS, ["C", "D"])
    if part in ("local", "all"):
        if LOCAL_MODEL in use_models:
            add(LOCAL_MODEL, FORMAL_TOPICS, ["A", "B"])
            add(LOCAL_MODEL, CASUAL_TOPICS, ["C", "D"])
    return jobs


PROMPTS = {
    "A": formal_free_prompt, "B": formal_contract_prompt,
    "C": casual_free_prompt, "D": casual_contract_prompt,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["api-formal", "api-casual", "local", "all"])
    ap.add_argument("--budget-seconds", type=int, default=500)
    ap.add_argument("--models", default="",
                    help="comma-separated model IDs to use (subset); default all")
    ap.add_argument("--seeds", default="",
                    help="comma-separated seeds (subset); default all 5")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m] or None
    seeds = [int(s) for s in args.seeds.split(",") if s] or None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.part, models, seeds)
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

    t0 = time.time()
    n_done = 0
    with RECORDS.open("a", encoding="utf-8") as fh:
        for model_id, t, arm, seed in jobs:
            if time.time() - t0 > args.budget_seconds:
                print(f"budget exhausted after {n_done}; re-run to resume")
                break
            prompt = PROMPTS[arm](t)
            if model_id == LOCAL_MODEL:
                import torch

                torch.manual_seed(seed)
                text, elapsed = gen_local(local_model, local_tok, prompt)
            else:
                text, elapsed = gen_api(api_key, model_id, prompt)
            if len(text) < 80:
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
                "lang": "en",
                "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10],
                "text": text,
                "char_len": len(text),
                "gen_seconds": round(elapsed, 1),
                "spec_version": "w4en-2x2-v1",
                "created_at": datetime.now(UTC).isoformat(),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            print(f"  {model_id.split('/')[-1]:<18} {t['id']}/{arm}/s{seed} {len(text)}c {elapsed:.0f}s")
    print(f"done: {n_done} new -> {RECORDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
