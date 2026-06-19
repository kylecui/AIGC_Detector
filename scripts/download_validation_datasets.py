"""Download and convert HuggingFace validation datasets to project JSONL schema.

Produces records of the form:
    {"id": str, "text": str, "label": "human"|"ai", "lang": "zh"|"en", "source": str}

Sources (chosen per librarian research, see .sisyphus/plans/upgrade-linguistic-detection.md):
- Rajarshi-Roy-research/Defactify_Text_Dataset  (English, 73k, pre-split, binary label)
- Hello-SimpleAI/HC3-Chinese                     (Chinese, 12.9k pairs, explode answers)
- ilyasoulk/ai-vs-human                          (English, 5.4k pairs, MIT, optional held-out)

Usage:
    # English: Defactify, full splits
    uv run python scripts/download_validation_datasets.py defactify \\
        --out-dir dataset/validation_en

    # Chinese: HC3-Chinese, explode answers
    uv run python scripts/download_validation_datasets.py hc3-zh \\
        --config all --out-dir dataset/validation_zh

    # English small held-out: ai-vs-human
    uv run python scripts/download_validation_datasets.py ai-vs-human \\
        --out-dir dataset/validation_en_heldout

Resume support: skips files that already exist with non-zero size.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Project text-length floor (matches data/processor.py convention).
MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 10_000


def _clean(text: str) -> str:
    """Normalize whitespace; return empty string if input is not usable text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _is_usable(text: str) -> bool:
    """Length + non-boilerplate filter."""
    if not text:
        return False
    if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS:
        return False
    return True


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ======================================================================
# Defactify (English)
# ======================================================================


def download_defactify(out_dir: Path, max_per_split: int | None = None) -> None:
    """English: 73k pre-split. Label_A 0=human, 1=ai. Label_B = model name."""
    from datasets import load_dataset

    logger.info("Loading Defactify_Text_Dataset (this may download a few hundred MB)...")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Text_Dataset")

    split_map = {"train": "train", "validation": "val", "test": "test"}
    out_dir.mkdir(parents=True, exist_ok=True)

    for hf_split, our_split in split_map.items():
        if hf_split not in ds:
            logger.warning("Defactify split '%s' not present, skipping", hf_split)
            continue

        out_path = out_dir / f"{our_split}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info("Skipping existing %s", out_path)
            continue

        rows = ds[hf_split]
        if max_per_split:
            rows = rows.select(range(min(max_per_split, len(rows))))

        records: list[dict[str, Any]] = []
        skipped = 0
        for i, row in enumerate(rows):
            text = _clean(row.get("Text", ""))
            if not _is_usable(text):
                skipped += 1
                continue
            label_a = row.get("Label_A")
            if label_a is None:
                skipped += 1
                continue
            label = "ai" if int(label_a) == 1 else "human"
            model_src = row.get("Label_B", "unknown")
            if label == "human":
                # Defactify uses "Human_Story" for human rows; normalize.
                model_src = "human"
            records.append(
                {
                    "id": f"def_{our_split}_{i:06d}",
                    "text": text,
                    "label": label,
                    "lang": "en",
                    "source": f"defactify/{model_src}",
                }
            )

        n_written = _write_jsonl(records, out_path)
        logger.info(
            "Defactify %s -> %s : %d records (%d skipped)",
            hf_split,
            out_path,
            n_written,
            skipped,
        )


# ======================================================================
# HC3-Chinese
# ======================================================================


def download_hc3_zh(
    out_dir: Path,
    config: str = "all",
    max_per_split: int | None = None,
    min_chars: int = 50,
) -> None:
    """Chinese: pair-format. Explode human_answers and chatgpt_answers lists.

    ``min_chars`` defaults to 50 (lower than project standard 200) because
    HC3 human answers are forum-style and naturally short; the standard
    200-char floor would drop ~all human samples and skew the label balance.
    """
    from datasets import load_dataset

    logger.info("Loading Hello-SimpleAI/HC3-Chinese config='%s' ...", config)
    # The HF dataset's cache only exposes 'default' as a config name on some
    # setups; try the user's config first and fall back to 'default' on cache
    # errors (which usually means the parquet snapshot is the merged view).
    try:
        ds = load_dataset("Hello-SimpleAI/HC3-Chinese", config)
    except ValueError as e:
        if "default" in str(e):
            logger.warning("Config '%s' not available in cache; using 'default'.", config)
            ds = load_dataset("Hello-SimpleAI/HC3-Chinese", "default")
        else:
            raise
    # HC3 has only a "train" split. We will partition ourselves: 80/10/10.

    if "train" not in ds:
        logger.warning("HC3-Chinese has no 'train' split; available: %s", list(ds.keys()))
        return

    rows = ds["train"]
    if max_per_split:
        rows = rows.select(range(min(max_per_split, len(rows))))

    human_records: list[dict[str, Any]] = []
    ai_records: list[dict[str, Any]] = []
    skipped = 0

    for i, row in enumerate(rows):
        source = row.get("source", "unknown")
        # Each row may have multiple human and chatgpt answers.
        human_answers = row.get("human_answers") or []
        chatgpt_answers = row.get("chatgpt_answers") or []

        for j, ans in enumerate(human_answers):
            text = _clean(ans)
            if not text or len(text) < min_chars or len(text) > MAX_TEXT_CHARS:
                skipped += 1
                continue
            human_records.append(
                {
                    "id": f"hc3_h_{i:06d}_{j:02d}",
                    "text": text,
                    "label": "human",
                    "lang": "zh",
                    "source": f"hc3-zh/{source}",
                }
            )

        for j, ans in enumerate(chatgpt_answers):
            text = _clean(ans)
            if not text or len(text) < min_chars or len(text) > MAX_TEXT_CHARS:
                skipped += 1
                continue
            ai_records.append(
                {
                    "id": f"hc3_a_{i:06d}_{j:02d}",
                    "text": text,
                    "label": "ai",
                    "lang": "zh",
                    "source": f"hc3-zh/{source}",
                }
            )

    # Shuffle with fixed seed for stable partitioning.
    import random

    rng = random.Random(42)
    rng.shuffle(human_records)
    rng.shuffle(ai_records)

    # Stratified 80/10/10 split: maintain label balance in each split.
    def _split(records: list[dict[str, Any]]) -> tuple[list, list, list]:
        n = len(records)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return records[:n_train], records[n_train : n_train + n_val], records[n_train + n_val :]

    h_train, h_val, h_test = _split(human_records)
    a_train, a_val, a_test = _split(ai_records)

    train = h_train + a_train
    val = h_val + a_val
    test = h_test + a_test
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("val", val), ("test", test)):
        out_path = out_dir / f"{name}.jsonl"
        n = _write_jsonl(recs, out_path)
        logger.info("HC3-zh %s -> %s : %d records", name, out_path, n)

    logger.info(
        "HC3-zh total: human=%d ai=%d (skipped=%d)",
        len(human_records),
        len(ai_records),
        skipped,
    )


# ======================================================================
# ai-vs-human (English small held-out)
# ======================================================================


def download_ai_vs_human(out_dir: Path, max_rows: int | None = None) -> None:
    """English: paired columns 'human' and 'ai'. MIT license."""
    from datasets import load_dataset

    logger.info("Loading ilyasoulk/ai-vs-human ...")
    ds = load_dataset("ilyasoulk/ai-vs-human", split="train")

    rows = ds if max_rows is None else ds.select(range(min(max_rows, len(ds))))

    records: list[dict[str, Any]] = []
    skipped = 0
    for i, row in enumerate(rows):
        for label, key in (("human", "human"), ("ai", "ai")):
            text = _clean(row.get(key, ""))
            if not _is_usable(text):
                skipped += 1
                continue
            records.append(
                {
                    "id": f"aih_{label[0]}_{i:06d}",
                    "text": text,
                    "label": label,
                    "lang": "en",
                    "source": "ai-vs-human",
                }
            )

    out_path = out_dir / "test.jsonl"
    n = _write_jsonl(records, out_path)
    logger.info("ai-vs-human -> %s : %d records (%d skipped)", out_path, n, skipped)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert HuggingFace validation datasets to project JSONL schema.",
    )
    sub = parser.add_subparsers(dest="source", required=True)

    p_en = sub.add_parser("defactify", help="English Defactify (73k pre-split).")
    p_en.add_argument("--out-dir", type=Path, required=True)
    p_en.add_argument("--max-per-split", type=int, default=None, help="Cap rows per split (debugging).")

    p_zh = sub.add_parser("hc3-zh", help="Chinese HC3-Chinese (pair-exploded).")
    p_zh.add_argument("--config", default="all", help="HC3 config: all/baike/finance/law/medicine/open_qa/psychology")
    p_zh.add_argument("--out-dir", type=Path, required=True)
    p_zh.add_argument("--max-per-split", type=int, default=None)
    p_zh.add_argument("--min-chars", type=int, default=50, help="Min chars per answer (HC3 default 50).")

    p_heldout = sub.add_parser("ai-vs-human", help="English small held-out (MIT).")
    p_heldout.add_argument("--out-dir", type=Path, required=True)
    p_heldout.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()

    if args.source == "defactify":
        download_defactify(args.out_dir, max_per_split=args.max_per_split)
    elif args.source == "hc3-zh":
        download_hc3_zh(args.out_dir, config=args.config, max_per_split=args.max_per_split, min_chars=args.min_chars)
    elif args.source == "ai-vs-human":
        download_ai_vs_human(args.out_dir, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
