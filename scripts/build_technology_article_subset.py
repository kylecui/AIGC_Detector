"""Derive technology_article_zh domain assets from the professional_zh corpus."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

SOURCE = Path("dataset/seeds/professional_zh/seed_samples_v1.jsonl")
SEED_DIR = Path("dataset/seeds/technology_article_zh")
ADAPT_DIR = Path("dataset/technology_article_zh_adaptation_v1")
SEED = 42


def load_domain_records(path: Path, domain: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang") == "zh" and row.get("label") in {"ai", "human"} and row.get("domain") == domain:
                rows.append(row)
    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[record["label"]].append(record)
    for label_records in buckets.values():
        rng.shuffle(label_records)

    splits = {"train": [], "val": [], "test": []}
    for label, label_records in buckets.items():
        n = len(label_records)
        test_n = max(1, round(n * 0.2))
        val_n = max(1, round(n * 0.2))
        if test_n + val_n >= n:
            test_n = 1
            val_n = 1
        train_n = n - test_n - val_n
        splits["train"].extend(label_records[:train_n])
        splits["val"].extend(label_records[train_n : train_n + val_n])
        splits["test"].extend(label_records[train_n + val_n :])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def build(seed: int) -> None:
    records = load_domain_records(SOURCE, "technology_article")
    if not records:
        raise ValueError("No technology_article records found")

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEED_DIR / "seed_samples.jsonl", records)
    write_jsonl(SEED_DIR / "hard_case_eval_v1.jsonl", records)

    splits = stratified_split(records, seed)
    ADAPT_DIR.mkdir(parents=True, exist_ok=True)
    for split, split_records in splits.items():
        write_jsonl(ADAPT_DIR / f"{split}.jsonl", split_records)

    meta = {
        "type": "technology_article_zh_adaptation_subset",
        "source": str(SOURCE),
        "seed": seed,
        "seed_dir": str(SEED_DIR),
        "adapt_dir": str(ADAPT_DIR),
        "counts": {split: dict(Counter(r["label"] for r in split_records)) for split, split_records in splits.items()},
        "notes": [
            "Derived from the broader professional_zh corpus.",
            "This is a bootstrap dedicated domain pack for technology_article_zh.",
            "Use for local evaluation and later small-sample adaptation experiments.",
        ],
    }
    (ADAPT_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    table = Table(title="technology_article_zh domain pack")
    table.add_column("Artifact", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("AI", justify="right")
    table.add_column("Human", justify="right")

    seed_counts = Counter(r["label"] for r in records)
    table.add_row("seed_samples", str(len(records)), str(seed_counts.get("ai", 0)), str(seed_counts.get("human", 0)))
    for split, split_records in splits.items():
        counts = Counter(r["label"] for r in split_records)
        table.add_row(split, str(len(split_records)), str(counts.get("ai", 0)), str(counts.get("human", 0)))
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build technology_article_zh domain assets")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
