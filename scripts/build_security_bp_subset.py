"""Build a tiny LoRA-ready adaptation subset for security_bp_zh.

This combines the current seed and hard-case files, keeps only zh binary-labeled
records, then writes reproducible train/val/test splits into a dedicated subset
directory. The output uses the same JSONL shape expected by train_cloud.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.aigc_detector.config import settings

console = Console()

DEFAULT_INPUTS = (
    Path("dataset/seeds/security_bp_zh/seed_samples.jsonl"),
    Path("dataset/seeds/security_bp_zh/hard_case_eval_v1.jsonl"),
)
DEFAULT_OUTPUT = Path("dataset/security_bp_zh_adaptation_v1")
SEED = 42


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen_ids: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("lang") != "zh":
                    continue
                if record.get("label") not in {"human", "ai"}:
                    continue
                rid = str(record.get("id", ""))
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                records.append(record)
    return records


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
        if n < 3:
            raise ValueError(f"Need at least 3 records for label={label}, got {n}")

        test_n = max(1, round(n * 0.2))
        val_n = max(1, round(n * 0.2))
        if test_n + val_n >= n:
            test_n = 1
            val_n = 1
        train_n = n - test_n - val_n
        if train_n < 1:
            raise ValueError(f"Not enough records to create train split for label={label}")

        splits["train"].extend(label_records[:train_n])
        splits["val"].extend(label_records[train_n : train_n + val_n])
        splits["test"].extend(label_records[train_n + val_n :])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_splits(splits: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    table = Table(title="security_bp_zh adaptation subset")
    table.add_column("Split", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("AI", justify="right")
    table.add_column("Human", justify="right")
    table.add_column("Subtypes", style="dim")

    for split, records in splits.items():
        counts = Counter(r["label"] for r in records)
        subtype_counts = Counter(r.get("subtype", "seed") for r in records)
        summary[split] = {
            "total": len(records),
            "ai": counts.get("ai", 0),
            "human": counts.get("human", 0),
        }
        subtype_text = ", ".join(f"{k}={v}" for k, v in sorted(subtype_counts.items()))
        table.add_row(
            split,
            str(len(records)),
            str(counts.get("ai", 0)),
            str(counts.get("human", 0)),
            subtype_text,
        )
    console.print(table)
    return summary


def build_subset(input_paths: list[Path], output_dir: Path, seed: int) -> None:
    records = load_records(input_paths)
    if not records:
        raise ValueError("No usable records found for security_bp_zh subset")

    splits = stratified_split(records, seed=seed)
    for split, split_records in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_records)

    summary = summarize_splits(splits)
    metadata = {
        "type": "security_bp_zh_adaptation_subset",
        "seed": seed,
        "input_files": [str(p) for p in input_paths],
        "summary": summary,
        "notes": [
            "Tiny LoRA-ready adaptation subset for security_bp_zh.",
            "Built from manual seeds and hard cases; not a production training corpus.",
            "Use for controlled adaptation experiments and error analysis.",
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote subset to {output_dir}[/]")
    console.print(f"[green]Metadata saved to {metadata_path}[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tiny security_bp_zh adaptation subset")
    parser.add_argument(
        "--input",
        nargs="*",
        default=[str(p) for p in DEFAULT_INPUTS],
        help="Input JSONL files to merge before splitting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for train/val/test JSONL files",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    build_subset(input_paths=input_paths, output_dir=args.output_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
