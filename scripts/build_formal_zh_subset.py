"""Build a proxy formal Chinese subset from existing processed splits.

This creates a dedicated dataset directory for a formal_zh adaptation run without
touching the default processed dataset. The proxy definition is intentionally
conservative: keep only zh records from finance and healthcare, then balance AI
and human samples within each domain per split.

Usage:
  uv run python scripts/build_formal_zh_subset.py
  uv run python scripts/build_formal_zh_subset.py --output-dir dataset/formal_zh_proxy
  uv run python scripts/build_formal_zh_subset.py --domains finance healthcare
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

DEFAULT_DOMAINS = ("finance", "healthcare")
DEFAULT_SPLITS = ("train", "val", "test")
SEED = 42


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _filter_records(records: list[dict], domains: set[str]) -> list[dict]:
    return [
        r for r in records if r.get("lang") == "zh" and r.get("domain") in domains and r.get("label") in {"human", "ai"}
    ]


def _balance_by_domain(records: list[dict], seed: int) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"human": [], "ai": []})
    for record in records:
        grouped[str(record["domain"])][str(record["label"])].append(record)

    balanced: list[dict] = []
    summary: dict[str, dict[str, int]] = {}
    for domain in sorted(grouped):
        human_records = grouped[domain]["human"]
        ai_records = grouped[domain]["ai"]
        keep = min(len(human_records), len(ai_records))
        if keep == 0:
            continue
        rng.shuffle(human_records)
        rng.shuffle(ai_records)
        chosen = human_records[:keep] + ai_records[:keep]
        rng.shuffle(chosen)
        balanced.extend(chosen)
        summary[domain] = {
            "human_kept": keep,
            "ai_kept": keep,
            "human_available": len(human_records),
            "ai_available": len(ai_records),
        }

    rng.shuffle(balanced)
    return balanced, summary


def _count_by_domain_label(records: list[dict]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for record in records:
        counter[(str(record.get("domain", "?")), str(record.get("label", "?")))] += 1
    return counter


def build_subset(input_dir: Path, output_dir: Path, domains: list[str], seed: int) -> None:
    domain_set = set(domains)
    metadata: dict[str, object] = {
        "type": "formal_zh_proxy_subset",
        "domains": sorted(domain_set),
        "seed": seed,
        "splits": {},
    }

    summary_table = Table(title="formal_zh proxy subset")
    summary_table.add_column("Split", style="cyan")
    summary_table.add_column("Input", justify="right")
    summary_table.add_column("Filtered", justify="right")
    summary_table.add_column("Output", justify="right")
    summary_table.add_column("Breakdown", style="dim")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in DEFAULT_SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split: {input_path}")

        input_records = _load_records(input_path)
        filtered_records = _filter_records(input_records, domain_set)
        balanced_records, domain_summary = _balance_by_domain(filtered_records, seed=seed)
        output_path = output_dir / f"{split}.jsonl"
        _write_records(output_path, balanced_records)

        breakdown = _count_by_domain_label(balanced_records)
        summary_text = ", ".join(f"{domain}/{label}={count}" for (domain, label), count in sorted(breakdown.items()))
        summary_table.add_row(
            split,
            str(len(input_records)),
            str(len(filtered_records)),
            str(len(balanced_records)),
            summary_text or "-",
        )

        metadata["splits"][split] = {
            "input_records": len(input_records),
            "filtered_records": len(filtered_records),
            "output_records": len(balanced_records),
            "domain_summary": domain_summary,
            "breakdown": {f"{domain}/{label}": count for (domain, label), count in sorted(breakdown.items())},
        }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(summary_table)
    console.print(f"[green]Wrote subset to {output_dir}[/]")
    console.print(f"[green]Metadata saved to {metadata_path}[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a proxy formal Chinese subset from processed splits")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.dataset_dir / "processed",
        help="Directory containing train/val/test JSONL splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.dataset_dir / "formal_zh_proxy",
        help="Output directory for the dedicated subset",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        help="Professional/formal proxy domains to keep (default: finance healthcare)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for balanced sampling")
    args = parser.parse_args()

    build_subset(input_dir=args.input_dir, output_dir=args.output_dir, domains=args.domains, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
