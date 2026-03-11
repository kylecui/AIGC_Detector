"""End-to-end dataset generation pipeline.

Orchestrates: crawl human text → generate AI text → process → split.

Steps:
  1. Collect human text (HC3 + Wikipedia)
  2. Process & deduplicate human text
  3. Generate AI text using local LLMs (one model at a time)
  4. Process & deduplicate AI text
  5. Merge human + AI records
  6. Split into train / val / test (80/10/10)

Usage:
  uv run python scripts/generate_dataset.py --step all
  uv run python scripts/generate_dataset.py --step crawl
  uv run python scripts/generate_dataset.py --step generate --models qwen2.5-7b chatglm3-6b
  uv run python scripts/generate_dataset.py --step process
  uv run python scripts/generate_dataset.py --step split
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure src is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.aigc_detector.config import settings
from src.aigc_detector.data.crawler import collect_human_texts
from src.aigc_detector.data.generator import generate_all
from src.aigc_detector.data.processor import process_records
from src.aigc_detector.data.splitter import print_split_stats, split_dataset

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = settings.dataset_dir / "raw"
PROCESSED_DIR = settings.dataset_dir / "processed"

HUMAN_RAW = RAW_DIR / "human_raw.jsonl"
AI_RAW = RAW_DIR / "ai_raw.jsonl"
HUMAN_PROCESSED = PROCESSED_DIR / "human_processed.jsonl"
AI_PROCESSED = PROCESSED_DIR / "ai_processed.jsonl"
MERGED = PROCESSED_DIR / "merged.jsonl"


def _count_jsonl(path: Path) -> dict[str, int]:
    """Count records in a JSONL file, grouped by label and lang."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            label = record.get("label", "?")
            lang = record.get("lang", "?")
            key = f"{label}/{lang}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def print_dataset_status() -> None:
    """Print overview of all dataset files and their record counts."""
    table = Table(title="Dataset Status")
    table.add_column("File", style="cyan")
    table.add_column("Exists", style="green")
    table.add_column("Records", style="magenta")
    table.add_column("Breakdown", style="dim")

    for path in [HUMAN_RAW, AI_RAW, HUMAN_PROCESSED, AI_PROCESSED, MERGED]:
        exists = path.exists()
        counts = _count_jsonl(path) if exists else {}
        total = sum(counts.values())
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        table.add_row(
            str(path.relative_to(settings.dataset_dir)),
            "✓" if exists else "✗",
            str(total) if exists else "-",
            breakdown if breakdown else "-",
        )

    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = PROCESSED_DIR / split
        exists = path.exists()
        counts = _count_jsonl(path) if exists else {}
        total = sum(counts.values())
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        table.add_row(
            str(path.relative_to(settings.dataset_dir)),
            "✓" if exists else "✗",
            str(total) if exists else "-",
            breakdown if breakdown else "-",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def step_crawl(num_wiki: int = 15000, no_hc3: bool = False) -> None:
    """Step 1: Collect human text from Wikipedia + HC3."""
    console.rule("[bold]Step 1: Crawl Human Text")

    if HUMAN_RAW.exists():
        counts = _count_jsonl(HUMAN_RAW)
        total = sum(counts.values())
        console.print(f"[yellow]human_raw.jsonl already exists with {total} records.[/]")
        console.print("[yellow]Delete it to re-crawl, or skip to the next step.[/]")
        return

    asyncio.run(
        collect_human_texts(
            output_dir=RAW_DIR,
            num_wiki_per_lang=num_wiki,
            hc3=not no_hc3,
        )
    )

    counts = _count_jsonl(HUMAN_RAW)
    total = sum(counts.values())
    console.print(f"[bold green]Crawl complete:[/] {total} records written to {HUMAN_RAW}")


def step_generate(
    models: list[str] | None = None,
    num_per_prompt: int = 50,
) -> None:
    """Step 2: Generate AI text using local LLMs."""
    console.rule("[bold]Step 2: Generate AI Text")

    if AI_RAW.exists():
        counts = _count_jsonl(AI_RAW)
        total = sum(counts.values())
        console.print(f"[yellow]ai_raw.jsonl already exists with {total} records.[/]")
        console.print("[yellow]Delete it to re-generate, or skip to the next step.[/]")
        return

    generate_all(
        output_dir=RAW_DIR,
        num_per_prompt=num_per_prompt,
        models=models,
        prompt_config_path="configs/prompts.yaml",
    )

    counts = _count_jsonl(AI_RAW)
    total = sum(counts.values())
    console.print(f"[bold green]Generation complete:[/] {total} records written to {AI_RAW}")


def step_process() -> None:
    """Step 3: Process and deduplicate raw data."""
    console.rule("[bold]Step 3: Process & Deduplicate")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process human text
    if HUMAN_RAW.exists():
        if HUMAN_PROCESSED.exists():
            console.print(f"[yellow]{HUMAN_PROCESSED.name} already exists, skipping.[/]")
        else:
            console.print("[bold blue]Processing human text...[/]")
            process_records(
                input_path=HUMAN_RAW,
                output_path=HUMAN_PROCESSED,
                min_chars=200,
                max_chars=10000,
            )
    else:
        console.print("[red]human_raw.jsonl not found. Run --step crawl first.[/]")

    # Process AI text
    if AI_RAW.exists():
        if AI_PROCESSED.exists():
            console.print(f"[yellow]{AI_PROCESSED.name} already exists, skipping.[/]")
        else:
            console.print("[bold blue]Processing AI text...[/]")
            process_records(
                input_path=AI_RAW,
                output_path=AI_PROCESSED,
                min_chars=200,
                max_chars=10000,
            )
    else:
        console.print("[red]ai_raw.jsonl not found. Run --step generate first.[/]")


def step_merge() -> None:
    """Step 4: Merge processed human + AI data."""
    console.rule("[bold]Step 4: Merge Datasets")

    if MERGED.exists():
        counts = _count_jsonl(MERGED)
        total = sum(counts.values())
        console.print(f"[yellow]merged.jsonl already exists with {total} records.[/]")
        return

    all_records: list[dict] = []

    for path in [HUMAN_PROCESSED, AI_PROCESSED]:
        if not path.exists():
            console.print(f"[red]{path.name} not found. Run --step process first.[/]")
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    MERGED.parent.mkdir(parents=True, exist_ok=True)
    with open(MERGED, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = _count_jsonl(MERGED)
    total = sum(counts.values())
    console.print(f"[bold green]Merge complete:[/] {total} records written to {MERGED}")
    for k, v in sorted(counts.items()):
        console.print(f"  {k}: {v}")


def step_split() -> None:
    """Step 5: Split into train/val/test."""
    console.rule("[bold]Step 5: Train/Val/Test Split")

    if not MERGED.exists():
        console.print("[red]merged.jsonl not found. Run --step merge first.[/]")
        return

    # Check if splits already exist
    train_path = PROCESSED_DIR / "train.jsonl"
    if train_path.exists():
        console.print("[yellow]Split files already exist. Delete them to re-split.[/]")
        return

    stats = split_dataset(
        input_path=MERGED,
        output_dir=PROCESSED_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        stratify_by=["label", "lang"],
    )

    print_split_stats(stats)
    console.print(f"[bold green]Split complete.[/] Files written to {PROCESSED_DIR}")


def step_all(
    num_wiki: int = 15000,
    no_hc3: bool = False,
    models: list[str] | None = None,
    num_per_prompt: int = 50,
) -> None:
    """Run all steps in sequence."""
    step_crawl(num_wiki=num_wiki, no_hc3=no_hc3)
    step_generate(models=models, num_per_prompt=num_per_prompt)
    step_process()
    step_merge()
    step_split()

    console.rule("[bold green]Pipeline Complete")
    print_dataset_status()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end dataset generation pipeline for AIGC Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["all", "crawl", "generate", "process", "merge", "split", "status"],
        default="status",
        help="Pipeline step to run (default: status)",
    )
    parser.add_argument(
        "--num-wiki",
        type=int,
        default=15000,
        help="Wikipedia articles per language (default: 15000)",
    )
    parser.add_argument(
        "--no-hc3",
        action="store_true",
        help="Skip HC3 dataset loading",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific generation model keys (default: all BnB-compatible models)",
    )
    parser.add_argument(
        "--num-per-prompt",
        type=int,
        default=50,
        help="Generations per prompt template (default: 50)",
    )

    args = parser.parse_args()

    if args.step == "status":
        print_dataset_status()
    elif args.step == "crawl":
        step_crawl(num_wiki=args.num_wiki, no_hc3=args.no_hc3)
    elif args.step == "generate":
        step_generate(models=args.models, num_per_prompt=args.num_per_prompt)
    elif args.step == "process":
        step_process()
    elif args.step == "merge":
        step_merge()
    elif args.step == "split":
        step_split()
    elif args.step == "all":
        step_all(
            num_wiki=args.num_wiki,
            no_hc3=args.no_hc3,
            models=args.models,
            num_per_prompt=args.num_per_prompt,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
