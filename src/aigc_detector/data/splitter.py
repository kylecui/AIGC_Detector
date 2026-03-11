"""Dataset splitter for creating train/val/test splits with stratification."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

console = Console()


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by: list[str] | None = None,
) -> dict:
    """
    Split dataset into train/val/test with optional stratification.

    Args:
        input_path: Path to input JSONL file.
        output_dir: Path to output directory.
        train_ratio: Proportion for training set (default: 0.8).
        val_ratio: Proportion for validation set (default: 0.1).
        test_ratio: Proportion for test set (default: 0.1).
        seed: Random seed for reproducibility (default: 42).
        stratify_by: List of columns to stratify by (default: ["label", "lang"]).

    Returns:
        Dictionary with split statistics including counts and distributions.
    """
    if stratify_by is None:
        stratify_by = ["label", "lang"]

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio}. train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    # Read JSONL file
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {input_path}")

    # Create stratification key
    stratify_keys = []
    for record in records:
        key_parts = []
        for col in stratify_by:
            if col not in record:
                raise ValueError(f"Column '{col}' not found in record. Available keys: {list(record.keys())}")
            key_parts.append(str(record[col]))
        stratify_keys.append("_".join(key_parts))

    # First split: train vs (val + test)
    train_ratio_adjusted = train_ratio / (train_ratio + val_ratio + test_ratio)
    val_test_ratio = (val_ratio + test_ratio) / (train_ratio + val_ratio + test_ratio)

    train_records, val_test_records, train_keys, val_test_keys = train_test_split(
        records,
        stratify_keys,
        train_size=train_ratio_adjusted,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_keys if stratify_by else None,
    )

    # Second split: val vs test (from val+test)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    test_ratio_adjusted = test_ratio / (val_ratio + test_ratio)

    val_records, test_records, _, _ = train_test_split(
        val_test_records,
        val_test_keys,
        train_size=val_ratio_adjusted,
        test_size=test_ratio_adjusted,
        random_state=seed,
        stratify=val_test_keys if stratify_by else None,
    )

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)
    _write_jsonl(test_path, test_records)

    # Calculate statistics
    stats = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "label_distribution": _get_distribution(train_records + val_records + test_records, "label"),
        "lang_distribution": _get_distribution(train_records + val_records + test_records, "lang"),
    }

    # Add per-split distributions if stratifying
    if stratify_by:
        stats["train_label_distribution"] = _get_distribution(train_records, "label")
        stats["train_lang_distribution"] = _get_distribution(train_records, "lang")
        stats["val_label_distribution"] = _get_distribution(val_records, "label")
        stats["val_lang_distribution"] = _get_distribution(val_records, "lang")
        stats["test_label_distribution"] = _get_distribution(test_records, "label")
        stats["test_lang_distribution"] = _get_distribution(test_records, "lang")

    return stats


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_distribution(records: list[dict], key: str) -> dict[str, int]:
    """Get count distribution for a key."""
    dist = defaultdict(int)
    for record in records:
        if key in record:
            dist[record[key]] += 1
    return dict(sorted(dist.items()))


def print_split_stats(stats: dict) -> None:
    """
    Pretty print dataset split statistics using rich tables.

    Args:
        stats: Dictionary containing split statistics.
    """
    # Overview table
    table = Table(title="Dataset Split Overview")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    total = stats["train_count"] + stats["val_count"] + stats["test_count"]

    table.add_row(
        "train",
        str(stats["train_count"]),
        f"{100 * stats['train_count'] / total:.1f}%",
    )
    table.add_row(
        "val",
        str(stats["val_count"]),
        f"{100 * stats['val_count'] / total:.1f}%",
    )
    table.add_row(
        "test",
        str(stats["test_count"]),
        f"{100 * stats['test_count'] / total:.1f}%",
    )
    table.add_row("Total", str(total), "100.0%")

    console.print()
    console.print(table)

    # Label distribution
    if "label_distribution" in stats:
        label_table = Table(title="Label Distribution (Overall)")
        label_table.add_column("Label", style="cyan")
        label_table.add_column("Count", style="magenta")
        label_table.add_column("Percentage", style="green")

        for label, count in stats["label_distribution"].items():
            label_table.add_row(
                label,
                str(count),
                f"{100 * count / total:.1f}%",
            )
        console.print(label_table)

    # Language distribution
    if "lang_distribution" in stats:
        lang_table = Table(title="Language Distribution (Overall)")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Count", style="magenta")
        lang_table.add_column("Percentage", style="green")

        for lang, count in stats["lang_distribution"].items():
            lang_table.add_row(
                lang,
                str(count),
                f"{100 * count / total:.1f}%",
            )
        console.print(lang_table)

    # Per-split label distributions
    if "train_label_distribution" in stats:
        split_label_table = Table(title="Label Distribution by Split")
        split_label_table.add_column("Label", style="cyan")

        # Get all unique labels
        all_labels = set()
        for dist_key in [
            "train_label_distribution",
            "val_label_distribution",
            "test_label_distribution",
        ]:
            all_labels.update(stats.get(dist_key, {}).keys())

        for split in ["train", "val", "test"]:
            split_label_table.add_column(
                f"{split.capitalize()} Count",
                style="magenta",
            )

        for label in sorted(all_labels):
            row = [label]
            for split in ["train", "val", "test"]:
                dist_key = f"{split}_label_distribution"
                count = stats.get(dist_key, {}).get(label, 0)
                row.append(str(count))
            split_label_table.add_row(*row)

        console.print(split_label_table)

    # Per-split language distributions
    if "train_lang_distribution" in stats:
        split_lang_table = Table(title="Language Distribution by Split")
        split_lang_table.add_column("Language", style="cyan")

        # Get all unique languages
        all_langs = set()
        for dist_key in [
            "train_lang_distribution",
            "val_lang_distribution",
            "test_lang_distribution",
        ]:
            all_langs.update(stats.get(dist_key, {}).keys())

        for split in ["train", "val", "test"]:
            split_lang_table.add_column(
                f"{split.capitalize()} Count",
                style="magenta",
            )

        for lang in sorted(all_langs):
            row = [lang]
            for split in ["train", "val", "test"]:
                dist_key = f"{split}_lang_distribution"
                count = stats.get(dist_key, {}).get(lang, 0)
                row.append(str(count))
            split_lang_table.add_row(*row)

        console.print(split_lang_table)


def main() -> None:
    """CLI for dataset splitting."""
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/processed/",
        help="Output directory for split files (default: dataset/processed/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set proportion (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set proportion (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set proportion (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Reading dataset from: {input_path}")
    stats = split_dataset(
        input_path,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Splits written to: {output_dir}")
    print_split_stats(stats)


if __name__ == "__main__":
    main()
