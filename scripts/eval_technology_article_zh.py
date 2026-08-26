"""Run baseline detection on the dedicated technology_article_zh corpus."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.aigc_detector.detection.pipeline import DetectionPipeline

console = Console()

DEFAULT_INPUT = Path("dataset/seeds/technology_article_zh/hard_case_eval_v1.jsonl")
DEFAULT_OUTPUT = Path("reports/technology_article_zh_baseline_v1.json")


def build_pipeline() -> DetectionPipeline:
    """Delegate to the shared PlanRunner assembly (v0.2a drift fix).

    This script previously hand-rolled its own pipeline (drifted from the
    canonical construction: missing linguistic calibration, stale weights).
    """
    from evaluate_paired_experiment import build_pipeline as _shared

    return _shared()


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate(input_path: Path, output_path: Path) -> None:
    pipeline = build_pipeline()
    records = load_records(input_path)
    results: list[dict] = []

    for record in records:
        t0 = time.perf_counter()
        detected = pipeline.detect(record["text"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append(
            {
                **record,
                "prediction": detected.to_dict(),
                "latency_ms_wall": round(elapsed_ms, 1),
                "correct": (detected.predicted_label == "AI-generated") == (record["label"] == "ai"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    table = Table(title="technology_article_zh baseline")
    table.add_column("Subtype", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("AI Correct", justify="right")
    table.add_column("Human Correct", justify="right")
    table.add_column("Avg P(AI)", justify="right")

    summary: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "ai_total": 0,
            "ai_correct": 0,
            "human_total": 0,
            "human_correct": 0,
            "p_ai_sum": 0.0,
        }
    )
    overall_correct = 0

    for row in results:
        subtype = row.get("subtype", "unknown")
        s = summary[subtype]
        s["total"] += 1
        s["correct"] += int(row["correct"])
        s["p_ai_sum"] += float(row["prediction"]["p_ai"])
        if row["label"] == "ai":
            s["ai_total"] += 1
            s["ai_correct"] += int(row["correct"])
        else:
            s["human_total"] += 1
            s["human_correct"] += int(row["correct"])
        overall_correct += int(row["correct"])

    for subtype, s in sorted(summary.items()):
        avg_p_ai = s["p_ai_sum"] / max(s["total"], 1)
        table.add_row(
            subtype,
            str(int(s["total"])),
            str(int(s["correct"])),
            f"{int(s['ai_correct'])}/{int(s['ai_total'])}",
            f"{int(s['human_correct'])}/{int(s['human_total'])}",
            f"{avg_p_ai:.4f}",
        )

    console.print(table)
    console.print(f"[bold green]Saved detailed results to {output_path}[/]")
    console.print(f"[bold]Accuracy:[/] {overall_correct}/{len(results)} = {overall_correct / max(len(results), 1):.3f}")


def main() -> int:
    evaluate(DEFAULT_INPUT, DEFAULT_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
