"""Run baseline detection on the security_bp_zh hard-case eval set.

This script uses the same detector stack as the API and writes full per-sample
outputs for qualitative analysis before any security_bp_zh retraining.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.aigc_detector.detection.pipeline import DetectionPipeline

console = Console()

DEFAULT_INPUT = Path("dataset/seeds/security_bp_zh/hard_case_eval_v1.jsonl")
DEFAULT_OUTPUT = Path("reports/security_bp_hard_case_eval_v1.json")


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

    table = Table(title="security_bp_zh hard-case baseline")
    table.add_column("ID", style="cyan")
    table.add_column("Subtype")
    table.add_column("Gold")
    table.add_column("Pred")
    table.add_column("P(AI)", justify="right")
    table.add_column("Correct", justify="center")

    correct = 0
    for row in results:
        pred_label = row["prediction"]["predicted_label"]
        p_ai = row["prediction"]["p_ai"]
        is_correct = row["correct"]
        if is_correct:
            correct += 1
        table.add_row(
            row["id"],
            row["subtype"],
            row["label"],
            pred_label,
            f"{p_ai:.4f}",
            "✓" if is_correct else "✗",
        )

    console.print(table)
    console.print(f"[bold green]Saved detailed results to {output_path}[/]")
    console.print(f"[bold]Accuracy:[/] {correct}/{len(results)} = {correct / max(len(results), 1):.3f}")


def main() -> int:
    evaluate(DEFAULT_INPUT, DEFAULT_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
