"""Efficient EN detector tuning: collect per-stage p_ai once, sweep weights offline.

1. Runs ALL stages (stat + ling + encoder) on every record (no early exit)
2. Captures individual stage p_ai per record
3. Sweeps ensemble weight combinations offline
4. Reports optimal configuration

Usage:
  uv run python scripts/tune_en_detector.py --n 500
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import f1_score, roc_auc_score

from aigc_detector.config import settings
from aigc_detector.detection.encoder import EncoderClassifier
from aigc_detector.detection.language import LanguageRouter
from aigc_detector.detection.linguistic import LinguisticClassifier, LinguisticFeatureExtractor
from aigc_detector.detection.pipeline import DetectionPipeline
from aigc_detector.detection.statistical import StatisticalClassifier, StatisticalFeatureExtractor
from aigc_detector.models.manager import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tune")

DATA_DIR = Path("dataset/validation_en")
OUT_DIR = Path("models/e2e-validation")

LABEL_MAP = {"AI-generated": "ai", "Human-written": "human", "ai": "ai", "human": "human"}


def collect_stage_scores(n: int) -> list[dict]:
    """Run all 3 stages on each record, collect per-stage p_ai."""
    records = []
    with open(DATA_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(42)
    human = [r for r in records if r["label"] == "human"]
    ai = [r for r in records if r["label"] == "ai"]
    random.shuffle(human)
    random.shuffle(ai)
    n_each = n // 2
    sampled = human[:n_each] + ai[:n_each]
    random.shuffle(sampled)
    logger.info("Sampled %d records (human=%d, ai=%d)", len(sampled), n_each, n_each)

    mm = ModelManager(max_vram_gb=settings.max_vram_gb)
    lr = LanguageRouter(device="cuda")
    lr.load()

    stat_ext = {"en": StatisticalFeatureExtractor(model_name="openai-community/gpt2-xl", device="cuda")}
    stat_clf = {"en": StatisticalClassifier()}
    stat_clf["en"].load(settings.model_dir / "statistical-en" / "classifier.joblib")

    ling_ext = {"en": LinguisticFeatureExtractor()}
    ling_clf = {"en": LinguisticClassifier()}
    ling_clf["en"].load(settings.model_dir / "linguistic-en" / "classifier.joblib")
    cal = settings.model_dir / "linguistic-en" / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text(encoding="utf-8"))
        if "optimal_threshold" in c:
            ling_clf["en"].set_threshold(float(c["optimal_threshold"]))

    enc_clf = {
        "en": EncoderClassifier(
            base_model_name="microsoft/deberta-v3-large",
            adapter_path=settings.model_dir / "encoder-en",
            device="cuda",
        )
    }

    # Build pipeline with early_exit disabled
    pipeline = DetectionPipeline(
        language_router=lr,
        statistical_extractors=stat_ext,
        statistical_classifiers=stat_clf,
        encoder_classifiers=enc_clf,
        binoculars_detectors={},
        linguistic_extractors=ling_ext,
        linguistic_classifiers=ling_clf,
        model_manager=mm,
        early_exit_threshold=1.01,  # Disable early exit
    )

    results = []
    t0 = time.perf_counter()
    for i, rec in enumerate(sampled):
        text = rec["text"]
        true_label = rec["label"]
        source = rec.get("source", "")
        model_name = source.split("/", 1)[1] if "/" in source else source

        try:
            result = pipeline.detect(text)
            # Extract per-stage p_ai from breakdown
            breakdown = result.breakdown or {}
            p_stat = breakdown.get("statistical", {}).get("p_ai")
            p_ling = breakdown.get("linguistic", {}).get("p_ai")
            p_enc = breakdown.get("encoder", {}).get("p_ai")

            results.append({
                "id": rec.get("id", ""),
                "true_label": true_label,
                "source_model": model_name,
                "p_stat": p_stat,
                "p_ling": p_ling,
                "p_enc": p_enc,
                "p_ai_final": result.p_ai,
                "stages": result.stages_used,
            })
        except Exception as e:
            logger.warning("Record %d failed: %s", i, e)

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d (%.1fs)", i + 1, len(sampled), time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    logger.info("Collection done: %d records in %.1fs", len(results), elapsed)
    return results


def sweep_weights(results: list[dict]) -> list[dict]:
    """Try different ensemble weight combinations offline."""
    y_true_str = [r["true_label"] for r in results]
    y_true_int = np.array([1 if label == "ai" else 0 for label in y_true_str])

    p_stat = np.array([r["p_stat"] or 0.5 for r in results])
    p_ling = np.array([r["p_ling"] or 0.5 for r in results])
    p_enc = np.array([r["p_enc"] or 0.5 for r in results])

    configs = []
    # Weight grid: stat, ling, enc (bino=0, renormalized)
    for ws, wl, we in product(
        [0.0, 0.10, 0.15, 0.20, 0.30],
        [0.0, 0.15, 0.20, 0.30, 0.40, 0.50],
        [0.0, 0.30, 0.50, 0.60, 0.70, 0.80],
    ):
        total = ws + wl + we
        if total == 0:
            continue
        ws_n, wl_n, we_n = ws / total, wl / total, we / total
        p_fused = ws_n * p_stat + wl_n * p_ling + we_n * p_enc
        if len(np.unique(y_true_int)) < 2:
            continue
        try:
            auc = roc_auc_score(y_true_int, p_fused)
        except ValueError:
            continue
        preds = (p_fused >= 0.5).astype(int)
        f1 = f1_score(y_true_int, preds, zero_division=0)
        recall = float((preds[y_true_int == 1] == 1).mean()) if (y_true_int == 1).sum() > 0 else 0.0
        acc = float((preds == y_true_int).mean())
        configs.append({
            "weights": {"stat": ws, "ling": wl, "enc": we},
            "weights_norm": {"stat": round(ws_n, 4), "ling": round(wl_n, 4), "enc": round(we_n, 4)},
            "roc_auc": float(auc),
            "f1": float(f1),
            "recall": recall,
            "accuracy": acc,
        })

    configs.sort(key=lambda c: c["roc_auc"], reverse=True)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--reuse", action="store_true", help="Reuse cached stage scores if available")
    args = parser.parse_args()

    cache_path = OUT_DIR / "en_stage_scores.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reuse and cache_path.exists():
        logger.info("Reusing cached stage scores from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = collect_stage_scores(args.n)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    logger.info("Sweeping weight combinations on %d records...", len(results))
    configs = sweep_weights(results)

    # Also compute individual stage baselines
    y_true_int = np.array([1 if r["true_label"] == "ai" else 0 for r in results])
    for name, key in (("stat_only", "p_stat"), ("ling_only", "p_ling"), ("enc_only", "p_enc")):
        p = np.array([r[key] or 0.5 for r in results])
        try:
            auc = roc_auc_score(y_true_int, p)
            preds = (p >= 0.5).astype(int)
            f1 = f1_score(y_true_int, preds, zero_division=0)
            recall = float((preds[y_true_int == 1] == 1).mean())
            print(f"  {name:<12} AUC={auc:.4f} F1={f1:.4f} Recall={recall:.4f}")
        except Exception:
            print(f"  {name:<12} FAILED")

    print(f"\n{'='*72}")
    print(f"  TOP 10 ENSEMBLE CONFIGURATIONS (n={len(results)})")
    print(f"{'='*72}")
    print(f"  {'stat':>5} {'ling':>5} {'enc':>5} | {'ROC-AUC':>8} {'F1':>8} {'Recall':>8} {'Acc':>8}")
    print(f"  {'-'*60}")
    for c in configs[:10]:
        w = c["weights_norm"]
        print(f"  {w['stat']:>5.2f} {w['ling']:>5.2f} {w['enc']:>5.2f} | "
              f"{c['roc_auc']:>8.4f} {c['f1']:>8.4f} {c['recall']:>8.4f} {c['accuracy']:>8.4f}")
    print(f"{'='*72}")

    # Per-model breakdown for best config
    best = configs[0]
    wb = best["weights_norm"]
    for r in results:
        p = wb["stat"] * (r["p_stat"] or 0.5) + wb["ling"] * (r["p_ling"] or 0.5) + wb["enc"] * (r["p_enc"] or 0.5)
        r["p_ai_best"] = p

    print("\n  Best config per-model breakdown:")
    print(f"  {'Model':<22} {'n':>5} {'avg_p_ai':>10} {'detect%':>10}")
    print(f"  {'-'*50}")
    by_model = defaultdict(list)
    for r in results:
        if r["true_label"] == "ai":
            by_model[r["source_model"]].append(r["p_ai_best"])
    for model, probs in sorted(by_model.items()):
        if len(probs) < 5:
            continue
        print(f"  {model:<22} {len(probs):>5} {np.mean(probs):>10.4f} {np.mean(np.array(probs) >= 0.5):>10.1%}")
    print(f"{'='*72}")

    with open(OUT_DIR / "en_weight_sweep.json", "w", encoding="utf-8") as f:
        json.dump(configs[:20], f, indent=2)
    logger.info("Saved sweep results to %s", OUT_DIR / "en_weight_sweep.json")


if __name__ == "__main__":
    main()
