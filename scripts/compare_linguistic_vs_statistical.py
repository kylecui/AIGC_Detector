"""3-way comparison: linguistic-only vs statistical-only vs fusion.

Runs on the smoke validation data (dataset/validation_{en,zh}/) and produces
ROC curves + metrics for three classifiers on the same test set:
  1. Linguistic-only (loads pre-trained models/linguistic-{lang}-smoke/)
  2. Statistical-only (trains fresh on smoke train split using LM features)
  3. Fusion (simple 0.5/0.5 weighted average of the two p_ai scores)

Outputs:
  models/comparison-smoke/<lang>/
    ├── stat_features_<split>.jsonl   (extracted LM features)
    ├── stat_classifier.joblib        (trained stat classifier)
    ├── roc_data.json                 (fpr/tpr/threshold for all 3 curves)
    ├── metrics_summary.json          (AUC/F1/accuracy for all 3)
    └── roc_plot.png                  (visual comparison)

Usage:
  uv run python scripts/compare_linguistic_vs_statistical.py --lang en
  uv run python scripts/compare_linguistic_vs_statistical.py --lang zh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aigc_detector.detection.linguistic import LinguisticClassifier
from aigc_detector.detection.statistical import (
    StatisticalClassifier,
    StatisticalFeatureExtractor,
    extract_features_from_jsonl,
)
from aigc_detector.training.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_MODELS = {
    "en": {"model": "openai-community/gpt2-xl", "load_in_4bit": False},
    "zh": {"model": "IDEA-CCNL/Wenzhong-GPT2-110M", "load_in_4bit": False},
}

DATA_DIR_TEMPLATE = "dataset/validation_{lang}"
LINGUISTIC_DIR_TEMPLATE = "models/linguistic-{lang}-smoke"
OUT_DIR_TEMPLATE = "models/comparison-smoke/{lang}"


# ======================================================================
# Helpers
# ======================================================================


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_features_array(records: list[dict], features_key: str) -> np.ndarray:
    """Stack the 14-dim (linguistic) or 6-dim (statistical) feature vectors."""
    rows = []
    for rec in records:
        feats = rec.get(features_key, {})
        # StatisticalFeatures has 6 keys; LinguisticFeatures has 14.
        # Use the values in canonical order via the dataclass field order.
        rows.append(list(feats.values()))
    return np.array(rows, dtype=np.float64)


def _labels_to_int(records: list[dict], pos_label: str = "ai") -> np.ndarray:
    return np.array([1 if r.get("label") == pos_label else 0 for r in records], dtype=int)


def _labels_to_str(records: list[dict]) -> list[str]:
    return [r.get("label", "human") for r in records]


# ======================================================================
# Main comparison
# ======================================================================


def run_comparison(lang: str, fusion_weight_stat: float = 0.5) -> dict:
    """Run the 3-way comparison for one language.

    Returns the metrics summary dict.
    """
    data_dir = Path(DATA_DIR_TEMPLATE.format(lang=lang))
    ling_dir = Path(LINGUISTIC_DIR_TEMPLATE.format(lang=lang))
    out_dir = Path(OUT_DIR_TEMPLATE.format(lang=lang))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load smoke validation records ----
    train_recs = _load_records(data_dir / "train.jsonl")
    val_recs = _load_records(data_dir / "val.jsonl")
    test_recs = _load_records(data_dir / "test.jsonl")
    logger.info("[%s] loaded: train=%d val=%d test=%d", lang, len(train_recs), len(val_recs), len(test_recs))

    # ---- 2. Statistical feature extraction (GPU LM forward) ----
    stat_features_paths = {}
    cfg = REFERENCE_MODELS[lang]
    extractor = StatisticalFeatureExtractor(
        model_name=cfg["model"],
        device="cuda",
        load_in_4bit=cfg["load_in_4bit"],
    )
    extractor.load()
    try:
        for split, recs in (("train", train_recs), ("val", val_recs), ("test", test_recs)):
            in_path = data_dir / f"{split}.jsonl"
            out_path = out_dir / f"stat_features_{split}.jsonl"
            if not out_path.exists() or out_path.stat().st_size == 0:
                logger.info("[%s] extracting stat features for %s (%d records)...", lang, split, len(recs))
                stats = extract_features_from_jsonl(extractor, in_path, out_path)
                logger.info("[%s] %s: %s", lang, split, stats)
            else:
                logger.info("[%s] reusing existing %s", lang, out_path)
            stat_features_paths[split] = out_path
    finally:
        extractor.unload()

    # ---- 3. Train statistical classifier ----
    stat_train = _load_records(stat_features_paths["train"])
    stat_val = _load_records(stat_features_paths["val"])
    stat_test = _load_records(stat_features_paths["test"])

    X_train_stat = _load_features_array(stat_train, "features")  # noqa: N806
    X_val_stat = _load_features_array(stat_val, "features")  # noqa: N806
    X_test_stat = _load_features_array(stat_test, "features")  # noqa: N806
    y_train = _labels_to_int(stat_train)  # noqa: N806
    y_val = _labels_to_int(stat_val)  # noqa: N806
    y_test_str = _labels_to_str(stat_test)
    y_val = _labels_to_int(stat_val)  # noqa: N806

    stat_clf = StatisticalClassifier(backend="xgboost")
    stat_clf.fit(X_train_stat, y_train)
    # Calibrate threshold on val
    val_stat_proba = stat_clf.predict_proba(X_val_stat)[:, 1]
    from sklearn.metrics import f1_score

    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        preds = (val_stat_proba >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    stat_clf.set_threshold(best_t)
    stat_clf.save(out_dir / "stat_classifier.joblib")
    logger.info("[%s] statistical classifier trained; val F1=%.4f @ t=%.3f", lang, best_f1, best_t)

    # ---- 4. Load pre-trained linguistic classifier ----
    ling_clf = LinguisticClassifier()
    ling_clf.load(ling_dir / "classifier.joblib")
    cal_path = ling_dir / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        if "optimal_threshold" in cal:
            ling_clf.set_threshold(float(cal["optimal_threshold"]))

    # ---- 5. Compute p_ai for all 3 classifiers on val + test ----
    # Linguistic uses the ling features already in the records (from earlier smoke run)
    # For stat, we already have the features loaded above.
    X_test_ling = _load_features_array(  # noqa: N806
        _load_records(data_dir / "test.ling.jsonl"), "linguistic_features"
    )

    p_stat_test = stat_clf.predict_proba(X_test_stat)[:, 1]
    p_ling_test = ling_clf.predict_proba(X_test_ling)[:, 1]
    w = fusion_weight_stat
    p_fusion_test = w * p_stat_test + (1.0 - w) * p_ling_test

    # ---- 6. Evaluate on test set ----
    evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")

    def _eval(p_ai: np.ndarray) -> dict:
        # Skip ROC-AUC if single-class
        try:
            preds_str = ["ai" if p >= 0.5 else "human" for p in p_ai]
            m = evaluator.evaluate(y_test_str, preds_str, y_prob=p_ai)
            return {
                "accuracy": float(m.accuracy),
                "precision": float(m.precision),
                "recall": float(m.recall),
                "f1": float(m.f1),
                "roc_auc": float(m.roc_auc) if m.roc_auc is not None else None,
                "n_samples": int(m.n_samples),
            }
        except Exception as e:
            return {"error": str(e)}

    metrics = {
        "linguistic_only": _eval(p_ling_test),
        "statistical_only": _eval(p_stat_test),
        "fusion_0.5_0.5": _eval(p_fusion_test),
    }

    # ---- 7. ROC curve data ----
    roc_data: dict = {}
    for name, p_ai in (
        ("linguistic_only", p_ling_test),
        ("statistical_only", p_stat_test),
        ("fusion_0.5_0.5", p_fusion_test),
    ):
        try:
            roc = evaluator.roc_curve(y_test_str, p_ai)
            roc_data[name] = {
                "fpr": roc["fpr"],
                "tpr": roc["tpr"],
                "thresholds": roc["thresholds"],
            }
        except Exception as e:
            roc_data[name] = {"error": str(e)}
            logger.warning("[%s] ROC failed for %s: %s", lang, name, e)

    # ---- 8. Persist artifacts ----
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "roc_data.json", "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)
    logger.info("[%s] metrics + ROC data saved to %s", lang, out_dir)

    # ---- 9. Optional plot ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = {"linguistic_only": "tab:blue", "statistical_only": "tab:orange", "fusion_0.5_0.5": "tab:green"}
        for name, rd in roc_data.items():
            if "error" in rd:
                continue
            auc = metrics[name].get("roc_auc")
            label = f"{name}" + (f" (AUC={auc:.3f})" if auc is not None else "")
            ax.plot(rd["fpr"], rd["tpr"], label=label, color=colors.get(name), lw=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Comparison ({lang}, smoke test n={len(test_recs)})")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_plot.png", dpi=120)
        plt.close(fig)
        logger.info("[%s] plot saved: %s", lang, out_dir / "roc_plot.png")
    except ImportError:
        logger.warning("[%s] matplotlib not available, skipping plot", lang)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument(
        "--fusion-weight-stat",
        type=float,
        default=0.5,
        help="Weight for statistical in fusion (0..1).",
    )
    args = parser.parse_args()

    metrics = run_comparison(args.lang, fusion_weight_stat=args.fusion_weight_stat)

    print()
    print("=" * 64)
    print(f"  3-WAY COMPARISON — {args.lang.upper()} (smoke test)")
    print("=" * 64)
    print(f"  {'Classifier':<22} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
    print("-" * 64)
    for name, m in metrics.items():
        if "error" in m:
            print(f"  {name:<22} ERROR: {m['error']}")
            continue
        auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "n/a"
        print(f"  {name:<22} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {auc_str:>10}")
    print("=" * 64)


if __name__ == "__main__":
    main()
