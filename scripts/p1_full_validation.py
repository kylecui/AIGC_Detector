"""P1 full validation: 3-way comparison + cross-model generalization.

Steps per language:
  1. Extract statistical features on subsample (GPU) via extract_features_from_jsonl
  2. Train StatisticalClassifier on subsample train
  3. Load full-data LinguisticClassifier (models/linguistic-{lang}/)
  4. Predict p_ai on subsample test (1000 records) for both
  5. Compute 3-way metrics: linguistic-only / statistical-only / fusion
  6. For EN: break down by Label_B (model source) for cross-model generalization

Usage:
  uv run python scripts/p1_full_validation.py --lang en
  uv run python scripts/p1_full_validation.py --lang zh
  uv run python scripts/p1_full_validation.py --lang en --cross-model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
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
logger = logging.getLogger("p1")

# Config
DATA_DIR = Path("dataset/validation_{lang}")
STAT_INPUTS = {s: DATA_DIR / f"{s}.stat_sample.jsonl" for s in ("train", "val", "test")}
STAT_OUTPUTS = {s: Path("models/p1-{lang}") / f"stat_features_{s}.jsonl" for s in ("train", "val", "test")}
LING_CLF_DIR = Path("models/linguistic-{lang}")
STAT_CLF_DIR = Path("models/p1-{lang}")
OUT_DIR = Path("models/p1-{lang}")

REFERENCE_MODELS = {
    "en": {"model": "openai-community/gpt2-xl", "load_in_4bit": False},
    "zh": {"model": "IDEA-CCNL/Wenzhong-GPT2-110M", "load_in_4bit": False},
}


def _load_records(path_template: Path, lang: str, suffix: str = ".jsonl") -> list[dict]:
    path = Path(str(path_template).format(lang=lang)) if suffix == "" else Path(str(path_template).format(lang=lang).replace(".jsonl", suffix))
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _load_features_array(records: list[dict], features_key: str) -> np.ndarray:
    rows = []
    for rec in records:
        feats = rec.get(features_key, {})
        rows.append(list(feats.values()))
    return np.array(rows, dtype=np.float64)


def _labels_to_int(records: list[dict], pos_label: str = "ai") -> np.ndarray:
    return np.array([1 if r.get("label") == pos_label else 0 for r in records], dtype=int)


def _labels_to_str(records: list[dict]) -> list[str]:
    return [r.get("label", "human") for r in records]


def _eval_metrics(evaluator: Evaluator, y_true_str: list[str], p_ai: np.ndarray) -> dict:
    try:
        preds = ["ai" if p >= 0.5 else "human" for p in p_ai]
        m = evaluator.evaluate(y_true_str, preds, y_prob=p_ai)
        return {
            "accuracy": float(m.accuracy), "precision": float(m.precision),
            "recall": float(m.recall), "f1": float(m.f1),
            "roc_auc": float(m.roc_auc) if m.roc_auc is not None and not np.isnan(m.roc_auc) else None,
            "n_samples": int(m.n_samples),
        }
    except Exception as e:
        return {"error": str(e), "roc_auc": None}


def run_p1(lang: str, cross_model: bool = False) -> dict:
    """Run P1 validation for one language."""
    data_dir = Path(str(DATA_DIR).format(lang=lang))
    ling_dir = Path(str(LING_CLF_DIR).format(lang=lang))
    out_dir = Path(str(OUT_DIR).format(lang=lang))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Statistical feature extraction (GPU) ----
    cfg = REFERENCE_MODELS[lang]
    stat_inputs = {s: data_dir / f"{s}.stat_sample.jsonl" for s in ("train", "val", "test")}
    stat_outputs = {s: out_dir / f"stat_features_{s}.jsonl" for s in ("train", "val", "test")}

    extractor = StatisticalFeatureExtractor(model_name=cfg["model"], device="cuda", load_in_4bit=cfg["load_in_4bit"])
    extractor.load()
    try:
        for split in ("train", "val", "test"):
            in_path = stat_inputs[split]
            out_path = stat_outputs[split]
            if not in_path.exists():
                logger.warning("[%s] stat input missing: %s", lang, in_path)
                continue
            if out_path.exists() and out_path.stat().st_size > 0:
                logger.info("[%s] reusing %s", lang, out_path)
                continue
            logger.info("[%s] extracting stat features for %s ...", lang, split)
            stats = extract_features_from_jsonl(extractor, in_path, out_path)
            logger.info("[%s] %s -> %s : %s", lang, split, out_path, stats)
    finally:
        extractor.unload()

    # ---- 2. Train StatisticalClassifier ----
    stat_train = _load_records(stat_outputs["train"], lang)
    stat_val = _load_records(stat_outputs["val"], lang)
    stat_test = _load_records(stat_outputs["test"], lang)

    X_train = _load_features_array(stat_train, "features")  # noqa: N806
    X_val = _load_features_array(stat_val, "features")  # noqa: N806
    X_test = _load_features_array(stat_test, "features")  # noqa: N806
    y_train = _labels_to_int(stat_train)
    y_val = _labels_to_int(stat_val)
    y_test_str = _labels_to_str(stat_test)

    # Handle class imbalance: use scale_pos_weight for XGBoost
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_weight = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0
    logger.info("[%s] class balance: pos=%d neg=%d scale_pos_weight=%.2f", lang, n_pos, n_neg, scale_weight)

    from sklearn.metrics import f1_score

    stat_clf = StatisticalClassifier(backend="xgboost")
    # Inject scale_pos_weight into the XGBClassifier inside the pipeline
    stat_clf._pipeline.named_steps["classifier"].set_params(scale_pos_weight=scale_weight)
    stat_clf.fit(X_train, y_train)

    val_proba = stat_clf.predict_proba(X_val)[:, 1]
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_val, (val_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    stat_clf.set_threshold(best_t)
    stat_clf.save(out_dir / "stat_classifier.joblib")
    logger.info("[%s] stat classifier: val F1=%.4f @ t=%.3f", lang, best_f1, best_t)

    # ---- 3. Load full-data LinguisticClassifier ----
    ling_clf = LinguisticClassifier()
    ling_clf.load(ling_dir / "classifier.joblib")
    cal_path = ling_dir / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        if "optimal_threshold" in cal:
            ling_clf.set_threshold(float(cal["optimal_threshold"]))
    logger.info("[%s] linguistic classifier loaded", lang)

    # ---- 4. Predict on test subsample ----
    ling_test_recs = _load_records(data_dir / "test.ling.jsonl", lang)
    # Align records by id: build a lookup dict from the ling records (which are full-data),
    # then iterate stat_test in order to get aligned linguistic features.
    ling_by_id: dict[str, dict] = {r["id"]: r for r in ling_test_recs}

    X_ling_aligned_rows: list[list[float]] = []
    matched_count = 0
    for rec in stat_test:
        rid = rec.get("id", "")
        ling_rec = ling_by_id.get(rid)
        if ling_rec and "linguistic_features" in ling_rec:
            X_ling_aligned_rows.append(list(ling_rec["linguistic_features"].values()))
            matched_count += 1
        else:
            # If missing, use NaN row (will be imputed)
            X_ling_aligned_rows.append([float("nan")] * 14)

    X_ling_test = np.array(X_ling_aligned_rows, dtype=np.float64)  # noqa: N806
    if matched_count != len(stat_test):
        logger.warning("[%s] id alignment: stat=%d ling_matched=%d", lang, len(stat_test), matched_count)

    p_stat = stat_clf.predict_proba(X_test)[:, 1]
    p_ling = ling_clf.predict_proba(X_ling_test)[:, 1]
    p_fusion = 0.5 * p_stat + 0.5 * p_ling

    # ---- 5. 3-way metrics ----
    evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
    results = {
        "linguistic_only": _eval_metrics(evaluator, y_test_str, p_ling),
        "statistical_only": _eval_metrics(evaluator, y_test_str, p_stat),
        "fusion_0.5_0.5": _eval_metrics(evaluator, y_test_str, p_fusion),
    }
    results["n_test"] = len(stat_test)

    # ---- 6. Cross-model breakdown (EN only) ----
    if cross_model and lang == "en":
        cross_results: dict[str, list[float]] = defaultdict(list)
        source_col = "source"  # our schema: "defactify/<model>" or "defactify/human"
        for i, rec in enumerate(stat_test):
            src = rec.get(source_col, "unknown")
            model = src.replace("defactify/", "") if src.startswith("defactify/") else src
            if model != "human":
                cross_results[model].append(float(p_ling[i]))
        per_model = {}
        for model, probs in sorted(cross_results.items()):
            if len(probs) < 20:
                continue
            avg_p_ai = np.mean(probs)
            per_model[model] = {"n": len(probs), "avg_p_ai_linguistic": float(avg_p_ai)}
        results["cross_model_linguistic"] = per_model

    # ---- 7. Persist ----
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("[%s] results saved to %s", lang, out_dir / "metrics_summary.json")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--cross-model", action="store_true", help="Break down EN results by model source (Label_B).")
    args = parser.parse_args()

    results = run_p1(args.lang, cross_model=args.cross_model)

    print()
    print("=" * 64)
    print(f"  P1 FULL VALIDATION — {args.lang.upper()}")
    print("=" * 64)
    print(f"  n_test = {results.get('n_test', '?')}")
    print(f"  {'Classifier':<22} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
    print("-" * 64)
    for name in ("linguistic_only", "statistical_only", "fusion_0.5_0.5"):
        m = results[name]
        if "error" in m:
            print(f"  {name:<22} ERROR: {m['error']}")
            continue
        auc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "n/a"
        print(f"  {name:<22} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {auc:>10}")
    print("=" * 64)

    cross = results.get("cross_model_linguistic")
    if cross:
        print()
        print("  Cross-Model Generalization (linguistic avg p_ai per model):")
        print(f"  {'Model':<22} {'n':>6} {'avg_p_ai':>10}")
        print("-" * 42)
        for model, info in cross.items():
            print(f"  {model:<22} {info['n']:>6} {info['avg_p_ai']:>10.4f}")
        print("=" * 64)


if __name__ == "__main__":
    main()
