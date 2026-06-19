"""End-to-end AIGC detector validation on real test datasets.

Runs the FULL detection pipeline (statistical + linguistic + encoder, skipping
binoculars which requires models not in cache) on test records from Defactify
(EN) or HC3-Chinese (ZH), and reports the detector's discriminative ability.

Usage:
  uv run python scripts/validate_detector_e2e.py --lang en --n 500
  uv run python scripts/validate_detector_e2e.py --lang zh --n 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aigc_detector.config import settings
from aigc_detector.detection.language import LanguageRouter
from aigc_detector.detection.linguistic import LinguisticClassifier, LinguisticFeatureExtractor
from aigc_detector.detection.pipeline import DetectionPipeline
from aigc_detector.detection.statistical import StatisticalClassifier, StatisticalFeatureExtractor
from aigc_detector.detection.encoder import EncoderClassifier
from aigc_detector.models.manager import ModelManager
from aigc_detector.training.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("e2e")

DATA_DIR = Path("dataset/validation_{lang}")
OUT_DIR = Path("models/e2e-validation")


def build_pipeline(lang: str) -> DetectionPipeline:
    """Build the detection pipeline for one language."""
    model_manager = ModelManager(max_vram_gb=settings.max_vram_gb)

    # Language router
    language_router = LanguageRouter(device=settings.device)
    try:
        language_router.load()
    except Exception:
        logger.warning("Language model failed to load, using heuristic fallback")

    # Statistical
    stat_model = "openai-community/gpt2-xl" if lang == "en" else "IDEA-CCNL/Wenzhong-GPT2-110M"
    statistical_extractors = {
        lang: StatisticalFeatureExtractor(model_name=stat_model, device=settings.device, load_in_4bit=False),
    }
    statistical_classifiers: dict[str, StatisticalClassifier] = {}
    clf_path = settings.model_dir / f"statistical-{lang}" / "classifier.joblib"
    if clf_path.exists():
        clf = StatisticalClassifier()
        clf.load(clf_path)
        statistical_classifiers[lang] = clf
        logger.info("Loaded statistical classifier for %s", lang)
    else:
        logger.warning("Statistical classifier missing: %s", clf_path)

    # Linguistic (CPU, no model)
    linguistic_extractors = {lang: LinguisticFeatureExtractor()}
    linguistic_classifiers: dict[str, LinguisticClassifier] = {}
    ling_path = settings.model_dir / f"linguistic-{lang}" / "classifier.joblib"
    if ling_path.exists():
        lclf = LinguisticClassifier()
        lclf.load(ling_path)
        cal_path = settings.model_dir / f"linguistic-{lang}" / "calibration.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            if "optimal_threshold" in cal:
                lclf.set_threshold(float(cal["optimal_threshold"]))
        linguistic_classifiers[lang] = lclf
        logger.info("Loaded linguistic classifier for %s", lang)
    else:
        logger.warning("Linguistic classifier missing: %s", ling_path)

    # Encoder
    base_model = "microsoft/deberta-v3-large" if lang == "en" else "hfl/chinese-roberta-wwm-ext-large"
    adapter_path = settings.model_dir / f"encoder-{lang}"
    encoder_classifiers = {
        lang: EncoderClassifier(
            base_model_name=base_model,
            adapter_path=adapter_path,
            device=settings.device,
        ),
    }

    pipeline = DetectionPipeline(
        language_router=language_router,
        statistical_extractors=statistical_extractors,
        statistical_classifiers=statistical_classifiers,
        encoder_classifiers=encoder_classifiers,
        binoculars_detectors={},  # Skip binoculars (models not cached)
        linguistic_extractors=linguistic_extractors,
        linguistic_classifiers=linguistic_classifiers,
        model_manager=model_manager,
        early_exit_threshold=0.99,  # Raised from 0.95 — was too aggressive for modern LLM text
        ensemble_weights_by_lang={
            # EN: encoder LoRA trained on different domain; linguistic is primary signal.
            # Discovered via scripts/tune_en_detector.py weight sweep on Defactify.
            "en": {"linguistic": 0.85, "statistical": 0.15, "encoder": 0.0, "binoculars": 0.0},
            # ZH: encoder works well on HC3 ChatGPT text; keep default weights.
            "zh": None,  # None = use DEFAULT_WEIGHTS
        },
    )
    return pipeline


def run_validation(lang: str, n: int) -> dict:
    """Run end-to-end validation for one language."""
    data_dir = Path(str(DATA_DIR).replace("{lang}", lang))
    test_path = data_dir / "test.jsonl"

    logger.info("Loading test data from %s", test_path)
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Sample N records (stratified by label for balanced evaluation)
    import random
    random.seed(42)
    human_recs = [r for r in records if r.get("label") == "human"]
    ai_recs = [r for r in records if r.get("label") == "ai"]
    random.shuffle(human_recs)
    random.shuffle(ai_recs)
    n_each = n // 2
    sampled = human_recs[:n_each] + ai_recs[:n_each]
    random.shuffle(sampled)
    logger.info("Sampled %d records (human=%d, ai=%d)", len(sampled), n_each, n_each)

    # Build pipeline
    logger.info("Building detection pipeline for lang=%s ...", lang)
    pipeline = build_pipeline(lang)

    # Run detection
    y_true: list[str] = []
    y_pred: list[str] = []
    y_prob: list[float] = []
    stages_used_log: list[str] = []
    per_source_probs: dict[str, list[float]] = defaultdict(list)
    errors = 0
    t0 = time.perf_counter()

    for i, rec in enumerate(sampled):
        text = rec.get("text", "")
        true_label = rec.get("label", "human")
        source = rec.get("source", "unknown")
        try:
            result = pipeline.detect(text)
            y_true.append(true_label)
            y_pred.append(result.predicted_label)
            y_prob.append(result.p_ai)
            stages_used_log.append(",".join(result.stages_used))
            if true_label == "ai":
                model_name = source.split("/", 1)[1] if "/" in source else source
                per_source_probs[model_name].append(result.p_ai)
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning("Detection failed on record %d: %s", i, e)
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("Progress: %d/%d (errors=%d, %.1fs elapsed)", i + 1, len(sampled), errors, elapsed)

    elapsed = time.perf_counter() - t0
    logger.info("Done: %d records in %.1fs (%.0fms/record)", len(y_true), elapsed, elapsed * 1000 / max(len(y_true), 1))

    # Normalize pipeline display labels to canonical form
    LABEL_MAP = {"AI-generated": "ai", "Human-written": "human", "ai": "ai", "human": "human"}
    y_pred_norm = [LABEL_MAP.get(p, p) for p in y_pred]

    # Filter to binary labels only (pipeline may occasionally return "mixed" or "unknown")
    valid_mask = [(t in ("human", "ai") and p in ("human", "ai")) for t, p in zip(y_true, y_pred_norm)]
    y_true_f = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_f = [p for p, v in zip(y_pred_norm, valid_mask) if v]
    y_prob_f = [p for p, v in zip(y_prob, valid_mask) if v]
    dropped = len(y_true) - len(y_true_f)
    if dropped > 0:
        logger.warning("Dropped %d records with non-binary labels", dropped)

    # Compute metrics
    evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
    metrics = evaluator.evaluate(y_true_f, y_pred_f, y_prob=np.array(y_prob_f))

    # Stage usage stats
    stage_counter = Counter(stages_used_log)

    # Per-source breakdown (AI records only)
    per_source = {}
    for model_name, probs in sorted(per_source_probs.items()):
        if len(probs) >= 5:
            per_source[model_name] = {
                "n": len(probs),
                "avg_p_ai": float(np.mean(probs)),
                "detected_as_ai_pct": float(np.mean(np.array(probs) >= 0.5)),
            }

    result = {
        "lang": lang,
        "n_tested": len(y_true),
        "n_errors": errors,
        "elapsed_seconds": elapsed,
        "ms_per_record": elapsed * 1000 / max(len(y_true), 1),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "roc_auc": float(metrics.roc_auc) if metrics.roc_auc and not np.isnan(metrics.roc_auc) else None,
        "confusion_matrix": metrics.confusion,
        "stage_usage": dict(stage_counter),
        "per_source_model": per_source,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--n", type=int, default=500, help="Number of test records (stratified 50/50)")
    args = parser.parse_args()

    result = run_validation(args.lang, args.n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"e2e_{args.lang}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 64)
    print(f"  END-TO-END DETECTOR VALIDATION — {args.lang.upper()}")
    print("=" * 64)
    print(f"  n_tested = {result['n_tested']}  |  errors = {result['n_errors']}")
    print(f"  latency  = {result['ms_per_record']:.0f} ms/record")
    print(f"  {'Metric':<16} {'Value':>10}")
    print(f"  {'-'*30}")
    print(f"  {'Accuracy':<16} {result['accuracy']:>10.4f}")
    print(f"  {'Precision':<16} {result['precision']:>10.4f}")
    print(f"  {'Recall':<16} {result['recall']:>10.4f}")
    print(f"  {'F1':<16} {result['f1']:>10.4f}")
    auc = result.get("roc_auc")
    print(f"  {'ROC-AUC':<16} {auc:>10.4f}" if auc else f"  {'ROC-AUC':<16} {'n/a':>10}")
    print(f"  {'-'*30}")
    print(f"  Confusion: {result['confusion_matrix']}")
    print(f"  Stage usage: {result['stage_usage']}")
    print("=" * 64)

    if result["per_source_model"]:
        print(f"\n  Per-Source-Model (AI records):")
        print(f"  {'Model':<22} {'n':>6} {'avg_p_ai':>10} {'detect%':>10}")
        print(f"  {'-'*52}")
        for model, info in result["per_source_model"].items():
            print(f"  {model:<22} {info['n']:>6} {info['avg_p_ai']:>10.4f} {info['detected_as_ai_pct']:>10.1%}")
        print("=" * 64)


if __name__ == "__main__":
    main()
