"""DetectionPlan — declarative assembly for the detection framework (v0.2a).

One YAML plan describes the complete deployment: stage models, ensemble
weights, early-exit, calibration artifact wiring. PlanRunner is the SINGLE
assembly point replacing the six drifting hand-written constructions found
by the 2026-08-26 config-surface audit (api/main.py lifespan,
evaluate_paired_experiment.build_pipeline, and four standalone eval-script
builders that had already diverged).

v0.2a scope: assembly unification with ZERO behavior change. The plan file
expresses today's defaults; the API-only background-download thread stays
in api/main.py, fed by ``bundle.missing_binoculars``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = ("plan", "language", "statistical", "encoder", "binoculars", "weights")


@dataclass
class PipelineBundle:
    """Everything the old hand-written constructions produced."""

    pipeline: Any
    model_manager: Any
    language_router: Any
    missing_binoculars: list[tuple[str, str, str]] = field(default_factory=list)


class PlanRunner:
    """Load a DetectionPlan YAML and assemble the pipeline from it."""

    def __init__(self, plan: dict, plan_path: Path | None = None):
        self.plan = plan
        self.plan_path = plan_path
        self._validate()

    # ---------------- loading ----------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> PlanRunner:
        p = Path(path)
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")), plan_path=p)

    @classmethod
    def default(cls) -> PlanRunner:
        """The shipped default plan (repo-root plans/default.yaml)."""
        root = Path(__file__).resolve().parents[2]
        return cls.from_yaml(root / "plans" / "default.yaml")

    # ---------------- validation ----------------

    def _validate(self) -> None:
        for section in REQUIRED_SECTIONS:
            if section not in self.plan:
                raise ValueError(f"plan missing required section: {section}")
        for lang in ("en", "zh"):
            if lang not in self.plan["weights"]:
                raise ValueError(f"plan weights missing language: {lang}")
            total = sum(self.plan["weights"][lang].values())
            if not 0.99 <= total <= 1.01:
                raise ValueError(f"plan weights for {lang} do not sum to 1: {total}")
        ee = self.plan.get("early_exit_threshold", 0.99)
        if not 0 < ee <= 1.01:
            raise ValueError(f"early_exit_threshold out of range: {ee}")

    # ---------------- accessors (tested surface) ----------------

    @property
    def weights(self) -> dict[str, dict[str, float]]:
        return self.plan["weights"]

    @property
    def early_exit_threshold(self) -> float:
        return float(self.plan.get("early_exit_threshold", 0.99))

    # ---------------- assembly ----------------

    def build(self, adapter_zh: Path | None = None) -> PipelineBundle:
        """Assemble the pipeline exactly as the old canonical construction.

        adapter_zh overrides the production encoder-zh adapter (candidate
        gating; preserved from evaluate_paired_experiment.build_pipeline).
        """
        from aigc_detector.config import settings
        from aigc_detector.detection.binoculars import BinocularsDetector
        from aigc_detector.detection.encoder import EncoderClassifier
        from aigc_detector.detection.language import LanguageRouter
        from aigc_detector.detection.linguistic import (
            LinguisticClassifier,
            LinguisticFeatureExtractor,
        )
        from aigc_detector.detection.pipeline import DetectionPipeline
        from aigc_detector.detection.statistical import (
            StatisticalClassifier,
            StatisticalFeatureExtractor,
        )
        from aigc_detector.models.manager import ModelManager
        from aigc_detector.utils.hf_cache import is_model_cached

        model_dir = Path(settings.model_dir)
        p = self.plan

        model_manager = ModelManager(max_vram_gb=settings.max_vram_gb)

        language_router = LanguageRouter(device=settings.device)
        try:
            language_router.load()
            model_manager.load("xlm-roberta-lang-detect", language_router)
        except Exception:  # noqa: BLE001 — fallback mirrors lifespan behavior
            logger.warning("Language detection model failed to load, using heuristic fallback")

        statistical_extractors = {
            lang: StatisticalFeatureExtractor(
                model_name=spec["extractor"],
                device=settings.device,
                load_in_4bit=p["statistical"].get("load_in_4bit", False),
            )
            for lang, spec in p["statistical"]["languages"].items()
        }
        statistical_classifiers = {}
        for lang in p["statistical"]["languages"]:
            clf_path = model_dir / f"statistical-{lang}" / "classifier.joblib"
            if clf_path.exists():
                clf = StatisticalClassifier()
                clf.load(clf_path)
                cal_path = model_dir / f"statistical-{lang}" / "calibration.json"
                if cal_path.exists():
                    try:
                        cal = json.loads(cal_path.read_text(encoding="utf-8"))
                        if "optimal_threshold" in cal:
                            clf.set_threshold(float(cal["optimal_threshold"]))
                    except Exception:  # noqa: BLE001
                        logger.warning("Failed to load calibration for %s", lang, exc_info=True)
                statistical_classifiers[lang] = clf
            else:
                logger.warning("Statistical classifier missing for %s: %s", lang, clf_path)

        linguistic_classifiers = {}
        for lang in ("en", "zh"):
            clf_path = model_dir / f"linguistic-{lang}" / "classifier.joblib"
            if clf_path.exists():
                clf = LinguisticClassifier()
                clf.load(clf_path)
                cal_path = model_dir / f"linguistic-{lang}" / "calibration.json"
                if cal_path.exists():
                    try:
                        cal = json.loads(cal_path.read_text(encoding="utf-8"))
                        if "optimal_threshold" in cal:
                            clf.set_threshold(float(cal["optimal_threshold"]))
                    except Exception:  # noqa: BLE001
                        logger.warning("Failed to load linguistic calibration for %s", lang, exc_info=True)
                linguistic_classifiers[lang] = clf
            else:
                logger.info("Linguistic classifier not found for %s (skipping)", lang)
        linguistic_extractors = {
            "en": LinguisticFeatureExtractor(),
            "zh": LinguisticFeatureExtractor(),
        }

        encoder_classifiers = {}
        for lang, spec in p["encoder"]["languages"].items():
            adapter = adapter_zh if (lang == "zh" and adapter_zh is not None) else (
                model_dir / spec["adapter"]
            )
            encoder_classifiers[lang] = EncoderClassifier(
                base_model_name=spec["base"],
                adapter_path=adapter,
                device=settings.device,
            )

        binoculars_detectors = {}
        missing_binoculars: list[tuple[str, str, str]] = []
        bino_cfg = p["binoculars"]
        for lang, spec in bino_cfg["pairs"].items():
            observer, performer = spec["observer"], spec["performer"]
            if is_model_cached(observer) and is_model_cached(performer):
                binoculars_detectors[lang] = BinocularsDetector(
                    observer_name=observer,
                    performer_name=performer,
                    mode=bino_cfg.get("mode", "low-fpr"),
                    device=settings.device,
                    load_in_4bit=bino_cfg.get("load_in_4bit", True),
                )
                logger.info("Binoculars enabled for %s (%s + %s)", lang, observer, performer)
            elif bino_cfg.get("require_cached", True):
                missing_binoculars.append((lang, observer, performer))
                logger.info("Binoculars pending for %s — will download in background", lang)

        pipeline = DetectionPipeline(
            language_router=language_router,
            statistical_extractors=statistical_extractors,
            statistical_classifiers=statistical_classifiers,
            encoder_classifiers=encoder_classifiers,
            binoculars_detectors=binoculars_detectors,
            linguistic_extractors=linguistic_extractors,
            linguistic_classifiers=linguistic_classifiers,
            model_manager=model_manager,
            early_exit_threshold=self.early_exit_threshold,
            ensemble_weights_by_lang={
                lang: dict(self.weights[lang]) for lang in ("en", "zh")
            },
        )
        return PipelineBundle(
            pipeline=pipeline,
            model_manager=model_manager,
            language_router=language_router,
            missing_binoculars=missing_binoculars,
        )
