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
    diagnostic_stages: dict[str, Any] = field(default_factory=dict)


class _DiagnosticPipelineWrapper:
    """Compose diagnostic stages around a DetectionPipeline (zero core changes).

    detect() delegates to the inner pipeline, then appends each diagnostic
    stage's result to ``EnsembleResult.breakdown["diagnostic_<id>"]`` as
    auditable evidence. Diagnostic stages never vote and never alter the
    verdict; a failing stage degrades to a neutral entry (contract).
    """

    def __init__(self, inner: Any, stages: dict[str, Any]):
        self._inner = inner
        self._stages = stages
        # inner pipeline attributes are resolved via __getattr__ below
        # (routes/scripts access pipeline.binoculars_detectors etc.)

    def __getattr__(self, name: str) -> Any:
        # only called when normal lookup fails; delegate to inner pipeline
        # (routes/scripts access pipeline.binoculars_detectors etc.)
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def detect(self, text: str) -> Any:
        result = self._inner.detect(text)
        if not self._stages:
            return result
        lang = getattr(result, "detected_language", None)
        bd = dict(getattr(result, "breakdown", {}) or {})
        for sid, stage in self._stages.items():
            try:
                out = stage.predict(text, lang)
            except Exception:  # noqa: BLE001 — contract: degrade, never raise
                from aigc_detector.stages.contract import neutral_result

                out = neutral_result(sid)
            bd[f"diagnostic_{sid}"] = out
        try:
            result.breakdown = bd
        except Exception:  # noqa: BLE001 — frozen result: skip attach
            pass
        return result


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

        # v0.3: third-party diagnostic stages (declared in plan; zero core changes)
        diagnostic_stages = self._load_diagnostic_stages()
        if diagnostic_stages:
            pipeline = _DiagnosticPipelineWrapper(pipeline, diagnostic_stages)

        return PipelineBundle(
            pipeline=pipeline,
            model_manager=model_manager,
            language_router=language_router,
            missing_binoculars=missing_binoculars,
            diagnostic_stages=diagnostic_stages,
        )

    def _load_diagnostic_stages(self) -> dict:
        """Instantiate plan-declared diagnostic stages (module:Class spec).

        Fail-soft by contract: a diagnostic stage that cannot be imported or
        instantiated is SKIPPED with a warning — third-party evidence must
        never break service startup (the same never-degrade-the-verdict
        discipline as predict-time failures).
        """
        import importlib

        stages: dict = {}
        for spec in self.plan.get("diagnostic_stages", []) or []:
            sid, impl = spec["id"], spec["impl"]
            try:
                mod_name, cls_name = impl.split(":")
                cls = getattr(importlib.import_module(mod_name), cls_name)
                stage = cls()
                stage.load()
                stages[sid] = stage
                logger.info("diagnostic stage loaded: %s (%s)", sid, impl)
            except Exception as e:  # noqa: BLE001 — fail-soft, see docstring
                logger.warning(
                    "diagnostic stage %s (%s) unavailable, skipping: %s: %s",
                    sid, impl, type(e).__name__, e,
                )
        return stages
