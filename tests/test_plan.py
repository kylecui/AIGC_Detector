"""v0.2a PlanRunner tests: plan loading, validation, and shipped-default pinning.

The behavioral-equivalence guarantee (same assembly as the old hand-written
constructions) is enforced two ways: (1) the shipped default plan pins the
exact weights/threshold/models the old code hardcoded — these tests assert
the plan carries them; (2) GPU spot checks compare live detection outputs
against cached anchors (FN-1 replay) at integration time, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.plan import REQUIRED_SECTIONS, PlanRunner  # noqa: E402

REPO = Path(__file__).parent.parent


class TestPlanLoading:
    def test_default_plan_loads(self):
        runner = PlanRunner.default()
        assert runner.plan["plan"] == "default@0.2a"

    def test_from_yaml_explicit_path(self):
        runner = PlanRunner.from_yaml(REPO / "plans" / "default.yaml")
        assert runner.early_exit_threshold == 0.99


class TestPlanValidation:
    def test_missing_section_rejected(self):
        with pytest.raises(ValueError, match="missing required section"):
            PlanRunner({"plan": "x"})  # everything else absent

    def test_weights_must_sum_to_one(self):
        plan = PlanRunner.default().plan
        bad = {k: v for k, v in plan.items()}
        bad["weights"] = {"en": {"linguistic": 0.5}, "zh": bad["weights"]["zh"]}
        with pytest.raises(ValueError, match="do not sum to 1"):
            PlanRunner(bad)

    def test_early_exit_range(self):
        plan = PlanRunner.default().plan
        bad = {k: v for k, v in plan.items()}
        bad["early_exit_threshold"] = 1.5
        with pytest.raises(ValueError, match="early_exit_threshold"):
            PlanRunner(bad)

    def test_required_sections_constant(self):
        assert set(REQUIRED_SECTIONS) <= set(PlanRunner.default().plan)


class TestShippedDefaultPins:
    """Pin the plan to the exact values the six old constructions hardcoded.

    If this test fails after a plan edit, that is a DEPLOYMENT CHANGE:
    probes + verify gate must run before merge (see plans/default.yaml header).
    """

    def test_weights_match_pre_unification_constants(self):
        w = PlanRunner.default().weights
        assert w["en"] == {"linguistic": 0.85, "statistical": 0.15, "encoder": 0.0, "binoculars": 0.0}
        assert w["zh"] == {"linguistic": 0.10, "statistical": 0.10, "encoder": 0.60, "binoculars": 0.20}

    def test_early_exit_matches_override(self):
        # the old code overrode the 0.95 default to 0.99 in BOTH canonical builders
        assert PlanRunner.default().early_exit_threshold == 0.99

    def test_models_match_hardcoded_ids(self):
        p = PlanRunner.default().plan
        assert p["statistical"]["languages"]["en"]["extractor"] == "openai-community/gpt2-xl"
        assert p["statistical"]["languages"]["zh"]["extractor"] == "IDEA-CCNL/Wenzhong-GPT2-110M"
        assert p["encoder"]["languages"]["en"]["base"] == "microsoft/deberta-v3-large"
        assert p["encoder"]["languages"]["zh"]["base"] == "hfl/chinese-roberta-wwm-ext-large"
        assert p["binoculars"]["pairs"]["zh"]["observer"] == "Qwen/Qwen2-7B"
        assert p["binoculars"]["pairs"]["en"]["performer"] == "tiiuae/falcon-7b-instruct"

    def test_hub_delegates_to_planrunner(self):
        """The experiment hub is now a thin delegate (signature preserved)."""
        import inspect

        sys.path.insert(0, str(REPO / "scripts"))
        import evaluate_paired_experiment as hub

        sig = inspect.signature(hub.build_pipeline)
        assert list(sig.parameters) == ["adapter_zh"]
        assert "PlanRunner" in inspect.getsource(hub.build_pipeline)
