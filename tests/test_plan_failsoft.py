"""Regression: diagnostic-stage loading is fail-soft (v0.3.1).

The 2026-08-26 serve crash: plan-declared examples.stages.ttr_stage was
unimportable under the console-script path (no repo cwd in sys.path) and
PlanRunner re-raised, killing startup. Contract discipline: third-party
evidence must never break the service — load failures skip with a warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.plan import PlanRunner  # noqa: E402


def test_unimportable_diagnostic_stage_skipped(tmp_path, caplog):
    plan = PlanRunner.default().plan
    plan = {**plan, "diagnostic_stages": [
        {"id": "ghost", "impl": "no.such.module:Ghost"},
    ]}
    (tmp_path / "p.yaml").write_text(
        __import__("yaml").safe_dump(plan, allow_unicode=True), encoding="utf-8"
    )
    runner = PlanRunner.from_yaml(tmp_path / "p.yaml")
    # build() would load models; test the loader directly (unit scope)
    stages = runner._load_diagnostic_stages()  # noqa: SLF001
    assert stages == {}
    assert any("ghost" in r.message and ("skipping" in r.message or "unavailable" in r.message)
               for r in caplog.records)


def test_loadable_stage_still_works():
    runner = PlanRunner.default()
    stages = runner._load_diagnostic_stages()  # noqa: SLF001
    # in the repo layout the ttr example must load (tests run from repo root)
    assert "ttr" in stages
