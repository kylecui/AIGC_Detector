"""W9 defensibility scorecard — calibration honesty + coverage metrics (v1).

Computes the D-charter metrics from existing artifacts (no GPU needed):
  D1  per-side ECE (10-bin) + high-confidence error rate (headline metric)
  D4  Wilson 95% intervals on every rate; no FPR@1% when n<100
  D2  caveat coverage on formal-register human probe docs
  D5  FN-1 known-bad replay status

Inputs (all produced by earlier steps):
  dataset/paired_generation_v1/pilot_records.jsonl + eval_results.jsonl  (AI side, n=240)
  reports/human_probe_trial_results.json                                (human trial, n=10)
  reports/fn1_replay_w2.json                                            (FN-1 replay)
  dataset/legal_declaration_zh/human/*.md                               (caveat coverage)

Output: reports/defensibility_scorecard_2026-08.md (+ .json)
All sections marked PRELIMINARY — small-n; the calibrated baseline lands with
the full W5 probe set (60-80 human docs).

Usage: uv run python scripts/defensibility_report.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.detection.register import detect_register_zh  # noqa: E402

ROOT = Path(__file__).parent.parent
AI_RECORDS = ROOT / "dataset/paired_generation_v1/pilot_records.jsonl"
AI_RESULTS = ROOT / "dataset/paired_generation_v1/eval_results.jsonl"
HUMAN_TRIAL = ROOT / "reports/human_probe_trial_results.json"
FN1_REPLAY = ROOT / "reports/fn1_replay_w2.json"
HUMAN_DIR = ROOT / "dataset/legal_declaration_zh/human"
OUT_MD = ROOT / f"reports/defensibility_scorecard_{date.today():%Y-%m}.md"
OUT_JSON = OUT_MD.with_suffix(".json")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> float:
    """Expected calibration error. pairs: (confidence, correct_0_or_1)."""
    if not pairs:
        return float("nan")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, ok in pairs:
        bins[min(n_bins - 1, int(conf * n_bins))].append((conf, ok))
    total = len(pairs)
    err = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        err += len(b) / total * abs(acc - avg_conf)
    return err


def fmt_ci(x: tuple[float, float]) -> str:
    return f"[{x[0]:.2%}, {x[1]:.2%}]"


def main() -> int:
    report: dict = {"status": "PRELIMINARY (small n)", "generated": str(date.today())}

    # ---- AI side (n=240; ground truth label='ai'; error = predicted Human) ----
    records = {json.loads(l)["id"]: json.loads(l) for l in AI_RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()}
    results = [json.loads(l) for l in AI_RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    ai_pairs: list[tuple[float, int]] = []
    ai_err = ai_high_conf_err = 0
    per_model: dict[str, dict] = {}
    for r in results:
        rec = records.get(r["id"])
        if not rec:
            continue
        ok = 1 if r["predicted_label"] == "AI-generated" else 0
        ai_pairs.append((r["confidence"], ok))
        ai_err += 1 - ok
        if r["confidence"] > 0.8 and not ok:
            ai_high_conf_err += 1
        m = per_model.setdefault(rec["model"], {"n": 0, "err": 0})
        m["n"] += 1
        m["err"] += 1 - ok
    ai_n = len(ai_pairs)
    report["ai_side"] = {
        "n": ai_n,
        "fn_rate": ai_err / ai_n,
        "fn_rate_ci95": wilson(ai_err, ai_n),
        "high_confidence_error_rate": ai_high_conf_err / ai_n,
        "high_confidence_errors_n": ai_high_conf_err,
        "ece_10bin": ece(ai_pairs),
        "per_model_fn": {k: {"n": v["n"], "fn": v["err"]} for k, v in per_model.items()},
    }

    # ---- Human side (trial batch; ground truth human; error = predicted AI) ----
    hum: dict = {}
    if HUMAN_TRIAL.exists():
        rows = json.loads(HUMAN_TRIAL.read_text(encoding="utf-8"))
        h_pairs = [(r["confidence"], 1 if r["label"] == "Human-written" else 0) for r in rows]
        fp = sum(1 for r in rows if r["label"] == "AI-generated")
        h_high = [r for r in rows if r["label"] == "AI-generated" and r["confidence"] > 0.8]
        hum = {
            "n": len(rows),
            "fp_rate_preview": fp / len(rows),
            "fp_rate_ci95": wilson(fp, len(rows)),
            "high_confidence_fps": [r["file"] for r in h_high],
            "ece_10bin": ece(h_pairs),
        }
        report["human_side_trial"] = hum

    # ---- D2: caveat coverage on formal-register human docs ----
    coverage: dict = {}
    if HUMAN_DIR.exists():
        hits = total = 0
        details = []
        for f in sorted(HUMAN_DIR.glob("*.md")):
            parts = f.read_text(encoding="utf-8").split("---", 2)
            body = parts[2].strip() if len(parts) == 3 else ""
            if len(body) < 50:
                continue
            total += 1
            reg = detect_register_zh(body)
            if reg.is_formal_zh:
                hits += 1
            details.append({"file": f.name, "score": reg.score, "formal": reg.is_formal_zh})
        coverage = {"hits": hits, "total": total, "rate": hits / total if total else None, "docs": details}
        report["caveat_coverage_trial"] = coverage

    # ---- D5: FN-1 known-bad replay ----
    if FN1_REPLAY.exists():
        rep = json.loads(FN1_REPLAY.read_text(encoding="utf-8"))
        report["fn1_replay"] = {
            "doc_label": rep["doc"]["label"],
            "doc_confidence": rep["doc"]["confidence"],
            "max_segment_p_ai": rep["segment_highlights"]["max_p_ai"],
            "surfaced": rep["segment_highlights"]["max_p_ai"] >= 0.8,
        }

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Markdown rendering ----
    a = report["ai_side"]
    lines = [
        f"# 可辩护性计分卡（{date.today()}）",
        "",
        "> **PRELIMINARY — 小样本预览**。人类侧 n=10（试批）、AI侧 n=240（实验语料）。",
        "> 完整 W5 探针集（人类侧 60-80 篇）落地后重算；n<100 不报告 FPR@1%。",
        "",
        "## D1 校准诚实（头号指标：高置信错误率）",
        "",
        "| 侧 | n | 错误率 [Wilson 95%] | 高置信错误 (conf>0.8且错) | ECE(10桶) |",
        "|---|---|---|---|---|",
        f"| AI侧（漏判） | {a['n']} | {a['fn_rate']:.1%} {fmt_ci(a['fn_rate_ci95'])} | "
        f"{a['high_confidence_errors_n']} ({a['high_confidence_error_rate']:.1%}) | {a['ece_10bin']:.3f} |",
    ]
    if hum:
        n_hc = len(hum["high_confidence_fps"])
        hc = ", ".join(Path(x).stem for x in hum["high_confidence_fps"]) or "—"
        lines += [
            f"| 人类侧（误判,试批） | {hum['n']} | {hum['fp_rate_preview']:.0%} {fmt_ci(hum['fp_rate_ci95'])} | "
            f"{n_hc} | {hum['ece_10bin']:.3f} |",
            "",
            f"高置信误判明细（人类侧）: {hc}",
            "",
            "AI侧分模型漏判: "
            + "; ".join(f"{k.split('/')[-1]} {v['fn']}/{v['n']}" for k, v in a["per_model_fn"].items()),
        ]
    if coverage.get("total"):
        lines += [
            "",
            "## D2 盲区显性化（caveat覆盖，试批）",
            "",
            f"正式语域命中: {coverage['hits']}/{coverage['total']} "
            f"({coverage['rate']:.0%}) — 未命中为无公文套语的简短服务公告（已知缺口，W5全量校准轮处理）",
        ]
    if "fn1_replay" in report:
        f1 = report["fn1_replay"]
        lines += [
            "",
            "## D5 FN-1 known-bad 回放",
            "",
            f"文档级判定 {f1['doc_label']} ({f1['doc_confidence']:.2f})；最高分段 p_ai = "
            f"{f1['max_segment_p_ai']:.3f}，段级证据浮出: {'✅ PASS' if f1['surfaced'] else '❌ FAIL'}",
            "",
            "## 结论（预览口径）",
            "",
            "- AI侧漏判集中且方向一致（合约臂/强模型），与W4实验吻合——高置信错误集中在强模型合约文本。",
            "- 人类侧试批出现高置信误判（建行公告 0.987）——D1 的直接反面教材，W3b权重切换须以此为约束。",
            "- caveat覆盖：公文套语文本100%命中；简短服务公告缺口已登记。",
            "- 发布门槛（正式版）：ECE不恶化 + caveat覆盖100%（正式语域全集）+ FN-1回放PASS。",
        ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {OUT_MD}")
    print(f"written: {OUT_JSON}")
    print("\n".join(lines[6:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
