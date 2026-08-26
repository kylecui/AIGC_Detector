"""W11-1: global temperature scaling on cached detection scores.

v3.1 arbitration: temperature scaling treats D1 (calibration honesty), NOT
the FN-1 verdict error. Safety is provable: sigmoid(logit(p)/T) > 0.5 iff
p > 0.5 — no label flips, no ranking changes, decisions identical.

Fitting data: 62 human probe docs + 320 AI-side records. The AI corpus is
2:1 hard-arm tilted (arm B oversamples evasion) — class-balanced weights
(human:AI = 50:50 prior correction) with arm/model composition preserved.
Evaluated per-slice so no slice hides behind the average.

Usage: uv run python scripts/fit_global_temperature.py
Output: models/calibration/global_temperature.json + console report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from defensibility_report import ece  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

ROOT = Path(__file__).parent.parent


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> int:
    # --- load scores ---
    ai = [
        json.loads(line)
        for line in
        (ROOT / "dataset/paired_generation_v1/eval_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recs = {
        json.loads(line)["id"]: json.loads(line)
        for line in
        (ROOT / "dataset/paired_generation_v1/pilot_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    hum = json.loads((ROOT / "reports/human_probe_trial_results.json").read_text(encoding="utf-8"))

    # human slice
    h_p = np.array([r["p_ai"] for r in hum])
    h_y = np.zeros(len(hum))

    # AI slices by model|arm
    slices: dict[str, list[float]] = {}
    ai_p_all, ai_y_all, ai_w_all = [], [], []
    for r in ai:
        rec = recs.get(r["id"])
        if not rec:
            continue
        key = f"{rec['model'].split('/')[-1]}|arm{rec['arm']}"
        slices.setdefault(key, []).append(r["p_ai"])
        ai_p_all.append(r["p_ai"])
        ai_y_all.append(1.0)
        # class-balanced weight: total AI weight == total human weight
        ai_w_all.append(len(hum) / len(ai))

    p_all = np.concatenate([h_p, np.array(ai_p_all)])
    y_all = np.concatenate([h_y, np.array(ai_y_all)])
    w_all = np.concatenate([np.ones(len(hum)), np.array(ai_w_all)])

    # --- fit temperature (NLL on class-balanced weights) ---
    z = logit(p_all)

    def nll(temp: float) -> float:
        q = sigmoid(z / max(temp, 1e-3))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return float(-np.sum(w_all * (y_all * np.log(q) + (1 - y_all) * np.log(1 - q))))

    res = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    temp = float(res.x)

    # --- report: ECE before/after per slice ---
    def conf_of(p: np.ndarray, label_is_ai: bool) -> np.ndarray:
        """confidence = prob assigned to the PREDICTED label (as deployed)."""
        return np.where(label_is_ai, p, 1 - p)

    def conf_after(p: np.ndarray, label_is_ai: bool) -> np.ndarray:
        q = sigmoid(logit(p) / temp)
        return np.where(label_is_ai, q, 1 - q)

    lines = ["=== W11-1 global temperature scaling ===", f"fitted T = {temp:.3f} (class-balanced NLL, n=382)", ""]

    def row(name: str, p: np.ndarray, is_ai: np.ndarray) -> str:
        # predicted-AI side per element: confidence = p if predicting AI else 1-p
        pred_ai = p >= 0.5
        ok = (pred_ai == is_ai).astype(int)
        pairs_b = [(float(c), int(o)) for c, o in zip(conf_of(p, pred_ai), ok)]
        pairs_a = [(float(c), int(o)) for c, o in zip(conf_after(p, pred_ai), ok)]
        e_b, e_a = ece(pairs_b), ece(pairs_a)
        hc_b = sum(1 for c, o in pairs_b if c > 0.8 and not o)
        hc_a = sum(1 for c, o in pairs_a if c > 0.8 and not o)
        return (f"{name:<26} ECE {e_b:.3f} -> {e_a:.3f}   "
                f"high-conf errors {hc_b} -> {hc_a}")

    # human slice
    lines.append(row("human (n=62)", h_p, np.zeros(len(h_p), dtype=bool)))
    # AI slices
    for key in sorted(slices):
        ps = np.array(slices[key])
        lines.append(row(f"{key} (n={len(ps)})", ps, np.ones(len(ps), dtype=bool)))
    # overall
    lines.append(row("overall (n=382)", p_all, y_all.astype(bool)))

    # --- FN-1 replay confidence compression ---
    fn1_p = 0.1091  # frozen anchor (tests/fixtures + reports/fn1_replay_w2.json)
    fn1_conf_b = 1 - fn1_p
    fn1_conf_a = 1 - float(sigmoid(logit(np.array([fn1_p])) / temp)[0])
    lines += [
        "",
        f"FN-1 replay: p_ai={fn1_p:.4f}  confidence {fn1_conf_b:.4f} -> {fn1_conf_a:.4f} "
        f"(verdict unchanged: Human-written — by construction no label flips)",
    ]

    # --- safety proof check: zero decision flips on all 382 ---
    flips = int(np.sum((p_all >= 0.5) != (sigmoid(z / temp) >= 0.5)))
    lines.append(f"decision flips at 0.5 threshold: {flips} (provably 0 for T>0)")

    # --- persist artifact (production untouched; applied only in W11-2 review) ---
    out = ROOT / "models/calibration/global_temperature.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "T": temp,
        "method": "global temperature scaling on cached ensemble p_ai (logit domain)",
        "fit": {"n": 382, "human": 62, "ai": 320, "weighting": "class-balanced 50:50"},
        "safety": {"label_flips": flips, "ranking_invariance": "monotone map (T>0)"},
        "fn1_replay": {"p_ai": fn1_p, "conf_before": fn1_conf_b, "conf_after": fn1_conf_a},
        "applied": False,  # becomes True only after W11-2 register review
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    lines += ["", f"artifact: {out} (applied=false — pending W11-2 review)"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
