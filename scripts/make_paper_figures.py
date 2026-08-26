"""Paper figures for W13 (specification-slack paper).

Generates from on-disk experiment data (no hardcoded numbers; prints
cross-check values for verification against DETECTOR_NOTES tables):

  Fig 1  5x4 miss-rate heatmap (W4c zh, the single-competence-cell visual)
  Fig 2  specification-slack interaction scatter (formal dA-B vs casual
         dC-D per model, with bootstrap CIs, diagonal = no register effect)
  Fig 3  dose-response: W4's 3-point "monotone ladder" vs W4c's 5-point
         non-monotone reality (family > parameter count)
  Fig 4  corpus-level pooling AUROC vs k (the DeepSeek inversion)

Output: docs/research/figures/fig{N}_*.pdf (+ .png previews)
Usage:  uv run python scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset/paired_generation_v1"
OUT = ROOT / "docs/research/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

LADDER = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "THUDM/GLM-4-9B-0414",
          "Qwen/Qwen3-14B", "deepseek-ai/DeepSeek-V3.2"]
SHORT = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen/Qwen3-8B": "Qwen3-8B",
         "THUDM/GLM-4-9B-0414": "GLM-4-9B", "Qwen/Qwen3-14B": "Qwen3-14B",
         "deepseek-ai/DeepSeek-V3.2": "DeepSeek-V3.2"}
PARAMS = {"Qwen/Qwen2.5-7B-Instruct": "7B", "Qwen/Qwen3-8B": "8B",
          "THUDM/GLM-4-9B-0414": "9B", "Qwen/Qwen3-14B": "14B",
          "deepseek-ai/DeepSeek-V3.2": "MoE"}
ARMS = [("formal", "A"), ("formal", "B"), ("casual", "C"), ("casual", "D")]
ARM_LBL = ["formal\nfree", "formal\ncontract", "casual\nfree", "casual\ncontract"]
THR = 0.47

# ---- load ----
w4c_res = [json.loads(l) for l in (DATA / "w4c_eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
w4c_rec = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "w4c_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
w4_res = [json.loads(l) for l in (DATA / "eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
w4_rec = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "pilot_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
analysis = json.loads((DATA / "w4c_analysis.json").read_text(encoding="utf-8"))
pool = json.loads((ROOT / "reports/w14_corpus_level.json").read_text(encoding="utf-8"))


def cell_stats(res, recs):
    """(register, arm) -> {model: [p_ai...]} + miss counts."""
    agg = {}
    for r in res:
        rec = recs.get(r["id"])
        if rec is None:
            continue
        agg.setdefault((rec["register"], rec["arm"]), {}).setdefault(rec["model"], []).append(
            (r["p_ai"], r["predicted_label"] != "AI-generated"))
    return agg


agg = cell_stats(w4c_res, w4c_rec)

# ================= FIGURE 1: miss-rate heatmap =================
miss = np.full((5, 4), np.nan)
for j, (reg, arm) in enumerate(ARMS):
    for i, m in enumerate(LADDER):
        rows = agg.get((reg, arm), {}).get(m, [])
        if rows:
            miss[i, j] = np.mean([r[1] for r in rows])

print("Fig1 cross-check (miss rates %):")
for i, m in enumerate(LADDER):
    print(" ", SHORT[m], [f"{v:.0%}" if not np.isnan(v) else "-" for v in miss[i]])

fig, ax = plt.subplots(figsize=(3.5, 2.6))
im = ax.imshow(miss, cmap="Reds", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(4), ARM_LBL)
ax.set_yticks(range(5), [f"{SHORT[m]}\n({PARAMS[m]})" for m in LADDER])
for i in range(5):
    for j in range(4):
        v = miss[i, j]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                color="white" if v > 0.55 else "#333333", fontsize=8)
ax.set_title("Miss rate at deployed threshold (ZH path, W4c)")
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("miss rate", fontsize=8)
# outline the competence cells individually (miss <= 20%): GLM formal-A (9%),
# Qwen2.5-7B formal-A (15%) — non-adjacent rows, so per-cell outlines
for (i, j) in [(2, 0), (0, 0)]:
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#0a7", lw=1.8))
ax.text(0.02, 0.99, "competence cells (miss $\\leq$ 20%)", transform=ax.transAxes,
        va="top", fontsize=7, color="#0a7")
fig.savefig(OUT / "fig1_miss_heatmap.pdf")
fig.savefig(OUT / "fig1_miss_heatmap.png")
plt.close(fig)

# ================= FIGURE 2: interaction scatter =================
fig, ax = plt.subplots(figsize=(3.3, 3.0))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
rng = np.random.default_rng(3)
for c, m in zip(colors, LADDER):
    fd = analysis["contrasts"].get(f"H1-replication|{m}")
    cd = analysis["contrasts"].get(f"H2-formality-vs-contract|{m}")
    if not fd or not cd:
        continue
    fx, fy = fd["mean_diff"], cd["mean_diff"]
    flo, fhi = fd.get("boot_ci95", [fx, fx])
    clo, chi = cd.get("boot_ci95", [fy, fy])
    ax.errorbar(fx, fy, xerr=[[fx - flo], [fhi - fx]], yerr=[[fy - clo], [chi - fy]],
                fmt="o", color=c, ms=5, capsize=2, lw=1)
    dx = 0.015 if m != "Qwen/Qwen3-14B" else -0.015
    ha = "left" if dx > 0 else "right"
    ax.annotate(SHORT[m], (fx, fy), xytext=(fx + dx, fy + 0.012), fontsize=7, ha=ha, color=c)
lims = [-0.05, 0.62]
ax.plot(lims, lims, ls="--", c="gray", lw=0.8, zorder=0)
ax.text(0.40, 0.44, "y = x\n(no register effect)", fontsize=6.5, color="gray", rotation=38)
ax.axhline(0, c="lightgray", lw=0.6)
ax.axvline(0, c="lightgray", lw=0.6)
ax.set_xlim(lims[0], lims[1])
ax.set_ylim(-0.06, 0.45)
ax.set_xlabel(r"formal contract effect  $\Delta$(free $-$ contract)")
ax.set_ylabel(r"casual contract effect  $\Delta$(free $-$ contract)")
ax.set_title("Specification slack by register\n(below diagonal = formal-path collapse)")
fig.savefig(OUT / "fig2_interaction.pdf")
fig.savefig(OUT / "fig2_interaction.png")
plt.close(fig)
print("Fig2 cross-check (formal d, casual d):")
for m in LADDER:
    fd = analysis["contrasts"].get(f"H1-replication|{m}", {}).get("mean_diff")
    cd = analysis["contrasts"].get(f"H2-formality-vs-contract|{m}", {}).get("mean_diff")
    print(f"  {SHORT[m]:<14} {fd:+.3f} {cd:+.3f}")

# ================= FIGURE 3: dose-response =================
# W4 first run: arm-B ensemble means, topics t01-t40 only, 3 models
w4_b = {}
for r in w4_res:
    rec = w4_rec.get(r["id"])
    if not rec or rec["arm"] != "B":
        continue
    if not rec["topic_id"].startswith("t0") and not rec["topic_id"].startswith("t1") and not rec["topic_id"].startswith("t2") and not rec["topic_id"].startswith("t3") and not rec["topic_id"].startswith("t40"):
        continue
    w4_b.setdefault(rec["model"], []).append(r["p_ai"])
w4_pts = {}
for m in ["Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414", "deepseek-ai/DeepSeek-V3.2"]:
    if w4_b.get(m):
        w4_pts[m] = float(np.mean(w4_b[m]))
# W4c: arm B means from analysis cells
w4c_b = {}
for m in LADDER:
    key = f"formal|B|{m}"
    if key in analysis["cells"]:
        w4c_b[m] = analysis["cells"][key]["mean"]

pos = {m: i for i, m in enumerate(LADDER)}
fig, ax = plt.subplots(figsize=(3.5, 2.7))
xs = [pos[m] for m in LADDER]
ys = [w4c_b.get(m, np.nan) for m in LADDER]
lo = [analysis["cells"][f"formal|B|{m}"]["ci95"][0] for m in LADDER]
hi = [analysis["cells"][f"formal|B|{m}"]["ci95"][1] for m in LADDER]
ax.fill_between(xs, lo, hi, alpha=0.18, color="C0", label="W4c 95% CI")
ax.plot(xs, ys, "o-", color="C0", lw=1.4, ms=4.5, label="W4c (5 models, 5 seeds)")
w4_x = [pos[m] for m in w4_pts]
w4_y = [w4_pts[m] for m in w4_pts]
ax.plot(w4_x, w4_y, "s--", color="C3", lw=1.2, ms=5, label="W4 first run (3 points)")
ax.axhline(THR, c="gray", ls=":", lw=0.9)
ax.text(3.55, THR + 0.015, "decision\nthreshold", fontsize=6.5, color="gray")
ax.set_xticks(xs, [f"{SHORT[m]}\n({PARAMS[m]})" for m in LADDER])
ax.set_ylabel(r"ensemble $p_{AI}$, formal-contract arm")
ax.set_title("Dose-response across the capability ladder\nW4's 3-point monotone trend does not survive 5 models")
ax.legend(loc="upper right", frameon=False)
fig.savefig(OUT / "fig3_dose_response.pdf")
fig.savefig(OUT / "fig3_dose_response.png")
plt.close(fig)
print(f"Fig3 cross-check: W4 3pts = {({SHORT[m]: round(v,3) for m,v in w4_pts.items()})}")
print(f"  W4c 5pts = {({SHORT[m]: round(v,3) for m,v in w4c_b.items()})}")

# ================= FIGURE 4: pooling inversion =================
fig, ax = plt.subplots(figsize=(3.3, 2.6))
styles = {
    "GLM-4-9B-0414-A": ("C0", "o-"), "GLM-4-9B-0414-B": ("C0", "o--"),
    "DeepSeek-V3.2-A": ("C3", "s-"), "DeepSeek-V3.2-B": ("C3", "s--"),
    "Qwen3-8B-B": ("C2", "^--"), "Qwen3-14B-B": ("C4", "v--"),
}
for key, (c, st) in styles.items():
    cell = pool["cells"].get(key)
    if not cell:
        continue
    ks = sorted(int(k) for k in cell if k.isdigit())
    ax.plot(ks, [cell[str(k)]["auroc"] for k in ks], st, color=c, lw=1.2, ms=4,
            label=key.replace("-0414", ""))
ax.axhline(0.5, c="gray", ls=":", lw=0.9)
ax.text(15.5, 0.52, "chance", fontsize=6.5, color="gray")
ax.set_xlabel("corpus size $k$ (documents pooled)")
ax.set_ylabel("AUROC vs human corpus")
ax.set_title("Corpus-level pooling (mean-$p$)\nsolid=free, dashed=contract; DeepSeek inverts")
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.set_ylim(-0.02, 1.05)
fig.savefig(OUT / "fig4_pooling.pdf")
fig.savefig(OUT / "fig4_pooling.png")
plt.close(fig)
print("Fig4 done")

print(f"\nall figures -> {OUT}")
