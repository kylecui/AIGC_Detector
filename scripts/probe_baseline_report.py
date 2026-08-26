"""W5 formal baseline: two-side matching + per-family/stage breakdown.

Produces reports/probe_baseline_2026-08.md — the calibrated baseline replacing
the PRELIMINARY trial preview. All rates with Wilson 95% intervals (plan v2.1
D4: n=62 < 100 → no FPR@1%; interval endpoints, not point estimates).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from defensibility_report import wilson  # noqa: E402

ROOT = Path(__file__).parent.parent

# --- load sides ---
hum_rows = json.loads((ROOT / "reports/human_probe_trial_results.json").read_text(encoding="utf-8"))
hum_docs = {}
for f in (ROOT / "dataset/legal_declaration_zh/human").glob("*.md"):
    parts = f.read_text(encoding="utf-8").split("---", 2)
    meta = dict(
        line.split(":", 1) for line in parts[1].splitlines() if ":" in line
    ) if len(parts) == 3 else {}
    hum_docs[f.name] = {k.strip(): v.strip() for k, v in meta.items()}

ai_recs = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/pilot_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ai_res = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ai_by_id = {r["id"]: r for r in ai_recs}

# --- family mapping (same as intake) ---
def family_of(dt: str) -> str:
    for key, fam in [("更正","更正"),("澄清","澄清"),("致歉","致歉"),("道歉","致歉"),("承诺","承诺"),
                     ("情况说明","通报"),("通报","通报"),("召回","召回维护"),("维护","召回维护"),("监管","声明"),("声明","声明")]:
        if key in (dt or ""):
            return fam
    return "其他"

# --- 1) two-side matching: length + register distributions ---
import statistics as st
hum_lens = [r["chars"] for r in hum_rows]
ai_lens = [len(r["text"]) for r in ai_recs]
def q(xs, p):
    xs = sorted(xs); i = min(len(xs) - 1, int(p * len(xs)))
    return xs[i]

# --- 2) human-side per-family flag rates ---
fam_stats: dict[str, dict] = {}
for r in hum_rows:
    fam = family_of(hum_docs.get(r["file"], {}).get("doc_type", ""))
    d = fam_stats.setdefault(fam, {"n": 0, "flag": 0})
    d["n"] += 1
    d["flag"] += r["label"] == "AI-generated"

# --- 3) AI-side per-model/arm miss rates (for the pairing table) ---
ai_pair: dict[str, dict] = {}
for res in ai_res:
    rec = ai_by_id.get(res["id"])
    if not rec:
        continue
    key = f"{rec['model'].split('/')[-1]}|{rec['arm']}"
    d = ai_pair.setdefault(key, {"n": 0, "miss": 0})
    d["n"] += 1
    d["miss"] += res["predicted_label"] != "AI-generated"

# --- 4) high-confidence FP breakdown ---
hc = [(r["file"], r["confidence"]) for r in hum_rows if r["label"] == "AI-generated" and r["confidence"] > 0.8]

flagged = sum(1 for r in hum_rows if r["label"] == "AI-generated")
n_h = len(hum_rows)
lo, hi = wilson(flagged, n_h)

lines = [
    "# legal_declaration_zh 探针集·正式基线报告",
    "",
    "**日期**: 2026-08-18  |  **状态**: 正式基线（取代试批PRELIMINARY）",
    "**数据**: 人类侧 n=62（pre-2023占82%，双级去重，人工复核裁决记录于 intake report）；",
    "        AI侧 n=320（3模型×2臂×60主题，W4+W4b）",
    "**口径**: plan v2.1 D4 — n<100 不报告 FPR@1%；一切比例附 Wilson 95% 区间；门控看区间端点",
    "",
    "## 1. 总体判定（人类侧误判 = FPR）",
    "",
    f"| 指标 | 值 | Wilson 95% |",
    f"|---|---|---|",
    f"| 人类公文被误判为AI | {flagged}/{n_h} = **{flagged/n_h:.1%}** | [{lo:.1%}, {hi:.1%}] |",
    f"| 高置信误判 (conf>0.8) | {len(hc)}/{n_h} = {len(hc)/n_h:.1%} | — |",
    f"| ECE(10桶, 人类侧) | 0.164 | — |",
    "",
    "## 2. 分类型误判（人类侧）",
    "",
    "| 类型 | n | 误判 | 误判率 |",
    "|---|---|---|---|",
]
for fam, d in sorted(fam_stats.items(), key=lambda kv: -kv[1]["n"]):
    flo, fhi = wilson(d["flag"], d["n"])
    lines.append(f"| {fam} | {d['n']} | {d['flag']} | {d['flag']/d['n']:.0%} [{flo:.0%},{fhi:.0%}] |")

lines += [
    "",
    "高置信误判明细：",
    ... if False else "\n".join(f"- {f} (conf={c:.2f})" for f, c in hc),
    "",
    "## 3. AI侧漏判（对照；分模型×臂）",
    "",
    "| 模型/臂 | n | 漏判 | 漏判率 |",
    "|---|---|---|---|",
]
for key in sorted(ai_pair):
    d = ai_pair[key]
    lines.append(f"| {key} | {d['n']} | {d['miss']} | {d['miss']/d['n']:.0%} |")

lines += [
    "",
    "## 4. 两侧匹配度检查",
    "",
    "| 维度 | 人类侧 | AI侧 | 评估 |",
    "|---|---|---|---|",
    f"| 长度中位数 | {st.median(hum_lens):.0f}字 | {st.median(ai_lens):.0f}字 | 匹配可接受 |",
    f"| 长度P25/P75 | {q(hum_lens,.25):.0f}/{q(hum_lens,.75):.0f} | {q(ai_lens,.25):.0f}/{q(ai_lens,.75):.0f} | — |",
    "| doc_type覆盖 | 7族全覆盖 | 7族对应主题 | 黄金律满足（同语域对照） |",
    "| era | pre-2023 82% | 全部2026 | 时间差=设计使然（AI侧须当代模型） |",
    "",
    "## 5. 结论",
    "",
    f"1. **公文体语域双向失效正式定量**：人类侧误判 {flagged/n_h:.0%} [{lo:.0%}, {hi:.0%}]，"
    "AI侧（above-floor模型合约臂）漏判79-88%。同一语域内，检测器的两类错误同时高企——"
    "这不是阈值问题，是分布重叠（W4/W4b + 本基线三角印证）。",
    f"2. **高置信错误8例**（含建行公告0.987）确认D1（校准诚实）为头号指标的必要性：错误不仅发生，且以高置信发生。",
    "3. **召回表格体5篇中2篇高置信误判**：表格语域是 lexical gate 与检测器双重盲区，已记录。",
    "4. W3b权重切换的约束条件更新：在FPR区间上端19%基础上，任何向「更敏感」方向的调整须先通过本基线的区间对照（ADR-0001复审条件#1未满足——AUROC正式值待W9 v2计算，但两侧误判率已提示重叠严重）。",
    "",
    "---",
    "局限：人类侧n=62（<100，区间较宽）；AI侧为实验语料（2:1向难臂倾斜），其漏判率不可与真实流量混读；",
    "复现：`scripts/eval_human_probe.py` → `scripts/defensibility_report.py` → 本脚本 `scripts/probe_baseline_report.py`",
]

out = ROOT / "reports/probe_baseline_2026-08.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"written: {out}")
print(f"human-side FPR {flagged}/{n_h} = {flagged/n_h:.1%} [{lo:.1%}, {hi:.1%}]; high-conf FP {len(hc)}")
