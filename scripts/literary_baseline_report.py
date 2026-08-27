"""Literary-prose zh: preliminary baseline report (human n=40 vs AI eval_results).

KEY question: does ANY single stage separate literary AI prose from literary
human prose? Outputs per-stage mean p_ai per side + rank-AUC separation,
human-side flag rate and AI-side miss rate with Wilson 95% CIs.
Writes reports/literary_baseline_2026-08.md (PRELIMINARY).
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from defensibility_report import wilson  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset/literary_prose_zh"

# --- load ---
hum = [json.loads(l) for l in (DATA / "human_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ai_all = [json.loads(l) for l in (DATA / "eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# analysis unit = the designed 270-job grid (30 topics x 3 models x 3 seeds).
# The records file contains duplicate generations of some jobs (two concurrent
# generator instances) and legacy rows whose id != sha1(model|topic|seed);
# dedup by (model, topic, seed) first-seen is immune to both.
ai_by_job: dict[tuple, dict] = {}
for r in ai_all:
    ai_by_job.setdefault((r["model"], r["topic"], r["seed"]), r)
ai = list(ai_by_job.values())

rec_lines = len([l for l in (DATA / "ai_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
rec_jobs: set[tuple] = set()
for l in (DATA / "ai_records.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l)
        rec_jobs.add((r["model"], r["topic_id"], r["seed"]))

STAGES = ["statistical", "linguistic", "encoder", "binoculars"]

# --- (a) human-side flag rate ---
h_flag = sum(1 for r in hum if r["label"] == "AI-generated")
n_h = len(hum)
h_lo, h_hi = wilson(h_flag, n_h)
h_hc = sum(1 for r in hum if r["label"] == "AI-generated" and r["conf"] > 0.8)

# --- (b) AI-side miss rate, overall + per model ---
n_ai = len(ai)
a_miss = sum(1 for r in ai if r["label"] != "AI-generated")
a_lo, a_hi = wilson(a_miss, n_ai)

per_model: dict[str, dict] = {}
for r in ai:
    m = r["model"].split("/")[-1]
    d = per_model.setdefault(m, {"n": 0, "miss": 0})
    d["n"] += 1
    d["miss"] += r["label"] != "AI-generated"

# --- (c) per-stage separation (KEY) ---


def auc(xs: list[float], ys: list[float]) -> float:
    """P(random AI stage score > random human stage score), ties=0.5."""
    if not xs or not ys:
        return float("nan")
    wins = 0.0
    for x in xs:
        for y in ys:
            wins += 1.0 if x > y else (0.5 if x == y else 0.0)
    return wins / (len(xs) * len(ys))


stage_rows = []
for s in STAGES:
    hv = [r["stage_p_ai"][s] for r in hum if s in r.get("stage_p_ai", {})]
    av = [r["stage_p_ai"][s] for r in ai if s in r.get("stage_p_ai", {})]
    if not hv or not av:
        stage_rows.append((s, len(hv), len(av), None, None, None, None, None))
        continue
    a = auc(av, hv)
    hf = sum(1 for v in hv if v >= 0.5) / len(hv)
    af = sum(1 for v in av if v >= 0.5) / len(av)
    thr = max(hv)
    zfp = sum(1 for v in av if v > thr) / len(av)
    stage_rows.append((s, len(hv), len(av), st.mean(hv), st.mean(av), a, (hf, af), zfp))

# ensemble separation for reference
ens_a = auc([r["p_ai"] for r in ai], [r["p_ai"] for r in hum])
ens_thr = max(r["p_ai"] for r in hum)
ens_zfp = sum(1 for r in ai if r["p_ai"] > ens_thr) / n_ai

verdict = {
    s: ("分离" if row[5] is not None and row[7] is not None and (row[5] > 0.67 or row[5] < 0.33) else "不分离")
    for s, row in zip(STAGES, stage_rows)
}

lines = [
    "# literary_prose_zh 文学散文语域·初步基线报告",
    "",
    "**日期**: 2026-08-27  |  **状态**: PRELIMINARY（人类侧 n=40，AI侧去重后 n=%d/%d 个生成作业）" % (n_ai, len(rec_jobs)),
    "**数据**: 人类侧 = 40篇经典/当代文学散文过管道（human_eval.jsonl）；",
    "        AI侧 = literary-v1 生成记录（GLM-4-9B / Qwen3-8B / DeepSeek-V3.2，arm=L）经 literary_prose_eval.py 评估",
    "**口径**: 比例附 Wilson 95% 区间；样本量小，看区间端点而非点估计；PPL 分带未落盘，本报告不含（见§5）",
    "",
    "## 1. 总体判定",
    "",
    "| 侧 | 指标 | 值 | Wilson 95% |",
    "|---|---|---|---|",
    f"| 人类散文被误判为AI | flag rate | {h_flag}/{n_h} = **{h_flag/n_h:.1%}** | [{h_lo:.1%}, {h_hi:.1%}] |",
    f"| 高置信误判 (conf>0.8) | — | {h_hc}/{n_h} = {h_hc/n_h:.1%} | — |",
    f"| AI散文被漏判 | miss rate | {a_miss}/{n_ai} = **{a_miss/n_ai:.1%}** | [{a_lo:.1%}, {a_hi:.1%}] |",
    "",
    "## 2. AI侧分模型漏判",
    "",
    "| 模型 | n | 漏判 | 漏判率 |",
    "|---|---|---|---|",
]
for m, d in sorted(per_model.items(), key=lambda kv: -kv[1]["n"]):
    mlo, mhi = wilson(d["miss"], d["n"])
    lines.append(f"| {m} | {d['n']} | {d['miss']} | {d['miss']/d['n']:.0%} [{mlo:.0%},{mhi:.0%}] |")

lines += [
    "",
    "## 3. 分阶段分离度（核心问题：任一阶段能否把文学AI与文学人分开？）",
    "",
    "| 阶段 | 人类侧n | AI侧n | 人类侧均值p_ai | AI侧均值p_ai | AUC(P_AI>人类) | @0.5: 人类误标/AI捕获 | 零人类FP捕获率* | 判读 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for (s, nh_, na_, hm, am, a, fr, zfp) in stage_rows:
    if a is None:
        lines.append(f"| {s} | {nh_} | {na_} | — | — | — | — | — | 单侧缺失，无法比较 |")
    else:
        lines.append(
            f"| {s} | {nh_} | {na_} | {hm:.4f} | {am:.3f} | {a:.3f} | {fr[0]:.0%} / {fr[1]:.0%} | {zfp:.0%} | {verdict[s]} |"
        )
ens_hf = sum(1 for r in hum if r["p_ai"] >= 0.5) / n_h
ens_af = sum(1 for r in ai if r["p_ai"] >= 0.5) / n_ai
lines += [
    f"| （集成整体 p_ai） | {n_h} | {n_ai} | {st.mean(r['p_ai'] for r in hum):.4f} | "
    f"{st.mean(r['p_ai'] for r in ai):.3f} | {ens_a:.3f} | {ens_hf:.0%} / {ens_af:.0%} | {ens_zfp:.0%} | 参考 |",
    "",
    "\\* 零人类FP捕获率 = 以该阶段人类侧最大值为阈值时的AI捕获率（最保守工作点）。",
    "",
    "AUC 判读标准（探索性）：|AUC-0.5|>0.17 视为有分离迹象。注意：人类侧各阶段得分普遍贴近地板（如 encoder 人类最大值仅0.0047），"
    "AUC 会被大量近地板配对抬高——判断实用性以「零人类FP捕获率」这一保守列为准。",
    "",
    "## 4. 数据质量注记",
    "",
    f"- ai_records.jsonl 共 {rec_lines} 行、{len(rec_jobs)} 个唯一生成作业（设计网格 30题×3模型×3种子=270）："
    "存在同作业重复生成（两个生成实例并发）与 46 行旧 id 方案（id≠sha1(model|topic|seed)）。"
    "本报告按作业键 (model,topic,seed) 去重（首现优先）；生成侧去重与 id 方案统一待修复。",
    "- 人类侧 stage_p_ai 不含 binoculars（评估时该阶段缺席或被跳过）；AI侧 binoculars 由 agree-rule 经常跳过，"
    "故该阶段两侧 n 均小且不成对，其 AUC 仅作方向参考。",
    "",
    "## 5. PPL 分带对比",
    "",
    "不可得：literary_prose_eval.py 只落盘了各阶段 p_ai（stage_p_ai），PPL 原始值未写入 eval_results.jsonl。"
    "若后续需要，须在评估脚本中加存 breakdown 中的 ppl 字段后重测。",
    "",
    "## 6. 结论（PRELIMINARY，n=40/%d）" % n_ai,
    "",
]
sep_stages = [s for s in STAGES if verdict[s] == "分离"]
lines += [
    f"1. **人类侧零误判，与预期相反**：40篇文学散文全部判为人类（flag 0.0% [0.0%, 8.8%]，最高 p_ai 仅 "
    f"{max(r['p_ai'] for r in hum):.3f}）。任务书中预期的人类侧 flag≈47% 未复现——该预期可能来自早期部分数据或其他口径，"
    "须以本全量口径为准并回溯差异来源。",
]
if sep_stages:
    lines.append(
        f"2. **有阶段出现分离迹象：{('、'.join(sep_stages))}**。其中 encoder AUC≈0.89 但被近地板配对抬高："
        "其 0.5 阈值仅捕获 9% 的AI散文；在最保守工作点（阈值=人类侧最大值0.0047）也只捕获 55%。"
        "结论：**特征存在弱-中等分离，但现有校准阈值完全错过AI得分主体**。"
    )
else:
    lines.append("2. 无任何阶段表现出分离迹象：各阶段 AUC 均落在 0.33–0.67 的重叠带内。")
lines += [
    f"3. **失效是单向的（漏判型），且瓶颈在集成决策而非阶段特征**：AI侧漏判 {a_miss/n_ai:.0%} [{a_lo:.0%}, {a_hi:.0%}]，"
    f"而人类侧 0 误判。encoder/集成的排序能力（AUC 0.89/0.85）没有被 0.5 决策阈值利用——文学语域的AI得分整体下移"
    "（AI侧 encoder 均值仅 0.109），沿用通用语域校准的阈值必然大面积漏判。",
    "4. 行动项：(a) 若要在文学语域可用，须做语域内重校准（如 encoder 阈值降至~0.005 量级，代价是逼近人类得分地板、"
    "泛化风险须另行验证）；(b) 修复生成侧 id 方案与并发重复生成；(c) 评估脚本加存 PPL 与 binoculars 人类侧，补齐§3/§5；"
    "(d) 扩样至 ≥100/侧 后出正式基线；能力声明盲区清单可先注明「文学散文：AI侧漏判高企」。",
    "",
    "---",
    "局限：两侧均小样本，区间宽；AI侧为单臂（arm=L）实验语料，漏判率不可与真实流量混读；",
    "人类侧为名家散文（时代语域偏移未审，pre-2023占比未量化）；AUC/零FP捕获率均为探索性统计，无显著性检验；",
    "binoculars 人类侧缺失导致该阶段无法比较。",
    "复现：`scripts/literary_prose_eval.py`（评估）→ `scripts/literary_baseline_report.py`（本报告）",
]

out = ROOT / "reports/literary_baseline_2026-08.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"written: {out}")
print(f"human flag {h_flag}/{n_h} = {h_flag/n_h:.1%} [{h_lo:.1%}, {h_hi:.1%}]; "
      f"AI miss {a_miss}/{n_ai} = {a_miss/n_ai:.1%} [{a_lo:.1%}, {a_hi:.1%}]")
print("\nper-stage table:")
for (s, nh_, na_, hm, am, a, fr, zfp) in stage_rows:
    if a is None:
        print(f"  {s:<12} human_n={nh_} ai_n={na_}  (one side missing)")
    else:
        print(f"  {s:<12} human_n={nh_} ai_n={na_}  human_mean={hm:.4f} ai_mean={am:.3f} "
              f"AUC={a:.3f} @0.5 human-flag={fr[0]:.0%} ai-catch={fr[1]:.0%} "
              f"zeroFP-catch={zfp:.0%}  -> {verdict[s]}")
print(f"  {'ensemble':<12} human_n={n_h} ai_n={n_ai}  AUC={ens_a:.3f} ai-catch@0.5={ens_af:.0%} zeroFP-catch={ens_zfp:.0%}")
