"""Breakthrough probe: lexical EN-formal gate + abstention — risk-coverage on local data.

The fatal pattern: EN path outputs high-confidence WRONG verdicts on formal
registers it never saw in calibration (71% of human formal EN flagged, 14
high-conf). Cheapest fix candidate: a zh-register-gate-style lexical EN
formal detector that triggers ABSTENTION (verdict downgraded to "uncovered
register") instead of a confident call.

Test on local data:
- 35 human EN formal docs (should be gated: catastrophe cut)
- 454 W4-EN AI cells (formal arms gated = honest abstain where AUROC 0.36-0.72
  anyway; casual arms should mostly pass the gate)
- zh texts (sanity: gate must NOT fire on zh — cross-language false positives)

Usage: uv run python scripts/en_formal_gate_probe.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# --- lexical EN formal gate (v0, mirrors the zh gate's design) ---
_MARKERS_EN = {
    # closing/opening formulae
    "REGARDING:": 3, "We are writing to": 2, "hereby": 2, "pursuant to": 3,
    "in accordance with": 2, "We hereby affirm": 3,
    # structural
    "Effective date": 2, "Issued by": 2, "To:": 2, "Date:": 1,
    "shall": 1, "undertake": 2, "Contact:": 1,
}
_STRUCTURE_EN: list[tuple[re.Pattern[str], int, int]] = [
    (re.compile(r"(?im)^REGARDING:.{3,60}$"), 3, 1),
    (re.compile(r"(?im)^To:.{3,60}$"), 2, 1),
    (re.compile(r"(?m)^Field:\s*\S"), 2, 1),  # Field: value blocks
    (re.compile(r"(?m)^\s*(I|II|III|IV)\.\s+[A-Z]"), 2, 2),  # roman clauses
    (re.compile(r"(?im)^(Issuer|Scope|Effective Date|Contact)\s*:"), 2, 2),
    (re.compile(r"(?i)(recall|sincerely|regards)[,.]?\s*$", re.M), 1, 1),
]
THRESHOLD_EN = 6


def detect_register_en(text: str) -> tuple[int, list[str]]:
    score = 0
    hits: list[str] = []
    for m, w in _MARKERS_EN.items():
        if m.lower() in text.lower():
            score += w
            hits.append(m)
    for pat, w, need in _STRUCTURE_EN:
        n = len(pat.findall(text))
        if n >= need:
            score += w
            hits.append(pat.pattern[:20])
    return score, hits


def main() -> int:
    import json

    print("=== 1. human EN formal probe (n=35): gate hit = abstain = catastrophe cut ===")
    files = sorted((ROOT / "dataset/legal_declaration_en/human").glob("*.md"))
    gated = 0
    flagged_total = hc_total = 0
    rows = json.loads((ROOT / "reports/human_probe_results_en_human.json").read_text(encoding="utf-8"))
    flag_map = {r["file"]: r for r in rows}
    miss = []
    for f in files:
        body = f.read_text(encoding="utf-8").split("---", 2)[2]
        score, _ = detect_register_en(body)
        hit = score >= THRESHOLD_EN
        gated += hit
        r = flag_map.get(f.name)
        if r and r["label"] == "AI-generated":
            flagged_total += 1
            hc_total += r["confidence"] > 0.8
            if not hit:
                miss.append((f.name, score, round(r["confidence"], 2)))
    print(f"gated: {gated}/{len(files)} ({gated/len(files):.0%}); "
          f"flagged humans: {flagged_total}, of which high-conf: {hc_total}")
    print(f"leak-through (flagged humans NOT gated): {miss or 'NONE'}")

    print("\n=== 2. W4-EN AI cells: abstain rate per arm (honest abstain where AUROC is 0.36-0.72) ===")
    w4en_records_path = ROOT / "dataset/paired_generation_v1/w4en_records.jsonl"
    recs = [
        json.loads(line)
        for line in w4en_records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    stats: dict[str, list[int]] = {}
    for r in recs:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        score, _ = detect_register_en(r["text"])
        key = f"{r['model'].split('/')[-1]}|{r['arm']}"
        d = stats.setdefault(key, [0, 0])
        d[1] += 1
        d[0] += score >= THRESHOLD_EN
    for k in sorted(stats):
        g, n = stats[k]
        print(f"  {k:<28} gated {g}/{n} ({g/n:.0%})")

    print("\n=== 3. zh sanity (gate must not fire cross-language) ===")
    zh_dir = ROOT / "dataset/legal_declaration_zh/human"
    zh_gated = 0
    zh_n = 0
    for f in sorted(zh_dir.glob("*.md"))[:20]:
        body = f.read_text(encoding="utf-8").split("---", 2)[2]
        s, _ = detect_register_en(body)
        zh_gated += s >= THRESHOLD_EN
        zh_n += 1
    print(f"zh docs gated by EN gate: {zh_gated}/{zh_n}")

    print("\n=== 4. risk-coverage verdict ===")
    # coverage on AI casual arms (the EN path's least-broken region)
    casual_total = sum(n for k, (g, n) in stats.items() if k.endswith("|C") or k.endswith("|D"))
    casual_gated = sum(g for k, (g, n) in stats.items() if k.endswith("|C") or k.endswith("|D"))
    print(f"casual AI coverage kept: {casual_total - casual_gated}/{casual_total}")
    gated_hc = hc_total - (flagged_total - len(miss)) if miss else 0
    print(f"human-formal high-conf errors after gate: {gated_hc} (of {hc_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
