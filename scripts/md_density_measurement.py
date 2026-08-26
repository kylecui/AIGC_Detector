"""Measure markdown-marker density: AI arm A (free) vs arm B (contract-banned) vs human corpus.

Purpose: test the hypothesis "AI-written markdown has far higher bold ratio than
human" against our paired data. AI side is CLEAN (raw model output, zero
conversion). Human side is CONTAMINATED for this metric — our web-reader/pypdf
conversion introduced markdown artifacts (the very bold-wrapping normalized in
intake scoring) — reported but flagged non-comparable.

Usage: uv run python scripts/md_density_measurement.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parent.parent

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+\S")
BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+\S")


def densities(text: str) -> dict[str, float]:
    n = max(1, len(text))
    bold_chars = sum(len(m) for m in BOLD_RE.findall(text))
    return {
        "bold_per_kb": len(BOLD_RE.findall(text)) * 1000 / n,
        "boldchar_pct": bold_chars * 100 / n,
        "heading_per_kb": len(HEADING_RE.findall(text)) * 1000 / n,
        "bullet_per_kb": len(BULLET_RE.findall(text)) * 1000 / n,
    }


def agg(rows: list[dict]) -> dict:
    out = {}
    for k in ("bold_per_kb", "boldchar_pct", "heading_per_kb", "bullet_per_kb"):
        xs = [r[k] for r in rows]
        out[k] = (st.median(xs), sum(1 for x in xs if x > 0) / len(xs))
    return out


# --- AI side: raw outputs, clean ---
recs = [
    json.loads(line)
    for line in
    (ROOT / "dataset/paired_generation_v1/pilot_records.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
groups: dict[str, list[dict]] = {}
for r in recs:
    key = f"{r['model'].split('/')[-1]}|arm{r['arm']}"
    groups.setdefault(key, []).append(densities(r["text"]))

print("=== AI side (raw model output — CLEAN measurement) ===")
print(f"{'group':<28}{'bold/kB med':>12}{'%docs bold>0':>14}{'head/kB med':>13}{'bullet/kB med':>14}")
for key in sorted(groups):
    med, share = agg(groups[key])["bold_per_kb"], agg(groups[key])["boldchar_pct"]
    h = agg(groups[key])["heading_per_kb"]
    b = agg(groups[key])["bullet_per_kb"]
    print(f"{key:<28}{med[0]:>12.1f}{med[1]:>13.0%}{h[0]:>13.1f}{b[0]:>14.1f}")

# arm A vs B contrast per model (the controllability test)
print("\n=== controllability: armA vs armB bold_per_kb (median) ===")
for model in ("Qwen2.5-7B-Instruct", "GLM-4-9B-0414", "DeepSeek-V3.2"):
    a = agg([d for r, d in [(r, densities(r["text"])) for r in recs if r["model"].endswith(model) and r["arm"] == "A"]])
    b = agg([d for r, d in [(r, densities(r["text"])) for r in recs if r["model"].endswith(model) and r["arm"] == "B"]])
    print(f"  {model:<22} A={a['bold_per_kb'][0]:>6.1f}   B={b['bold_per_kb'][0]:>6.1f}   (contract bans markdown)")

# --- human side: conversion-contaminated, labeled ---
hum = []
for f in sorted((ROOT / "dataset/legal_declaration_zh/human").glob("*.md")):
    parts = f.read_text(encoding="utf-8").split("---", 2)
    body = parts[2].strip() if len(parts) == 3 else ""
    if len(body) > 200:
        hum.append(densities(body))
ha = agg(hum)
print(f"\n=== human side (n={len(hum)} — CONTAMINATED: web-reader bold-wrapping artifacts, NOT authorship signal) ===")
bold_med, bold_share = ha['bold_per_kb']
print(f"  bold/kB median={bold_med:.1f}  docs-with-bold={bold_share:.0%}  heading/kB={ha['heading_per_kb'][0]:.1f}")
