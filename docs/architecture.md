# Architecture

Source-of-truth modules: `src/aigc_detector/detection/pipeline.py`
(cascade), `src/aigc_detector/detection/register.py` (register gates +
calibration/floor loaders), `src/aigc_detector/models/manager.py` (VRAM
lifecycle). This document describes the v0.1 system.

## Overview

A cascading, language-routed ensemble over four detection stages. Every
verdict is designed to be **auditable**: per-stage breakdown, segment
evidence, register caveats, and calibration provenance ride along in the
API response (see `docs/api.md`).

```
                          ┌────────────────────────────┐
                          │  POST /api/v1/detect(file) │
                          └─────────────┬──────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │  Language Router (XLM-RoBERTa)│  → zh | en
                        └─────────────┬─────────────────┘
                                        ▼
        ┌──────────────────────── Stage 1 (fast, parallel) ───────────────────────┐
        │  Statistical: LM-prob features        Linguistic: 14-feature stylometry │
        │   EN gpt2-xl / ZH Wenzhong-GPT2-110M   CPU-only; M5/M6 reuse Stage-1    │
        │   → classifier (joblib)                 token log-probs (no 2nd pass)   │
        └─────────────┬───────────────────────────────────────────────────────────┘
                      ▼  EN: stat confidence > 0.99 → early exit
        ┌───────────────────────────────┐
        │  Stage 2: Encoder (LoRA)      │  EN DeBERTa-v3-large / ZH Chinese-RoBERTa
        │  ZH arbitration: stat=human   │  + encoder p_ai ≥ 0.35 → encoder wins
        │  ZH: enc confidence > 0.99    │  → early exit (skip Binoculars)
        │  EN: stat & encoder agree     │  → combine, skip Binoculars
        └─────────────┬─────────────────┘
                      ▼  conflict / ZH always (HC3 blind-spot safety net)
        ┌───────────────────────────────┐
        │  Stage 3: Binoculars (4-bit)  │  EN falcon-7b pair / ZH Qwen2-7B pair
        │  zero-shot, low-fpr mode      │  score→p_ai sigmoid around threshold
        └─────────────┬─────────────────┘
                      ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │  Weighted ensemble (per-language weights, renormalized over stages  │
        │  that ran). ZH decision threshold 0.47 (vs 0.5 EN).                 │
        └─────────────┬────────────────────────────────────────────────────────┘
                      ▼
        ┌──────────────────────────── Register layer (post-hoc, fail-safe) ────┐
        │ 1. ZH lexical formal gate (score ≥ 6) → attach caveat (W3a/W3b)       │
        │ 2. EN formal gate (score ≥ 5)  → cap confidence at 0.49 + warning     │
        │    (W16 downgrade guard — measured 71% human-misclassification blind  │
        │    spot on EN formal register)                                        │
        │ 3. W15 floor OR-rule (formal_zh, cutoff 0.46, enabled): force-run     │
        │    Binoculars if early-exited; bino p_ai ≥ cutoff → verdict upgraded  │
        │    to AI-generated (decision_rule provenance attached)                │
        │ 4. W11 calibration (formal_zh, T=5.645): confidence ←                │
        │    sigmoid(logit(conf)/T); monotone ⇒ label-flip-free,               │
        │    ranking-preserving; T=1 elsewhere                                  │
        └─────────────┬─────────────────────────────────────────────────────────┘
                      ▼
          DetectionResponse (breakdown, segments, highlights,
          caveat, calibration, decision_rule)
```

## Detection Stages

| Stage | Models | Role |
|---|---|---|
| Statistical | GPT-2-XL (EN), Wenzhong-GPT2-110M (ZH) | Fast LM-probability features (perplexity/burstiness family) + joblib classifier. |
| Linguistic | CPU-only, no neural net | 14 features in three tiers: M1–M9 micro (sentence-length burstiness/CV/Gini, syntactic repetition, token-logprob skew, hedging, discourse templating), S1–S2 meso (paragraph variance), D1–D3 macro (MTLD lexical diversity…). |
| Encoder | DeBERTa-v3-large (EN), Chinese-RoBERTa-wwm-ext-large (ZH) + LoRA adapters | Main learned discriminator; ZH adapter oversampling-retrained 2026-06 against HC3 overfit. |
| Binoculars | falcon-7b + falcon-7b-instruct (EN), Qwen2-7B + Qwen2-7B-Instruct (ZH), 4-bit | Zero-shot perplexity-crossing fallback; the ZH path always runs it (HC3-era blind spot on modern LLM text). |

## Language Routing & Ensemble Weights

Weights are set per language and **renormalized over the stages that
actually ran** (early exits are first-class):

| Language | statistical | linguistic | encoder | binoculars | Rationale |
|---|---|---|---|---|---|
| zh | 0.10 | 0.10 | **0.60** | **0.20** | Retrained encoder is the strongest ZH signal; statistical kept as minor (HC3 overfit); Binoculars as contract-generation safety net. |
| en | 0.15 | **0.85** | 0.0 | 0.0 | EN encoder LoRA underperforms on modern multi-LLM text (2026-06-17 sweep, `DETECTOR_NOTES_2026-06.md`); linguistic axis is primary. |

ZH decisions use threshold 0.47 (`ZH_DECISION_THRESHOLD`); EN uses 0.5.

## Lazy Loading & VRAM Management

No detector weights load at import time — `ModelManager`
(`src/aigc_detector/models/manager.py`) loads each model on **first use**
and tracks estimated VRAM against an **11 GB budget inside the 12 GB
envelope** (RTX 3060 design target), evicting least-recently-used models
when a new load would exceed it. Binoculars 7B pairs (~14 GB/language on
disk) load in 4-bit; if not cached, a background thread downloads them
with resumable, sequential, retrying semantics while the service serves
degraded (no Binoculars stage).

## Register Gates, Calibration, and the Floor Rule

The register layer is deliberately **lexical, explainable, and
fail-safe** — any exception leaves the verdict untouched.

- **ZH formal gate** — weighted markers (特此声明/兹有/依据《…, threshold ≥6)
  plus structural patterns (《关于…的声明》 titles, ＿＿年＿＿月＿＿日
  signature lines, numbered clauses). On hit: `caveat` payload + entry into
  calibration and floor rules.
- **EN formal gate (W16)** — lexical markers (pursuant to, hereby,
  REGARDING:-style headers…) + structure regexes, threshold ≥5. On hit:
  confidence capped at 0.49 with a strong warning — the measured error on
  this register (71% of human formal docs flagged as AI, n=35) is worse
  than a coin flip, so no normal-confidence verdict is issued.
- **W11 calibration** — `models/calibration/global_temperature.json`
  (`applied: true`, T=5.645, fitted on a 382-doc class-balanced formal
  probe). Register-conditioned: formal_zh only, T=1 elsewhere.
- **W15 floor OR-rule** — `models/calibration/binoculars_floor.json`
  (`enabled: true`, cutoff 0.46). In the formal register, if the ensemble
  says Human but Binoculars p_ai ≥ cutoff, the verdict upgrades to
  AI-generated with `decision_rule` provenance. Evidence and gate review:
  `reports/w3b_floor_analysis.json`, `reports/w15_gate_review.md`.

## Artifact Layout

```
models/
  calibration/
    global_temperature.json   # W11: {T, applied, fit, safety, deployment}
    binoculars_floor.json     # W15: {enabled, cutoff, register, evidence}
    model_pins.json           # exact HF revisions for reproducible pulls
  statistical-{en,zh}/        # joblib classifier + calibration.json
  linguistic-{en,zh}/         # joblib classifier
  encoder-{en,zh}/            # LoRA adapters
```

Calibration artifacts ship with the repo and are resolved from either the
dev checkout or the installed-wheel layout (`register._calibration_dir`).
Related documents: `docs/capability-statement.md` (reliability
boundaries), `docs/adr/0001-w6-retrain-deferral.md` (why the ZH formal
encoder retraining was deferred), `docs/diagrams/` (draw.io sources).
