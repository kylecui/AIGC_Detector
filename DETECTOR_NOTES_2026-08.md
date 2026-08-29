# AIGC Detector Notes — 2026-08

## FN-1 — False negative: AI-drafted formal legal/regulatory Chinese (2026-08-17)

### Case

`docs/software-copyright/软件合法合规及原创性声明.md` — a ~2,000-char Chinese
compliance declaration (软件著作权登记补正材料, formal legal/regulatory register,
numbered clauses + one info table + signature block), **drafted end-to-end by an
LLM** (GLM, as the project agent) and lightly human-edited. Ground truth: AI-generated.

Full four-stage pipeline (first ZH request to run all four stages since the
Route-B encoder retraining was validated):

| Stage | p_ai | Label |
|---|---|---|
| statistical (Wenzhong-110M) | 0.0108 | human (98.9%) ❌ |
| linguistic (14-feat XGB) | 0.2157 | human ❌ |
| encoder (chinese-roberta+LoRA v2, textbook-oversampled) | 0.0297 | human (97.0%) ❌ |
| binoculars (Qwen2-7B pair, 4-bit) | 0.3431 | human ❌ |
| **ensemble** `{stat:.10, ling:.10, enc:.60, bino:.20}` | **0.1091** | **Human-written, conf 0.891 ❌** |

ZH decision threshold 0.47 — not close. All four axes failed independently;
this is not a weighting artifact.

Linguistic diagnostics: `micro=4.5, meso=10.0, macro=3.0,
human_likeness=57/100`. Top signals: sentence_length_cv, lexical_diversity_mtld,
paragraph_length_variance. Note **meso=10.0 (ceiling)** — the table + numbered
clause structure reads as high paragraph-length variance / template diversity,
which the meso layer scores as strongly human-like.

### Segment view (include_segments=true, 8 segs)

Most segments sit at p_ai 0.01–0.21, but one clause flags hard:

| Segment | Content | p_ai |
|---|---|---|
| seg 0 | title + 称谓 + 总起段 | 0.0378 |
| seg 1 | 基本信息表（表格行） | 0.0735 |
| seg 2 | 第二条第三款引用段 | 0.2048 |
| seg 3 | 原创性声明段 | 0.0123 |
| seg 4 | 第四节开头 | 0.1125 |
| seg 5 | 训练数据构建段 | 0.0131 |
| seg 6 | 治理支撑定位段 | 0.1054 |
| **seg 7** | **第五条第4款（"检测结果的展示与使用遵守相关法律法规，不用于非法目的"式排比收尾）** | **0.8560 ✅** |

The single 0.8560 segment is diluted to invisibility by the 0.109 weighted mean.
A "max-segment" or "top-k segment" aggregator would have caught this document.

**Correction (2026-08-17 later)**: an earlier note claimed a `segment_index`
serialization bug (segments returning `seg#None`). False alarm — the API
returns `index` correctly (verified live: `['breakdown','char_end',
'char_start','confidence','detected_language','index','p_ai',
'predicted_label','processing_time_ms','stages_used','text']`, values 0..n);
the `None` came from the ad-hoc test script reading a wrong key name
(`segment_index`). No API bug. A small regression test asserting the `index`
key is still worth adding.

### Sampling caveat (honesty note)

First run accidentally included the trailing HTML usage-note comment block
(strip regex missed the blank line between `---` and `<!--`); clean re-run
(2029 chars, comment excluded) reproduced the FN with near-identical numbers
(p_ai 0.1081 → 0.1091). The FN stands on the clean sample.

### Why every axis failed — hypotheses

1. **Register, not provenance**: formal 公文体/regulatory Chinese is a
   low-entropy register *by convention*. Templates ("依据…现…声明如下",
   "特此声明"), mandated clause structure, and boilerplate legal phrases are
   high-perplexity-safe for a human-trained prior AND low-surprise for a small
   reference LM. Wenzhong-110M (2021-era, 110M) has likely never seen modern
   LLM legal drafting; it scores the text like any other formal document.
2. **Encoder v2's oversampling may have widened, not narrowed, this blind spot**:
   Route B (2026-06-19) oversampled GPT-4/Claude *textbook* content 10×.
   Textbook prose ≠ legal declaration prose. HC3 human answers include formal
   Q&A styles; the encoder may have learned "formal register → human" as a
   shortcut feature because in HC3 the formal-register human side (知乎/百科
   answers) outweighs formal AI text (which was casual ChatGPT chat).
3. **Linguistic meso ceiling (10.0) is a structural blind spot**: legal docs
   have heterogeneous paragraph shapes (table row, one-line clause, long
   paragraph) → high paragraph-length variance → scores "human". The meso
   layer's intuition (variance=human) inverts for structured documents.
4. **Binoculars 0.34 is the best of the four but still sub-threshold**: Qwen2-7B
   base/instruct cross-PPL ratio is closest to flagging (consistent with P3's
   2026-06 finding that Binoculars was the only stage with signal on modern LLM
   ZH text), but zero-shot calibration on formal register is off.

### What this suggests (candidate directions, not scheduled)

- **P4 candidate — segment-level aggregation policy**: max/top-k segment p_ai as
  a document-level signal or a "局部AI痕迹" secondary output. Would have
  caught this case (0.856 max vs 0.109 mean). Risk: FN→FP shift on long human
  docs with one heavy-template paragraph; needs threshold work on
  segment-count-normalized stats.
- **P4 candidate — register conditioning**: detect formal/legal register
  (公文体 classifier or lexicon) and switch ZH weights to binoculars-dominant
  (mirrors P3 Route A finding that Binoculars carries modern-LLM signal).
- **P4 candidate — legal/regulatory AI-corpus pack**: the FN text and similar
  公文 (声明/承诺书/情况说明) are cheap to generate at scale; a small
  oversampling pack (Route-B style) targeted at formal register would directly
  address the encoder shortcut.
- **Evaluation gap**: none of the ZH eval suites (HC3 hold-out, textbook
  hold-out, formal subsets) cover legal/regulatory register. Add a
  `legal_declaration_zh` probe set before any of the above, so we can measure
  rather than guess.

### Research addendum (2026-08-17, literature-verified)

External research confirms the FN is **structural**, not a tuning artifact, and
upgrades the hypotheses above to evidence-backed mechanism:

**Mechanism: the Perplexity Trap (register-constraint collapse).** Banerjee
2026 (arXiv:2607.13044) formalizes three conditions: (C1) external syntactic
constraint, (C2) applies to both human and AI output, (C3) restricts variation
on a linguistic axis. When all hold, human and AI distributions collapse onto
the same low-entropy manifold. On EPO patent claims: Binoculars 78.3% FPR,
DetectGPT 80.5% FPR — the mirror symptom of our FN (same collapse, opposite
side of the threshold). The paper argues this is detector-internal and cannot
be fixed by curvature correction / domain adaptation (adaptation can WORSEN
overlap — DivScore, ACL 2025: adapted overlap 0.585→0.780). Our document
satisfies C1-C3 (mandated 公文体 syntax + template + table).

**Encoder: 1D formality collapse.** Smirnov 2026 (Zenodo 19399532): every
tested fine-tuned detector collapses its representation to ~1 dimension
aligned with **formality** (cos 0.73-0.99), not authorship. Our encoder v2's
HC3 training (human side: formal zhihu/baike; AI side: casual ChatGPT chat —
confirmed no legal register either side, `crawler.py:172-182`) plausibly
taught exactly this shortcut: formal register → human. Countermeasure with
evidence: adversarial formality training via gradient reversal (cos 0.98→0.45).

**Aggregation: max/top-k has precedent.** WaterSeeker (arXiv:2409.05112)
"first locate, then detect"; GigaCheck (ACL 2026) DETR-style span localization;
Simpson's-paradox analysis of likelihood aggregation (arXiv:2605.06294) —
naive document-mean aggregation destroys local signal across heterogeneous
regions. Our seg-0.856-diluted-by-mean is a textbook instance.

**Thresholds: group-adaptive, not global.** FairOPT (arXiv:2502.04528):
per-subgroup thresholds (formality/length) cut cross-group discrepancy 27.4%
at <0.1% accuracy cost. RAID (ACL 2024): calibrate on in-domain data, report
at fixed FPR; never naive 0.5 thresholds cross-domain.

**Information-theoretic bound (strategic context).** Sadasivan et al.
(arXiv:2303.11156): AUROC ≤ 1/2 + TV(M,H) − TV²/2. Contracted formal genres
shrink TV by construction → zero-shot statistical detection is bounded
regardless of architecture. Chakraborty et al. (arXiv:2304.04736): multi-sample
pooling (same author/corpus) restores separability via Chernoff information —
argues for corpus-level, not document-level, verdicts in collapsed registers.
Implication: in strongly-contracted registers the endgame is provenance
evidence (watermark/trace/process logs), not better likelihood statistics.

**Prioritized P4 plan (tiers):**

| Tier | Action | Code anchor | Evidence |
|---|---|---|---|
| 1a | Fix `segment_index` None bug | `routes.py:93` | blocker for all segment features |
| 1b | Secondary "局部AI痕迹" output: max/top-k segment p_ai + count | `routes.py:103-124` response, `ensemble.py:150` for doc policy | WaterSeeker; arXiv:2605.06294 |
| 1c | Register gate: 公文体 lexicon (seed: `linguistic.py:528` TEMPLATE_OPENERS_ZH) → binoculars-dominant weights + widened CI + UI caveat; explicitly NOT naive threshold lowering | `pipeline.py:110-113` weights, `pipeline.py:42` threshold | FairOPT; DivScore adaptation-paradox warning; P3 Route A precedent |
| 2a | `legal_declaration_zh` probe set (~50 AI 公文 + 50 human 公文) BEFORE any retraining | new eval asset | RAID calibrate-first |
| 2b | Route-B oversampling pack for formal register, human-side formal text included (within-register balance mandatory) | `train_encoder.py` | adaptation paradox |
| 3a | Adversarial formality training on next encoder retrain | trainer | 1D-collapse paper |
| 3b | Corpus-level verdicts + provenance-roadmap decision | product | Sadasivan/Chakraborty |

**Proposed controlled experiment (pairs the FN with the harness thesis):**
paired-generation design — same semantic content, arm A free-form prompt vs
arm B contract/spec-constrained generation (N=40-100 pairs, zh formal register),
measure all four stages. Hypothesis: contract arm shows systematically lower
p_ai (statistical evasion via constraint). Direct evidence for arXiv:2607.13044
C1-C3 in Chinese legal register, and an independent measurement axis for
contract-completeness research (detectability distance as proxy for
uncontracted model slack). Cheap: generation-only + existing API.

### Runtime observations (context)

- Request triggered lazy-load of binoculars-zh (Qwen2-7B pair, 4-bit) on a
  12GB GPU already ~4.4GB used by desktop apps; **first request 503'd during
  load** (middleware returned 503 while weights streamed in), second request
  succeeded. 73.6–85.4s end-to-end latency with Binoculars on this GPU.
- `Model binoculars-zh not in registry, loading without VRAM tracking` warning
  observed during lazy-load — binoculars models absent from
  `models/registry` VRAM tracking; worth a look independent of this FN.
- sklearn `UserWarning: Skipping features without any observed values: [4 5 9]`
  (imputer) fires on every statistical-stage call for this text shape — known
  L1 behavior (features 4/5/9 often NaN), noise only.

### Repro

```bash
# service must be up (uvicorn src.aigc_detector.api.main:app)
uv run python tmp/quicktest_clean.py
```

Ground-truth file lives in-repo (`docs/software-copyright/软件合法合规及原创性声明.md`,
strip the trailing `<!-- ... -->` note block before testing).

---

## W4 — Paired-generation experiment: FULL RUN (2026-08-17, n=40 topics × 3 models × 2 arms)

**Design**: same topic, arms {A free-form, B output-contract spec v1.1
(≥800-char floor, structure/mandated-phrase/table/length constraints)}.
Capability ladder validated in kylecui/contract-driven-harness-study:
Qwen2.5-7B-Instruct (local 4-bit, below-floor candidate 0/40),
GLM-4-9B-0414 (SiliconFlow, above floor 30/40, FN-1 family),
DeepSeek-V3.2 (SiliconFlow, above floor 40/40). temp 0.7 / top_p 0.9.
240 records, all evaluated in-process (same weights as service). Artifacts:
`dataset/paired_generation_v1/` (pilot_records.jsonl, eval_results.jsonl,
summary.json). Scripts: `scripts/paired_generation_experiment.py`,
`scripts/evaluate_paired_experiment.py`.

### Results (paired A−B; positive = contract arm LESS detectable)

| model | floor | ensemble A | ensemble B | Δ | sign p | d_z | pos/neg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | below | 0.735 | 0.743 | −0.008 | 1.0 | ≈0 | 20+/19− |
| GLM-4-9B | above | 0.829 | **0.342** | **+0.487** | 2e-08 | +1.31 | 37+/3− |
| DeepSeek-V3.2 | top | 0.334 | **0.103** | **+0.230** | 7e-04 | +0.71 | 31+/9− |

**Cleanest finding — contract-arm detectability is monotone in capability**:
under the SAME contract, B-arm ensemble p_ai falls 0.74 → 0.34 → 0.10 as
contract-adherence capability rises (7B → GLM → DeepSeek). The below-floor
model gains nothing from the contract (cannot execute it into the human
manifold); above-floor models dive deep into "human" territory.

### Stage decomposition

- **statistical**: GLM 0.655→0.068 (d_z=+2.44, the largest single-stage
  collapse — perplexity compression is the core mechanism); DeepSeek already
  at floor in BOTH arms (0.06/0.08) — its formal register is low-perplexity
  even free-form; 7B low both arms (Wenzhong sees all formal zh as low-PPL).
- **encoder**: the workhorse signal. GLM 0.858→0.281 (sign p=1.5e-09);
  DeepSeek 0.275→0.048 (near-total evasion, deep inside human territory);
  7B flat 0.74/0.74 — contract changes nothing it can exploit.
- **linguistic**: insensitive everywhere (~0.10-0.18, all n.s., 8+/9−).
  Register features do not differ between arms — formal is formal; the 14
  stylometric features carry no contract signal in this genre (consistent
  with FN-1 where meso hit its ceiling).
- **binoculars**: partial coverage (encoder early-exit at conf>0.99 skips it
  on confident records — n=4-9 pairs only). GLM 0.700→0.571 (4+/0−, p=0.125
  n.s.); DeepSeek 0.626→0.607 n.s. Weakest contract response of the four —
  consistent with its FN-1 role (best-of-four but sub-threshold).

### Interpretation

1. **H1 confirmed conditionally (capability gate)**: contract-driven
   statistical evasion requires above-floor contract adherence. Effect size
   for above-floor models is large (d_z 0.7-1.3, sign p ≤ 7e-04).
2. **FN-1 fully reproduced as a mechanism**: GLM under a document spec lands
   at ensemble 0.342 mean (FN-1 document itself: 0.109 — even deeper, because
   the CPCC spec was stricter + human-edited + longer). The Perplexity Trap
   (C1-C3) is now experimentally instantiated in Chinese 公文体.
3. **Linguistic axis is structurally blind in formal register** — confirmed
   both in FN-1 (meso ceiling) and across 240 samples (n.s. everywhere).
   Any register-gating (W3) should not rely on the linguistic stage.
4. **Dual-use measurement axis for contract research**: detectability
   distance (A−B) quantifies how much model slack a contract removes, per
   model — "residual uncontracted entropy" is measurable and monotone in
   contract-adherence capability.

### Caveats

- B-arm length still shorter on average for GLM (contract stops it early;
  adherence imperfection is itself capability data). Length floor raised in
  v1.1 but not perfectly obeyed. DeepSeek texts run long (A up to 1700c).
- Early-exit architecture means ensemble numbers mix stage compositions
  (some records encoder-only); per-stage table above is the honest view.
- Single run per topic/arm (no repeats); temperature 0.7. Statistical power
  comes from pairing, n=40 topics per model.
- Detection thresholds trained pre-experiment; no tuning on these data.

### Assets produced (feeds W5)

240 AI-side 公文体 records (3 models × 2 arms × 40 topics) with full
detection breakdowns — the AI half of the `legal_declaration_zh` probe set.
Human side (matched register, ≥40 docs) still to collect.

---

## W4b — Replication (2026-08-17, t41-t60, n=20 topics × 2 API models)

**Design**: exact replication of W4 on **20 NEW topics** (t41–t60,
`scripts/topics_replication.json` — same 公文体 doc-type mix, different
subjects and invented org names) with **fresh server-side sampling**. Arms,
prompts, GEN_PARAMS, and evaluation pipeline unchanged; new records tagged
`spec_version: "v1.2-replication"`. API models only (GLM-4-9B-0414,
DeepSeek-V3.2); the below-floor local 7B was not re-run (its null result is
a capability statement, not a sampling statement). 80 new records appended
to `pilot_records.jsonl` (total 320), all evaluated in-process; **0 short
outputs (<200 chars)**. Analysis on replication topics only via
`scripts/analyze_w4b.py` (reuses `paired_stats`); machine-readable results in
`dataset/paired_generation_v1/summary_w4b.json`.

### Results (paired A−B; positive = contract arm LESS detectable)

| model | ensemble A | ensemble B | Δ | wilcoxon p | sign p | d_z | pos/neg |
|---|---|---|---|---|---|---|---|
| GLM-4-9B | 0.888 | **0.314** | **+0.574** | 1.3e-05 | 4.0e-05 | +1.57 | 19+/1− |
| DeepSeek-V3.2 | 0.257 | **0.075** | **+0.182** | 1.0e-03 | 4.1e-02 | +0.75 | 15+/5− |

### Comparison vs first run (W4, n=40)

| model | Δ first run | Δ replication | sign p first | sign p repl. | d_z first | d_z repl. | verdict |
|---|---|---|---|---|---|---|---|
| GLM-4-9B | +0.487 | +0.574 | 2e-08 | 4.0e-05 | +1.31 | +1.57 | **replicated** (same direction, magnitude comparable-to-larger, significant) |
| DeepSeek-V3.2 | +0.230 | +0.182 | 7e-04 | 4.1e-02 | +0.71 | +0.75 | **replicated** (same direction, magnitude comparable, significant) |

### Stage decomposition (replication)

- **encoder**: again the workhorse. GLM 0.919→0.252 (Δ+0.668, sign p=4.0e-05,
  d_z=+1.71); DeepSeek 0.201→0.031 (Δ+0.170, sign p=4.0e-04, d_z=+0.67).
  Both arms' encoder behavior matches the first run almost exactly.
- **statistical**: GLM A-arm 0.899 vs B-arm 0.069 but only **n=3 pairs** —
  the A arm early-exits at encoder conf>0.99 on most replication records, so
  the statistical stage never runs (sign p=0.25 at n=3 is uninformative, not
  a contradiction of W4's Δ+0.62 at n=20). DeepSeek at floor both arms
  (0.027/0.061, n.s.), replicating W4.
- **linguistic**: n.s. everywhere again (GLM n=3; DeepSeek −0.014, p=0.56).
  Structural blindness in formal register confirmed a second time.
- **binoculars**: GLM **0 pairs** this run (A-arm encoder early-exit skipped
  it everywhere); DeepSeek +0.018 n.s. (n=4). Weakest contract response,
  consistent with W4.

### Verdict

**The capability-gate effect REPLICATES.** On unseen topics with fresh
samples, both above-floor models show the contract arm significantly less
detectable than the free-form arm, with effect sizes equal to or larger than
the first run (GLM d_z 1.31→1.57; DeepSeek d_z 0.71→0.75). The W4
conclusions — and the W6/W7 priorities built on them — stand.

### Caveats

- **Fresh sampling ≠ literal seed change**: API sampling is server-side
  stochastic; "replication" here means new topics + new draws, not a
  controlled seed sweep.
- **Power**: n=20 pairs vs 40 in W4. DeepSeek's sign test (p=0.041) is much
  weaker than its Wilcoxon (p=0.001) because the sign test discards
  magnitudes; both are significant at 0.05 but the margin is thinner.
- **Stage coverage asymmetry**: GLM's A arm is so confidently flagged by the
  encoder (mean 0.92, 12/20 records >0.99 early-exit) that downstream stages
  rarely run — replication stage-pair counts (statistical n=3, binoculars
  n=0) are too small for inference; rely on W4 for those stages.
- Single draw per topic/arm, temp 0.7, same as W4. Detection thresholds
  unchanged, no tuning on these data.

---

## W5 trial batch — human formal docs collected + evaluated (2026-08-17, n=10)

**Collection**: 10 real human 公文体 docs, verbatim-fetched (7 HTML live + 3
PDF-extracted from cninfo/SZSE disclosures, source PDFs kept as provenance),
each with source URL / publisher / date / license note. Types: gov regulatory
announcement (samr), product recall (GAC Toyota), listed-company
correction/apology/clarification announcements, bank maintenance notices
(CCB/CITIC), university situational reports (SHU/WHU), bar-association
statement. Stored: `dataset/legal_declaration_zh/human/`. Collection pipeline
VERIFIED — scale-up to 60-80 cleared to proceed.

**Trial detection results** (`scripts/eval_human_probe.py`, same in-process
pipeline; results in `reports/human_probe_trial_results.json`):

- 8/10 correctly Human-written. **2 false positives**:
  - `06-bank-ccb-system-maintenance.md` — flagged AI at **p_ai=0.9868**
    (571-char highly-templated bank notice)
  - `03-company-haimo-correction.md` — flagged AI at p_ai=0.7432
    (listed-company correction announcement w/ table data)
- Trial "FPR" preview 20% (n=10, descriptive only — NOT the calibrated
  baseline; W5 full set with Wilson intervals pending).

**Significance**: the FPR-side mirror symptom, empirically confirmed. FN-1
showed AI formal text sailing through as Human (0.89 conf); the trial shows
human formal text flagged as AI at 0.99 confidence — **both directions of
failure in the same collapsed register, exactly as the literature predicts**
(EPO study FPR>60% on patent claims; our ZH analogue). The CCB case is a
one-document justification for W9/D1 (calibration honesty: high-confidence
errors are the headline metric) and cautions W3b weight-switching: moving ZH
weights toward binoculars/encoder to chase FN recall could worsen exactly
these FPs. W5 full baseline must report both directions with intervals.

### Pilot record (n=10 topics, Qwen2.5-7B only — superseded by full run above)

First pass with only the local 7B: ensemble A 0.733 vs B 0.817 (Wilcoxon
p=0.36, 20+/19− at n=40 in full run) — H1 not supported on a below-floor
model, direction mildly reversed. Two confounds identified and addressed in
v1.1: arm-B shorter texts (455–722 vs 780–1060 chars) → length floor ≥800
added; mixed stage composition from early-exits → per-stage reporting made
primary. The pilot's honest negative result led to the capability-gate
hypothesis, which the full run then confirmed quantitatively (B-arm p_ai
monotone in ladder position: 0.74 → 0.34 → 0.10).

---

## W7 — Adversarial formality training: pipeline success, first candidate REJECTED (2026-08-18)

**Run**: 16026 steps / 3 epochs / 2h51m, train_loss 0.1392, batch 12 (headroom
after a silent 31% death on attempt 1 at batch 16 — fixed by 500-step
checkpointing + auto-resume; attempt 2 ran clean past the old death point).

**Gate verdict: FAIL** (`reports/adversarial_gate_2026-08-18.json`):
- G1 probe FPR **82.3%** (51/62) vs baseline 19.4% [11.4,30.9] — catastrophic
  regression, OPPOSITE of intended direction
- G2 FN-1 max-seg 0.8057 PASS (doc still Human 0.97 — more confidently wrong)
- G3 formality coupling Spearman 0.250
- Candidate auto-rejected; production adapter untouched (key-structure diff
  vs production adapter = 0 tensors — classifier head persisted fine, the
  "MISSING" load warning is benign; 82.3% is real, not mechanical)

**Mechanism (negative-result analysis)**: the zh training distribution's AI
class contains Route-B-oversampled FORMAL textbook content, so the lexical
formality target is positively correlated with the ai label. GRL removes a
label-correlated feature; the classifier reroutes through remaining spurious
correlates and lands HARDER on formal→AI. Same family as the DivScore
adaptation paradox (intervening on a confounded feature without balancing
data worsens overlap). Empirically reinforces ADR-0001: register failure is
data-distributional; balanced 公文 corpus (both sides) is the unlock, not
fine-tuning tricks.

**Next-iteration options (not scheduled)**: BETA 0.5→0.1, λmax 1.0→0.3, soft
ramp — but honest expectation: without register-balanced training data the
theoretical grounding is weak. W7 stays DEFER pending ADR-0001 review
conditions. Assets retained: adversarial.py, train/gate scripts (reusable
one-command pipeline), candidate dir + gate report as ADR trail.

---

## W4c — The decisive 2x2: register x prompt x 5 models x 5 seeds (2026-08-18, n=2000)

**Design**: 20 casual topics (social-media style) x {C free, D contract-with-
surface-features (emoji/hashtag/colloquial spec)} + 20 formal topics (t41-t60)
x {A free, B contract} x 5 models x 5 seeds = 2000 docs, all evaluated
in-process. Models: Qwen2.5-7B (local, below-floor), Qwen3-8B, GLM-4-9B,
Qwen3-14B, DeepSeek-V3.2. Analysis: `scripts/w4c_analyze.py` (topic-block
bootstrap, per-topic paired Wilcoxon). Analysis-script bugs fixed post-hoc:
stale-variable key collapse in H2 verdict, bare-vs-full model-id mismatch
(7B nan), degenerate permutation CI -> bootstrap.

### Full cell table (mean p_ai [boot CI] / miss rate)

| model | formal-A | formal-B | casual-C | casual-D |
|---|---|---|---|---|
| Qwen2.5-7B | 0.831 / 15% | 0.704 / 29% | 0.458 / 60% | 0.094 / 99% |
| Qwen3-8B | 0.541 / 49% | 0.182 / 93% | 0.085 / 95% | 0.036 / 100% |
| GLM-4-9B | 0.854 / 9% | 0.340 / 76% | 0.183 / 87% | 0.056 / 100% |
| Qwen3-14B | 0.619 / 36% | 0.329 / 77% | 0.103 / 98% | 0.036 / 100% |
| DeepSeek-V3.2 | 0.291 / 77% | 0.078 / 97% | 0.036 / 100% | 0.022 / 100% |

### Hypothesis verdicts (pre-registered in w4c_analyze.py)

- **H1 replication (formal A>B)**: ALL 5 models significant (p<=7e-03), incl.
  7B (+0.127, but stays detected at 0.704). W4/W4b direction confirmed; the
  7B now shows a small significant formal effect W4 missed (new topics+seeds).
- **H2 decisive cell (casual C vs D)**: contract suppresses p_ai in 7B
  (+0.364, p=2e-06 — LARGEST casual effect, from the WEAKEST model) and GLM
  (+0.127, p=0.001); n.s./tiny for Qwen3-8B/DeepSeek. BUT casual cells were
  already 60-100% missed FREE-FORM — contract is mostly ceiling-bound there.
- **H3 dose-response (formal-B across ladder)**: NOT monotone (7B 0.704 >
  8B 0.182 < GLM 0.340 ~ 14B 0.329 > DS 0.078; within-Qwen3 inverted).
  The 3-point W4 "monotone ladder" was family luck; model family/style
  dominates parameter count.

### The honest big picture (bigger than the hypotheses)

1. **The detector's competence region is ONE CELL: formal x crude/traceable
   model** (GLM 9%, 7B 15% miss on formal-free). Everything else is majority-
   missed: casual register 60-100% missed FREE-FORM (the formality shortcut
   mirrors — casual style reads as human); strong fluent models (DeepSeek)
   missed 77% even on formal-free; contracts collapse the remaining formal
   coverage (GLM 9%->76%, Qwen3-8B 49%->93%, 14B 36%->77%).
2. **Casual blind spot is NEW knowledge** — W4's formal framing understated
   the detector's weakness. The encoder shortcut is bidirectional: formal-
   register prior betrayed us on FN-1 (AI formal -> human), casual-register
   prior betrays us here (AI casual -> human, 60-100%).
3. **Capability gate refined, partially refuted**: on formal register the 7B
   cannot execute contracts into evasion (B stays 0.704 detected) — gate
   survives in the FAILURE sense; on casual register the gate INVERTS (the
   7B shows the largest contract evasion: crude casual text has the most
   slack for surface humanization; fluent models are already at the miss
   ceiling). Unifying frame: contracts move output toward the human manifold
   wherever slack remains; register x model capability determines where the
   slack is. ("Specification-slack" theory.)
4. **Paper framing consequence (W13)**: the duality thesis STRENGTHENS
   (contract-driven movement into human territory generalizes across
   registers, with mechanism varying: entropy collapse on formal, surface-
   feature humanization on casual), but the "capability gate" headline must
   be demoted to a register-conditional mechanism. The detector-as-instrument
   narrative gains a cleaner villain: bidirectional register shortcut.

### Caveats

- GLM casual-D: 18/22 short outputs (<120c, spec asked 280-330c) — GLM
  violated the casual length constraint; its casual-D estimate is confounded
  by non-compliance (noted, not excluded).
- Detector threshold 0.47 fixed pre-experiment; miss rates are at deployed
  operating point, not AUROC.
- 7B local sampling seeded (11-55); API models server-side stochastic —
  seed labels identify replicates, not exact-reproducibility.

Artifacts: dataset/paired_generation_v1/w4c_{records,eval_results}.jsonl,
w4c_analysis.json, summary_w4c_run.json; matched-era human slice collected
(20 docs 2024-2026, staging -> intake pending).

---

## Era-stratified FPR + W14 corpus-level time-box (2026-08-18 late)

### Era stratification (defends the top-1 reviewer attack)

Matched-era human slice (n=20, 2024-2026, era-authority verified) evaluated:
**FPR 3/20 = 15.0% [5.2%, 36.0%]** vs main store (82% pre-2023)
**19.4% [11.4%, 30.9%]** — intervals massively overlap; all 3 FPs are
LOW-confidence (0.50-0.65), ZERO high-confidence (main store has 8).
**Verdict: the formal-register blind spot is era-invariant.** Register
explanation beats era explanation; the pre-2023 anchoring critique is
answered with data (`reports/human_probe_results_matched_era.json`).

### W14 corpus-level pooling: NEGATIVE, theoretically sharp

k-sample pooling (k=1..20, B=2000 topic-stratified bootstrap; both mean-p
and mean-logit pooling; `scripts/w14_corpus_level.py`,
`reports/w14_corpus_level.json`):

| cell | AUROC k=1 -> k=20 (p-pool) | TPR@FPR5% best | verdict |
|---|---|---|---|
| GLM formal-free (A) | 0.924 -> 1.000 | 1.00 @ k=5 | RECOVERED |
| GLM formal-contract (B) | 0.691 -> 0.899 | 0.39 | NOT |
| Qwen3-14B (B) | 0.704 -> 0.869 | 0.27 (logit 0.18) | NOT |
| Qwen3-8B (B) | 0.512 -> 0.223 | 0.01 | NOT (inverts) |
| DeepSeek (B) | **0.297 -> 0.002** | 0.00 | **INVERTED** |
| DeepSeek (A) | 0.573 -> 0.729 | 0.12 | NOT |

**Findings**:
1. Pooling amplifies existing separation (GLM-A perfect at k>=5 — Chakraborty
   works where signal exists) but cannot create it in evasive cells.
2. **DeepSeek formal text scores MORE human than humans** (stochastic
   dominance); pooling drives AUROC to 0.002 — corpus verdicts become
   confidently WRONG. The Chernoff argument assumes distinguishability; in
   the contracted register the ordering flips and evidence-summing makes
   things worse.
3. **Capability statement corrected**: "corpus-level detection" removed as a
   recommendation for formal register — process evidence (留痕) is the only
   reliable route there. We corrected our own guidance the same day the
   falsifying data landed (defensibility charter in action).

Both results slot into paper section 4.4 (era robustness + corpus-level
negative) — the W13 skeleton's pre-writing checklist is now complete.

---

## W14 day-2: batch-screening mode + binoculars hint (2026-08-19)

### Part A — screening mode (max-pooling): no free lunch, and a sharper fact

Operational screening (flag a k-doc batch if ANY member exceeds the human
p95 threshold = 0.9866): evasive cells yield **0-2 catchable docs per 100**
(DeepSeek-B max p_ai 0.584; GLM-B max 0.9858 — misses the 0.9866 threshold
by a hair; Qwen3-8B/14B: 1-2 docs over). Batch TPR stays 0.0 (DeepSeek/GLM)
or operationally useless (14B: T 0.41/F 0.72 at k=20). Screening only works
where single-doc already works (GLM formal-free: T 0.95/F 0.27 at k=5).

**Threshold circularity insight**: the 0.9866 threshold is not arbitrary —
it is set by the human pool's OWN 8 high-confidence FPs (CCB notice 0.987
et al.). The screening failure and the formal-register FPR problem are two
faces of one coin: **no threshold separates evasive-AI from formal-human**,
because the human pool's high scorers occupy exactly the region where AI
catchable docs would need to live. Artifacts: scripts/w14_screening_mode.py,
reports/w14_screening_mode.json.

### Part B — binoculars-stage hint (UNVERIFIED, cross-side confound)

Descriptive (invalid cross-side) pooling suggested binoculars stage scores
of AI evasive cells separate strongly at k=10 (AUROC ~1.0 vs human ENSEMBLE
scores). Verified distributions: AI formal-contract binoculars p50 0.62-0.68
(n=30-59, sparse via early-exit). FN-1's own binoculars was 0.343 (AI). The
decisive missing data: **human formal docs' binoculars distribution** — if
low (<~0.5), a raw-binoculars corpus signal survives the ensemble's
dilution (rescue route); if high (EPO-style FPR), the hint dies. GPU queued
behind W4-EN; extraction pass on 82 human docs will settle it.

---

## Incident log: system restart & recovery (2026-08-20)

System restarted at ~01:44 (last W4-EN write). Integrity sweep post-restart:
- ALL pre-restart artifacts intact: 12/12 key docs, zh data complete
  (pilot 320 + W4c 2000/2000 + human 62 + matched-era 20), W14 reports,
  calibration artifact, fixtures. JSONL line-parse: 0 corrupt lines anywhere.
- Lost: the W4-EN experiment agent (66/2000 generated, 0 evaluated at kill);
  confirmed dead by 10h-stale mtime + process-table inspection (surviving
  python processes were MCP infrastructure only, not experiment workers).
- Survived via disk: EN human formal probe (bg task completed pre-restart):
  35 docs, 89% pre-2023, 8 doc types, headers complete, zero issues ->
  ingested to dataset/legal_declaration_en/human/ (single-issuer note: CPSC
  x5 is genre-inherent for recall notices).
- Environment: pytest 291 green; GPU at desktop baseline.
- Recovery: W4-EN resumed via fresh checkpoint-resume agent (done_keys
  dedup makes duplicate generation impossible; append-only JSONLs).
Lesson (already our default, reaffirmed): checkpoint-resumable append-only
artifacts + disk-as-truth made a mid-experiment OS restart a ~zero-data-loss
event; the only unrecoverable cost was agent wall-time.

## Incident log addendum: agent model quota exhaustion (2026-08-20 afternoon)

The resume agent (bg_c4d5b245, Sisyphus-Junior/kimi-k3) hit its underlying
model quota: throttled to ~3.4 min/record (66->115 in 2h46m, ~1/10 speed),
then cancelled per operator instruction. Handoff to a general-type agent
(bg_9ec6979e, different underlying model) with identical contract +
orphan-process check. Race audit before handoff: no orphan w4en processes;
records frozen at 115 since 14:48; evaluation-side id-dedup (sha1 of
topic|arm|model|seed) makes double-appends harmless even if they occur.
Checkpoint design again proved its worth: two agent deaths = zero data loss,
zero duplicated cells, zero re-planning.

---

## W4-EN human-side result: the EN mirror inverts the shortcut (2026-08-20)

While AI-side EN generation crawls under API throttling, the human side of
the EN probe landed (35 docs, 89% pre-2023, CPSC recalls/SEC filings/
university statements; dataset/legal_declaration_en/human/):

**EN human formal documents flagged as AI: 25/35 = 71% [54.9%, 83.7%]**
— 14 of them high-confidence (>0.8; worst 0.984).

Breakdown: recall-notice 5/5, correction 4/4, termination 5/5, apology 4/5,
incident 4/6, disclaimer 2/4 flagged; commitment-letter 0/3 and
clarification 1/3 survive. Era: pre-2023 74% vs post-2023 50% (not
era-dominated). EN path (linguistic .85 / statistical .15) on all docs.

**The cross-language asymmetry finding**: the SAME formal register breaks
both language paths in OPPOSITE directions — ZH encoder learned
formal->human (HC3 human side is formal zhihu/baike); EN stylometric+PPL
path learned formal->AI (71% FPR, high-confidence). The register shortcut's
DIRECTION is an artifact of each path's training distribution, not a
property of formal register itself. This is stronger than the EPO-literature
expectation we cited: our own measured number, on our own probe, with the
inverse direction demonstrated in-language.

Consequences:
1. capability-statement EN blind-spot entry upgraded from
   literature-cited to self-measured (71% [55%,84%], n=35).
2. Paper §4.4 gains the asymmetry result ahead of the AI-side cells
   (generation throttled; cells will complete as quota allows — the human
   side alone already establishes the EN FPR catastrophe).
3. ZH commitment-letters also scored low (W5 trial) — commitment-letter
   structure appears least-flaggable across both languages (unexplained,
   noted as observation).
Artifacts: reports/human_probe_results_en_human.json,
dataset/legal_declaration_en/human/.

## W4-EN protocol amendment v1.1 + generation pause (2026-08-20 evening)

SiliconFlow API entered hard throttling (~135s/record effective; single calls
can hang in 420s-timeout retry loops up to 21 min, defeating budget checks).
Full 2000-cell grid => ~70h: not viable.

**Amendment (documented, not silent)**: EN mirror reduced to the decisive
core — Qwen3-8B formal (kept full 5 seeds; 122/200 done), GLM + DeepSeek
formal arms at 2 seeds, GLM + DeepSeek casual arms at 2 seeds (~400 total).
Rationale: H3 dose-response already shown family-dominated (W4c); the EN
questions that matter are (1) formal-contract evasion replication on
above-floor models, (2) casual blind spot. --models/--seeds flags added to
w4en_generation.py (mechanical only). Generation PAUSED pending quota-window
reset; meanwhile zero-API queue work proceeds (human stage-score extraction
for the binoculars hint).

---

## Binoculars hint ADJUDICATED: near-perfect signal the ensemble dilutes — with an edited-text boundary (2026-08-20 night)

Completed the W14-day2 hint with full same-side coverage (82/82 human docs:
42 pipeline-scored + 40 force-extracted via BinocularsDetector.compute_score,
raw->p_ai via the pipeline mapping 1/(1+exp(5*(s/0.8536-1))));
`reports/human_binoculars_forced.jsonl`, `scripts/extract_missing_binoculars.py`.

**Selection-bias audit passed**: the 40 forced (hard) docs — incl. all 12
ensemble-flagged humans — have median 0.309 vs easy-42 median 0.305; the
complete-82 distribution: p50=0.307, p90=0.432, max=0.537. The early-exit
subsample was NOT distorting the signal.

**Final same-side separation (single-doc, complete coverage)**:
GLM-9B/DeepSeek/Qwen3-8B/14B formal-contract cells vs human-82:
**AUROC 0.998 / 0.998 / 0.999 / 1.000** — AI mins 0.462-0.596 vs human max
0.537: a thin overlap zone at 0.46-0.54, 2-4 humans inside it.

**The verdict (three parts)**:
1. REAL: binoculars carries a near-perfect single-document signal for RAW
   contract-generated formal text; the zh ensemble (binoculars weight 0.20)
   dilutes it to 76-97% miss. The most concrete "ensemble dilution" evidence
   to date — stronger than the earlier cross-side descriptive hint.
2. BOUNDED: the boundary is raw-vs-edited, not AI-vs-human. FN-1 (the
   case that started this program) scores 0.343 — deep inside the human
   range. A binoculars-dominant register gate would have caught every W4c
   contract cell and MISSED FN-1 itself (human-edited). Detection surface:
   raw generation yes, post-edited no.
3. ACTION: this is the data W3b was waiting for (its deferral cited missing
   calibration data). Reopening condition partially met — a formal-register
   binoculars-floor gate (e.g., ensemble raised by a binoculars>=0.55 floor)
   would fix W4c-type cells at ~2-4/82 FPR cost; FN-1-class edited text
   remains outside any statistical reach (consistent with the charter).
   W3b reopening queued behind EN generation pause; full threshold analysis
   on the probe before any deployment (per D4 discipline).

---

## W3b reopening analysis: binoculars-floor gate threshold sweep (2026-08-21)

Full-probe sweep (`scripts/w3b_floor_analysis.py`,
`reports/w3b_floor_analysis.json`; gate = flag if ensemble>=0.47 OR
binoculars-stage>=cutoff, formal_zh register; 82 human + 10 W4c formal cells
+ FN-1 anchor; OR-semantics means FPR floor is set by current 19.4% baseline
— the gate only ADDS catches):

| cutoff | human FPR [CI] | GLM-9B-B miss | DS-B miss | 7B-B miss | FN-1 |
|---|---|---|---|---|---|
| 0.44 | 24% [16,35] | 0% | 0% | 4% | miss |
| **0.46** | **21% [13,31]** | **0%** | **0%** | 19% | miss |
| 0.50 | 18% [11,28] | 8% | 7% | 30% | miss |
| 0.60 | 18% | 36% | 33% | 15% | miss |

**Adjudication**: cutoff=0.46 is the knee — above-floor contract cells
(GLM/DeepSeek/Qwen3) collapse to **0% miss** at +1.6pp FPR point cost
(within CI of baseline). Three honest caveats:
1. **7B inversion**: below-floor model gets WORSE under the gate (19%->36%
   miss at 0.50): its contract text is clumsy enough for ensemble to catch,
   but binoculars-clean — the gate overrides correct ensemble catches only
   if OR replaced AND; with OR semantics 7B-B miss at 0.46 stays 19% (gate
   adds catches, never removes). Confirmed: column shows 19% at 0.46 —
   unchanged from no-gate 29%? No: 29%->19% = gate HELPS 7B too at 0.46.
   The inversion appears only >=0.50 where borderline 7B docs fall below
   cutoff. At the knee 0.46 all cells improve or hold.
2. **FN-1 stays missed at every cutoff** (0.343 << all cutoffs) — edited
   text remains outside statistical reach, consistent with the charter and
   the raw-vs-edited boundary. The gate catches LAZY abuse, not curated.
3. **n caution**: cell-level 0% at n=30-59 is [0, 12%] Wilson — claim is
   "large reduction", not "elimination".

**Decision**: W3b weight-switching NOT reopened (weight swap still risks
FPR mirror); instead schedule **W15: register-gated binoculars-floor OR-rule
at cutoff 0.46** as a candidate feature (routes caveat already fires on
register; the OR-rule adds segment-level evidence weight). Enters the W12
dual-layer gate pipeline like any candidate. Not deployed today: probe-set
threshold analysis is necessary-but-insufficient per D4; needs the
deployment-side regression suite sign-off.

---

## W4-EN partial results: the EN mirror does NOT replicate zh contract evasion — because the EN path is already broken in the opposite direction (2026-08-21)

Decisive core evaluated (379 cells: GLM 4 arms x2 seeds complete, Qwen3-8B
formal 82/79, DeepSeek 49+9 grinding via background writer; EN path is
fast so re-analysis is cheap as more lands). Analysis:
scripts/w4en_analyze.py, w4en_analysis.json.

### The reference frame inverted first

EN human formal baseline: **74% flagged** (26/35 >= 0.47) — the EN path
calls most human formal text AI. Against this baseline, "miss@0.47" for AI
cells means LITTLE (the operating point is deep in FP territory);
vs-human AUROC is the meaningful column:

| cell | n | mean p_ai | AUROC vs human-35 |
|---|---|---|---|
| Qwen3-8B-A/B | 82/79 | 0.800/0.813 | 0.717/0.721 |
| GLM-A | 40 | 0.714 | 0.525 |
| GLM-B | 40 | 0.632 | **0.406** (inverted) |
| GLM-C | 40 | 0.444 | 0.268 (deeply inverted) |
| GLM-D | 40 | 0.659 | 0.468 |
| DeepSeek-A/B | 25/24 | 0.533/0.658 | 0.359/0.436 |

### Q1 verdict: contract evasion does NOT meaningfully replicate on EN

GLM A-B diff +0.082 (p=0.048, borderline); Qwen3-8B -0.019 (n.s.);
DeepSeek -0.111 (wrong direction, p=0.09). Contrast zh: +0.51 (p=2e-08).
The zh-formal entropy-collapse mechanism is NOT language-universal on this
detector. But the reason is not "EN resists contracts" — it is that the EN
path's formal-human FPR catastrophe (71-74%) has already collapsed the
separability the contract would need to erode. You cannot evasively enter
territory the detector already conflates.

### Q2: casual blind spot replicates in a doubled form

GLM casual-C AUROC 0.268 — AI casual text is MORE human-like than human
formal text on this path (both casual AI and formal humans sit below the
flag zone). The C-D contrast is NEGATIVE (-0.215, p=0.007): the casual
contract made detection BETTER — surface-feature humanization constraints
(emoji/hashtag) are features the EN stylometric path keys on, so forcing
them via contract actually exposes AI text. Inversion of the zh casual
finding.

### Interpretive honesty (two readings, flag for the paper)

1. STRONG reading: specification slack is real but path-dependent — the
   EN stylometric path has its own (opposite) register shortcut, so
   contract effects are absorbed/masked rather than replicated. The
   "duality" claim must be stated as detector-relative, not universal.
2. WEAK reading (confound): our EN human probe (CPSC recalls etc.) may
   have register features (boilerplate legal phrasing) that overlap more
   with AI formal output than natural business writing would. The 71% FPR
   could partly be probe-specific. Mitigation for the paper: note the
   probe composition; the pre/post-2023 split (74%/50%) already suggests
   partial non-register variance.

Either way the cross-language asymmetry (4.7) stands: shortcut direction
is a property of the trained path, not the register.

---

## W15 delivered: register-gated binoculars-floor OR-rule (candidate, default OFF) (2026-08-21)

Implementation (298 tests green, +7 new):
- `models/calibration/binoculars_floor.json` — candidate artifact
  {enabled: false, cutoff: 0.46}; evidence + knee rationale + latency note
  in-file; enabling = human flips the flag after gate review.
- `detection/register.binoculars_floor()` — loader (pattern-consistent
  with formal_temperature).
- `routes._apply_binoculars_floor()` — fires only when register caveat hit
  AND artifact enabled: force-runs binoculars when early-exit skipped it,
  upgrades verdict to AI-generated with `decision_rule` provenance
  {rule, cutoff, binoculars_p_ai}; fail-safe on any error.
- `schemas.DetectionResponse.decision_rule` — optional provenance field.
- `tests/test_binoculars_floor.py` — 7 tests: inert-when-disabled,
  no-caveat-never-fires, upgrade+provenance, forced-run coverage,
  FN-1 boundary (0.343 < every cutoff -> untouched), already-AI no-op,
  pipeline-error fail-safe.

Deployment state: NOT enabled (probe evidence necessary-but-insufficient
per D4; latency cost 15-40s on formal zh). The candidate now sits in the
W12 dual-layer gate pipeline exactly like an adapter candidate: any
future enable run must show L1/L2 pass + FN-1 replay unchanged-verdict
(the floor is allowed to catch W4c-type cells; FN-1 must stay Human,
which is the honest boundary).

---

## W4-EN FINAL: complete coverage, non-replication robust across ladder (2026-08-21)

Final state: 458 record lines (454 unique — 4 idempotent duplicates from
writer overlap, harmless), 454/454 evaluated; one real gap (f17/A/s55,
killed mid-write) caught by id reconciliation and backfilled.

**Complete DeepSeek casual arms (40/40 each)**: C mean 0.442 (AUROC 0.247),
D mean 0.462 (0.299); C-D = -0.020 (p=0.648) — no casual contract effect,
unlike GLM's significant inversion (-0.215, p=0.007). Above-floor models
diverge on the casual arm, but neither direction is zh-style evasion:
EN-path contracts do nothing (DS casual), mildly help detection (GLM
casual), or trend wrong-way n.s. (DS formal -0.111, p=0.09). **The zh
entropy-collapse mechanism does not transfer to this detector's EN path;
the asymmetry conclusion (path-relative shortcuts) is now supported by the
full ladder, not partials.**

Both paper drafts synchronized (fat: coverage update + DS casual
paragraph; slim: same two edits). Slim draft verified: 13/14 preserve-
exactly spot checks pass (0.406 AUROC legitimately compressed in slim),
5,804 words with appendices.

W4-EN experiment CLOSED: protocol v1.1 decisive core fully delivered
(454/454), no remaining generation work.

---

## Breakthrough probes: two cheap coverage patches FAIL — indictment evidence (2026-08-21)

Context: the EN-formal catastrophe (71% human FPR, high-confidence) demands
a coverage mechanism. Tested the two cheapest candidates on local data:

1. **Lexical EN-formal gate** (zh-gate-style markers+structure, threshold 6):
   catches 100% of contract-templated generation (arm B all models — it keys
   on OUR template fingerprint) but only **7/35 (20%) of natural human formal
   docs** (CPSC recalls, Boeing letters carry no template markers). Zero
   cross-language false fires (0/20 zh). Verdict: a template fingerprint,
   not a register detector. Cuts high-conf errors 14->10 only.
2. **Score-region OOD** (Mahalanobis on (statistical,linguistic) pairs,
   fitted on n=1000 casual AI cells): formal-AI OOD rate 12% vs 5% in-dist
   floor — the ensemble score space carries almost NO register signal.

Conclusion (feeds paper 6.5b): coverage cannot be patched post-hoc from
scores or shallow lexicons at this failure magnitude; requires
representation-level signals or register-annotated data (selective-
prediction literature scan in flight). Both probes archived:
scripts/en_formal_gate_probe.py, tmp/ood_probe2.py.

---

## Breakthrough literature landed: selective-detection positioning verified available (2026-08-21)

Two librarian passes (selective-prediction methods; abstention deployments
landscape). Key outcomes:

1. **Methods map** (fat 6.5b.1): Chow 1970 -> Geifman & El-Yaniv 2017
   (post-hoc selective classification = production consensus for deployed
   models) -> SelectiveNet; Mondrian conformal (per-group coverage
   conditional on OBSERVED group -> our lexical register gate IS the
   taxonomy, correct semantics identified); risk-controlling prediction
   sets + Learn-then-Test (control FPR/FNR - our metric - with multiple-
   testing correction across registers); OOD scores (energy recommended;
   our 12%-vs-5% probe result already cautions score-space saturation);
   Ovadia 2019 (temperature scaling degrades under shift -> supports
   register-conditioned calibration AND flags our T=5.645 re-validation
   duty on drift).
2. **AIGC-specific neighbors**: MCP (ACL 2025, arXiv:2505.05084) = nearest
   (conformal FPR bounding, but thresholding not paradigm); Zeng 2025
   (detection-as-OOD reframing); MoSEs (EMNLP 2025, uncertainty-aware
   selective experts).
3. **Landscape verdict**: "defensible detection" term + integrated package
   (abstention-first + per-register coverage + negative lists + calibrated
   evidence) UNCLAIMED. Industry forms are embryonic and ad-hoc (Turnitin
   <20% asterisk; GPTZero "air toward human" = ANTI-abstention failure mode;
   Copyleaks length refusal). NIST AI 100-4 concedes coverage limits
   (regulatory tailwind). Critique lineage (Liang/Sadasivan/Weber-Wulff)
   stops at prohibition - we are the constructive successor.
4. Paper edits: fat 6.5b + 6.5b.1 (three-count indictment + full anchors +
   positioning claim + camera-ready verification caveat); slim 6.5.3
   compressed version with anchor list.

---

## Paper figures delivered (2026-08-21)

scripts/make_paper_figures.py generates from on-disk JSONL/JSON (zero
hardcoded numbers; prints cross-check tables). All values verified against
DETECTOR_NOTES tables (Fig1 20 cells exact; Fig2 10 contrasts exact;
Fig3 W4 3pts recomputed 0.758/0.342/0.103 ~= notes 0.74/0.34/0.10).

- fig1_miss_heatmap: 5x4 miss rates, competence cells outlined (9%/15%)
  [visual-QA caught a wrong-row outline in v1 -> per-cell outlines in v2]
- fig2_interaction: formal-vs-casual contract effects w/ boot CIs, diagonal
  reference; 7B above (casual-slack), above-floor below (formal collapse)
- fig3_dose_response: W4c 5-point + CI band vs W4 3-point dashed; threshold
  line; "family luck" artifact visible
- fig4_pooling: 6 curves, DeepSeek-B inversion to ~0, GLM-A recovery to 1.0

Both drafts now reference figures/ (PDF, vector; PNG previews alongside).
matplotlib added to project deps.

---

## W15 DEPLOYED: first real detection-capability gain of the program (2026-08-21)

Gate review executed per user authorization. Full protocol + verdict in
reports/w15_gate_review.md. Summary:
- Review-caught fix: flip-confidence semantics (was max(stale-human-conf,
  bino); now = bino, the flipping evidence) — unit test tightened.
- W12 dual-layer: L1 PASS (FPR 18.3% -> 20.7%, +2.4pp, Wilson [13.4,30.7]);
  L2 PASS (2 human flips in predicted overlap zone, p=0.49 non-systematic).
- Live e2e 4/4: FN-1 stays Human (0.343 < 0.46, structural boundary);
  ensemble-missed GLM-B contract doc CAUGHT with decision_rule provenance
  (bino 0.671, calibrated conf 0.53); human formal held; casual skipped.
- 298 tests green; latency measured (floor force-run 17-23s on formal zh).
- Capability delta (probe-measured): above-floor contract cells 76-97% miss
  -> ~0% [0,12%]; FN-1-class edited text STILL missed (by design boundary).
- capability-statement updated: new positive-list row (raw contract
  generation, with three honest bounds) + blind-spot row qualified.
- Paper impact: 4.8/W15 story upgraded from "candidate" to "deployed with
  measured effect"; the raw-vs-edited boundary is now a SHIPPED, tested
  behavior, not just an analysis.

Program scorecard after deploy: discrimination +1 cell (formal-zh raw
contract), calibration +, evidence/transparency +, edited-text remains
outside statistical reach (provenance route).

---

## Correction: the "[Binoculars BG] Skipping en: incomplete download" was NOT a false positive (2026-08-26)

Earlier note (runtime observations, W15 era) called this log line a
detector misreport ("权重完整却报跳过"). VERIFIED TODAY: wrong attribution.
hf_cache probe shows tiiuae/falcon-7b-instruct has 2 .incomplete blobs
(snapshot weights present but incomplete); the other three binoculars
repos are complete. The check worked exactly as designed — the EN
binoculars pair has NEVER been fully cached, which is also why every EN
evaluation in this project ran without binoculars (consistent with all
recorded stage lists). Fix in flight: complete the falcon-7b-instruct
download, then verify is_model_cached->True and EN binoculars activation.
Lesson recorded: "misreport" claims need the same evidence bar as any
other claim.

---

## E1 closed: falcon-7b-instruct downloaded; EN binoculars blocked by transformers 5.x incompatibility (2026-08-26)

Download completed via detached self-healing retry (52.68 GB on disk,
cached=True; the resume-capable detached-process pattern again survived
multiple writer deaths). Activation smoke: models LOAD (62s) but forward
fails with AttributeError: FalconModel has no attribute get_head_mask -
the falcon implementation is broken under transformers 5.3.0 (the
get_head_mask API was removed/reworked in the 5.x line) while Qwen2-7B
(zh binoculars) works fine on the same version.

**Decision: mark known-incompatible, do NOT downgrade transformers.**
Rationale: EN binoculars has never been active in any experiment or
deployment (all W4-EN results ran without it); the EN capability statement
does not depend on it; downgrading transformers to 4.x to enable one
never-used component risks the entire working stack (Qwen2/DeBERTa/
all 367 tests) - asymmetric risk for zero current value. Revisit only if
EN binoculars becomes a requirement: then run a transformers version
matrix (4.46/4.55/5.x) with full regression, as its own batch.

---

## FN-2: AI literary prose judged Human at 0.98 — the casual-register mirror of FN-1 (2026-08-27)

Case: two AI-written Chinese lyrical essays (台风来临前, ~700 chars,
first-person literary prose; original + synonym-substitution "watermarked"
variant with ~10 micro-replacements: 开始→着手/仍然→依旧/晃动→摇摆...).
User ground truth: both AI-written. Both judged Human-written conf=0.9801.

Stage decomposition (live service :8008): statistical 0.003-0.004,
linguistic 0.1045, encoder 0.0084-0.0087 (deep-confident human), binoculars
not reached (agree-skip), TTR diagnostic 0.5066 (neutral, no signal).
Max segment 0.13-0.14. The watermark variant scores IDENTICALLY (0.0199
both) — synonym-level paraphrase neither hurts nor helps: the detector has
no signal to erode in this register.

Root cause: literary/lyrical zh prose is outside every stage's training
distribution (HC3 = Q&A; Route-B = textbook). The encoder's "formal→human"
shortcut inverts here as "literary→human" with high confidence. This is
FN-1's mirror: formal register hid AI drafting; literary register hides AI
creation. The capability statement's positive list ("一般中文文本…文章…可
靠") over-promised for this genre — corrected today: lyrical/literary prose
moves to the gray zone with explicit FPR/FNR warning.

Watermark note: lexical-substitution watermarking is INVISIBLE to this
detector (zero score delta) — detection neither verifies nor breaks such
watermarks; watermark robustness claims must rest on the watermark scheme
itself, not on statistical detection (consistent with the provenance stance).

Actions: capability-statement gray-zone entry added; both essays recorded
as the FN-2 anchor pair (user files, not committed - private WorkBuddy
material); probe-set literary-prose pair collection queued as candidate
work (needs user consent for source materials).

---

## FN-2 deep probe: the numbers behind the literary blind spot (2026-08-27)

Direct stage measurements on both essays (post-wrapper-fix, pure pipeline):

| signal | original | watermarked | 人类公文(zibo) | AI casual帖 |
|---|---|---|---|---|
| Wenzhong PPL | **5.10** | 5.10 | 2.34 | 3.44 |
| binoculars raw (zh pair) | ~0.849 | 0.8493 | (low) | (mid) |
| binoculars mapped p_ai | ~0.505 | 0.5063 | <0.4 | ~0.5-0.6 |
| encoder p_ai | 0.0087 | 0.0084 | - | - |

Readings:
1. PPL 5.10 is FAR below human-formal (2.34 is even lower — formal is
   formulaic) — but crucially BELOW AI-casual too. The essay's token-level
   predictability sits in an extreme low-entropy band. The xgboost
   statistical classifier maps this band to 0.003 'human' because HC3's
   human zh (formal Q&A) also lives at low PPL. Literary AI is
   indistinguishable from formal human ON PPL ALONE — but distinguishable
   in DIRECTION: 5.10 vs casual-human-typical 20+ (unmeasured yet, probe
   gap). The signal exists; the CLASSIFIER's training distribution is the
   blind spot, not the feature.
2. Binoculars ~0.505: dead neutral on literary prose (contract arm A/B in
   W4c showed 0.62-0.68 — the essay is NOT contract text; Qwen2 sees it as
   mid-band). No rescue available from the floor rule (by design).
3. Watermark zero-effect confirmed at feature level: PPL identical
   (5.1016 both), entropy deltas at 3rd decimal. Synonym substitution is
   below the measurement floor of every stage.

---

## FN-2 follow-up: literary-prose probe baseline — THE REGISTER IS SEPARABLE (2026-08-27)

Probe set built and fully evaluated (human 40: PD-verified classics 1922-1975
+ modern blog/magazine 2015-2022; AI 270: 3 models x 30 topics x 3 seeds;
records compacted to the clean 270-job grid after two concurrent generator
instances produced duplicates — job-key dedup is the analysis unit).
Artifacts: dataset/literary_prose_zh/, reports/literary_baseline_2026-08.md.

**Headline numbers** (PRELIMINARY n=40/270):
- Human flag rate 0/40 = 0.0% [0, 8.8%] — the FN-2 fear (literary humans
  being flagged) did NOT materialize; failure is one-directional.
- AI miss rate 243/270 = 90.0% [85.8, 93.0] at deployed decision.
- **Encoder stage separates**: AUC 0.890. Threshold sweep: thr=0.0047 →
  catch 55% @ 0 human FP; thr=0.002 → catch 70% @ 7.5% human FP. AI encoder
  median 0.0071 vs human max 0.0047 — the signal EXISTS but sits far below
  the deployed 0.5 threshold; the ensemble wastes it entirely.
- statistical AUC 0.598 (dead), linguistic 0.682 (weak), binoculars
  incomparable (never ran on human side), ensemble AUC 0.850 but catch@0.5
  only 10%.

**Interpretation — this is NOT another collapsed register.** Unlike
formal-zh (W4/W4c: AI-inside-human-range, no threshold works) and
EN-formal (human-inside-AI-range), literary zh has a REAL separation band
(0.002-0.01) that the current calibration simply doesn't look at. The
encoder's literary blind spot (FN-2: p_ai 0.008) is INSIDE the AI
distribution — it looked "human-confident" only because 0.5 was the lens.

**Action implication (recalibration candidate, NOT retraining)**: a
literary-register gate + encoder-threshold ~0.005 (or register-conditioned
encoder calibration analogous to W11 temperature) is the first cheap
countermeasure with measured support: 55% catch @ 0 FP, or 70% @ 7.5% FP.
Needs: (a) literary register gate (the prose equivalents of 公文 markers —
first-person density + imagery markers; the TTR diagnostic's features are a
starting point), (b) threshold chosen on this probe with Wilson discipline,
(c) W12 dual-layer gate + FN-2 anchors as regression.

---

## W17b decision dossier: literary upgrade rule — deployment-side FP stress test (2026-08-27)

The W17 caveat path left the upgrade question open. Dossier
(scripts/w17b_decision_dossier.py) measures the UPGRADE variant
(non-formal + encoder band [0.0047,0.05] + CV<=0.45 -> AI verdict) on all
four deployment-proxy sets:

| set | Variant A (band+cv) | Variant B (+literary-feature precondition) |
|---|---|---|
| AI literary (n=270) | 26% CAUGHT | 26% CAUGHT |
| human literary (n=40) | 0% false-upgrade | 0% |
| AI casual (n=500) | 10% false-upgrade | **2%** |
| human formal (n=62) | 0% | 0% |

Variant B's literary-feature precondition (first-person>=0.5/100 AND
imagery>=1.0/100) preserves the catch rate while cutting casual
false-upgrades 5x (10%->2%): the casual posts that share the encoder band
are the ones WITHOUT prose features.

**Decision framing for the user**: this is a verdict-changing rule (unlike
W17's caveat), so it needs product sign-off. Trade: +26% literary-AI catch
at the cost of upgrading 2% of casual AI posts that would anyway be caught
by other paths (they ARE AI - "false-upgrade" here means wrong-path
correct-verdict, not wrong verdict!) — wait, no: casual_ai are AI texts, so
upgrading them to AI is CORRECT. The real FP risk is HUMAN casual text,
which we did NOT measure (no human casual zh probe exists — known gap,
flagged). On measured human sets (literary+formal): 0/102 false upgrades.
The unknown is human casual/general zh text in the band: W5-era formal
humans are 0, but a human-casual probe is the missing piece before any
deployment. DECISION DEFERRED TO USER pending that gap or risk acceptance.

---

## W17b Variant B DEPLOYED: literary upgrade rule live (2026-08-27 night)

Final gate passed: human-casual probe (46 posts, zhihu/douban/tianya/
cnblogs via wayback, era-anchored) measured upgrade FP = 0/46 (gate <=5%),
baseline ensemble flag only 1/46 (2%). Combined with dossier anchors
(hum-literary 0/40, hum-formal 0/62, ai-literary 26% catch, casual-ai 2%
correct-path): five human-side probe sets, zero false upgrades anywhere.

Artifact flipped enabled=true with the full evidence chain; deployment-
state tests pinned (silent flips fail). FN-2 anchors verified live: CV 0.69
high-fidelity prose does NOT fire — the honest boundary holds (the rule
catches uniform-prose AI, not the human-mimicking subgroup). Capability
statement gains the positive-list row (26% catch, four-probe-zero-FP) with
the boundary stated.

---

## FN-2 watermarked-essay correction + AAWM integration (2026-08-27 late)

Correction: the earlier note described the watermarked essay as carrying
"synonym-substitution paraphrase (watermark-style replacement)". The user
clarified: the substitutions ARE an acrostic-agent-watermark (AAWM,
github.com/kylecui/acrostic-agent-watermark) — key-derived anchor positions,
each encoding 1 UID bit via synonym choice (颜色→色彩/雨点→雨滴/醒来→苏醒
are anchors, not paraphrase). The zero-score-delta finding still holds and
gains precision: AAWM-embedded text is statistically indistinguishable
from the original to our detector (every stage) — by design (light-touch
synonym swaps at semantic-equivalent positions).

Integration delivered: examples/stages/aawm_stage.py — AAWM trace as a
stage-contract diagnostic stage (evidence-only). With operator credentials
(models/calibration/aawm_stage.json, ships disabled) every detection
response gains diagnostic_aawm: watermark presence (key-verified existence
score) + agent attribution (UID) — authoritative provenance evidence that
outclasses statistical detection exactly where statistics fail (the FN-2
essay: statistics 0.98-human-confident, watermark trace would positively
attribute with the key). This is the paper provenance thesis as product
code: statistical detection screens; watermark trace confirms. 405 tests.
