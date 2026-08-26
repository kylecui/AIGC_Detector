# Specification Slack: How Output Contracts Move LLM Text Into Human Territory — and Shrink AI-Text Detectors to a Single Competence Cell

**Date**: 2026-08-21
**SLIM DRAFT v0.1 — derived from w13-fat-draft.md**

---

## Abstract

We report evidence that output contracts systematically move LLM-generated text into the human text manifold, rendering statistical detection ineffective. Through a 2×2×5×5 controlled experiment (register × prompt-contract × model × seed, n=2000 plus pilot rounds totaling n=400), we demonstrate that contract-constrained generation becomes significantly less detectable than free-form generation—but only when the model has sufficient capability to execute the contract. The detector's reliable competence region contracts to a single cell: formal register × crude/traceable models (GLM-4-9B at 9% miss rate, Qwen2.5-7B at 15%). All other conditions show majority missed detection: casual register misses 60-100% even free-form (the formality shortcut is bidirectional), and fluent models (DeepSeek-V3.2) miss 77% even on formal-free. We formalize this as specification slack theory: contracts remove model entropy, and the detector's signal measures precisely this residual slack. Where slack remains (formal × crude models), detection works; where contracts annihilate slack (formal × fluent models) or slack was never present (casual register), detection fails. Mechanism varies by register: formal contracts cause entropy collapse (perplexity compression), while casual contracts cause surface-feature humanization (emoji/hashtag injection). We document the detector's lifecycle through a false-negative case study, an adversarial training failure, and a principled decision to defer retraining—showing that in collapsed registers, the defensible product form is "instrument + boundary declaration," not judge.

---

## 1. Introduction

AI-text detection has focused on accuracy metrics, but a more fundamental question remains: where do detectors still work? We report evidence that the answer, for a multi-stage ensemble detector on Chinese text, is: in exactly one cell of the register×capability space—formal register × crude/traceable models.

Our starting point was a false-negative case (FN-1): an AI-drafted Chinese compliance declaration was classified as human-written with 89% confidence by a four-stage detection pipeline (statistical, linguistic, encoder, binoculars). All four stages failed independently. Root cause analysis revealed three mechanisms: (1) manifold collapse (the Perplexity Trap—formal register constrains variation, collapsing human and AI distributions), (2) encoder formality shortcut (training data imbalance taught "formal→human" as a spurious feature), and (3) aggregation dilution (max-segment signal 0.856 diluted to 0.109 by document-mean aggregation).

This failure pattern mirrors external literature. Banerjee 2026 (arXiv:2607.13044) formalized the Perplexity Trap: when external syntactic constraints apply to both human and AI output, restricting variation on linguistic axes, distributions collapse onto the same low-entropy manifold. On EPO patent claims, mainstream detectors show >60% FPR—our FN-1 is the FNR-side mirror. Smirnov 2026 showed that fine-tuned detectors collapse their 768-dim representations to ~1 dimension aligned with formality (cos 0.73-0.99), not authorship.

Crucially, the FN-1 document was generated under an output contract (CPCC correction requirements: structure, mandated clauses, legal phrasing, table schema, signature block). This suggested a dual measurement axis: detectability distance (free-form vs contract-constrained generation) as a proxy for "uncontracted model slack."

### 1.1 Key Findings

![Figure 1](figures/fig1_miss_heatmap.pdf)

The heatmap reveals three findings:

1. The competence region is ONE CELL: formal register × crude/traceable models (GLM 9%, 7B 15%). Everything else is majority-missed.
2. Casual register is a NEW blind spot: casual free-form already shows 60-100% missed (the formality shortcut is bidirectional—casual style reads as human).
3. Contracts annihilate the remaining formal coverage: GLM formal-free (9% miss) → formal-contract (76% miss); Qwen3-8B formal-free (49% miss) → formal-contract (93% miss).

### 1.2 Contributions

1. **2×2 factorial evidence of contract-driven evasion with capability gating**: We conduct a 2×2×5×5 controlled experiment (register × prompt-contract × model × seed, n=2000) with pre-registered hypotheses and topic-block bootstrap analysis. Contracts significantly reduce detectability only for above-floor models (GLM Δ=+0.513, p=2e-08; DeepSeek Δ=+0.213, p=1e-05); below-floor models show null or inverted effects.

2. **Specification slack theory as a unifying explanation**: We formalize the detector-contract duality: contracts reduce model entropy, and the detector's statistical signal measures precisely this residual slack. Mechanism varies by register: formal contracts cause entropy collapse (perplexity compression, largest single-stage effect d_z=2.44), while casual contracts cause surface-feature humanization (emoji/hashtag injection).

3. **Empirical demonstration of bidirectional register shortcuts, across registers AND languages**: The register shortcut is bidirectional: formal-register prior betrayed us on FN-1 (AI formal → human, 0.89 conf), casual-register prior betrays us in the 2×2 (AI casual → human, 60-100% missed even free-form). The English mirror extends the inversion across language paths: the SAME formal register yields 71% [54.9%, 83.7%] FPR on human EN formal documents (formal → AI) while the ZH path yields 19% — the shortcut's direction is an artifact of each path's training distribution. We further isolate a near-perfect Binoculars-stage signal the ensemble dilutes (AUROC 0.998–1.000 vs human-82) whose boundary is raw-vs-edited generation, not AI-vs-human: the FN-1 document (human-edited AI text) sits inside the human range at every threshold.

4. **Open probe set with regression framework and transparent non-independence disclosure**: We provide the legal_declaration_zh probe set (human side n=62, AI side n=320 from paired experiments) with Wilson intervals, era-stratified validation, and a corpus-level pooling time-box. We disclose that the harness study and this experiment share the same author and model pool—this is a "dual-domain examination," not independent validation.

---

## 2. Specification-Slack Theory

### 2.1 Core Thesis

Output contracts systematically remove model entropy, and zero-shot statistical detection measures precisely this residual slack. When a contract is complete enough, the contracted output distribution M_c becomes statistically indistinguishable from the target human distribution H: TV(M_c, H) → 0. Consequently, detection AUROC is information-theoretically bounded below 1/2.

This thesis unifies three empirical observations:

1. **FN-1 (formal contract collapse)**: A legal declaration generated under strict output contract achieved ensemble p_ai=0.109 (Human, 0.891 conf) despite being AI-drafted. All four detection stages failed independently. The Perplexity Trap (Banerjee 2026) formalizes this: when external syntactic constraints apply to both human and AI output, restricting variation, distributions collapse onto the same low-entropy manifold.

2. **W4/W4b paired experiments (capability gating)**: Contract-arm detectability is monotone in contract-adherence capability under the SAME contract: ensemble p_ai falls 0.74 → 0.34 → 0.10 as capability rises (7B → GLM → DeepSeek). The below-floor model gains nothing from the contract (cannot execute it into the human manifold); above-floor models dive deep into "human" territory.

3. **W4c 2×2 interaction (register-dependent mechanism)**: The contract effect differs by register. On formal register, contracts show strong effects for above-floor models (GLM Δ=+0.513, Qwen3-8B Δ=+0.359). On casual register, contracts show effects primarily for below-floor models (7B Δ=+0.364—the largest casual effect, from the weakest model); fluent models are already at the miss ceiling free-form.

### 2.2 Formalization

Let:
- S = task specification (output contract)
- c(S) ∈ [0,1] = contract completeness
- M₀ = model's free-form generation distribution
- M_c = model's generation distribution under contract S
- H = human text distribution for the target register

**Prediction**: c(S)↑ ⇒ TV(M_c, H)↓ ⇒ detection AUROC↓

The Perplexity Trap formalizes three sufficient conditions for collapse:
- C1: Output constrained by external syntactic requirements
- C2: Constraints apply to both human and AI output
- C3: Constraints restrict variation on at least one linguistic axis

When C1-C3 hold, human and AI distributions collapse onto the same low-entropy manifold. This is structural, not statistical—no amount of detector tuning can recover separability because the distributions genuinely overlap in the feature space the detector uses.

### 2.3 Register-Dependent Mechanisms

Specification slack theory predicts that contract effects are maximal where slack remains:

- **Formal register × above-floor models**: Slack exists in entropy space. Contracts cause entropy collapse—perplexity compression is the core mechanism (GLM statistical stage Δ=+0.62, d_z=2.44).
- **Casual register × below-floor models**: Slack exists in surface-feature space (the model cannot mimic casual register convincingly). Contracts cause surface-feature humanization—emoji/hashtag/colloquial spec injection moves output toward casual human style (7B casual Δ=+0.364).
- **Formal register × below-floor models**: No slack—the model cannot execute the contract at all (7B formal Δ=+0.127, but B-arm stays detected at 0.704).
- **Casual register × above-floor models**: No slack—fluent models already at the miss ceiling free-form (DeepSeek casual-free 100% missed).

### 2.4 Information-Theoretic Bound

Sadasivan et al. (arXiv:2303.11156) proved an information-theoretic bound for any likelihood-based detector:

AUROC ≤ 1/2 + TV(M,H) − TV(M,H)²/2

Where TV(M,H) is total variation distance between AI and human distributions. Strongly constrained registers by construction shrink TV. When TV → 0, AUROC → 0.5 (random guessing). This is not a detector flaw—it's an information-theoretic limit.

Chakraborty et al. (arXiv:2304.04736) showed that multi-sample pooling can restore separability via Chernoff information when n = Ω(1/δ²). This suggests corpus-level detection as a potential route—but we empirically falsified this for contracted formal register (W14 corpus-level pooling: DeepSeek AUROC 0.297 → 0.002, INVERTED).

### 2.5 Detector-Contract Duality

This experiment and the contract-driven-harness study measure the same quantity from opposite angles:

- **Detector measures residual slack**: Perplexity, burstiness, Binoculars cross-perplexity ratio, stylometric variance—all consume statistical slack left by free generation.
- **Contract annihilates residual slack**: Externalizing obligations removes slack from the sampling process.

**Duality proposition**: Contract completeness and statistical undetectability are two instruments measuring the same underlying quantity. When c(S) reaches "complete" (output contract covers structure, phrasing, format, boundaries), M_c becomes statistically exchangeable with H at the text level.

FN-1 is empirical validation: the declaration was generated under an output contract (CPCC correction requirements), and industrial-grade four-stage detection measured near-zero residual slack (p_ai=0.109). This is the strongest empirical evidence to date for the harness study's "binding constraint" thesis—not at the behavioral level (40/40 pass rate), but at the distributional level (statistically indistinguishable from human).

---

## 3. Experimental Design

### 3.1 Overview

We designed a 2×2×5×5 factorial experiment:

- **Register**: formal (公文-style legal/regulatory) vs casual (social-media style)
- **Prompt-contract**: free-form vs contract-constrained
- **Model**: 5 models spanning capability range
- **Seed**: 5 seeds per model

Total n=2000 documents (20 formal topics × 2 arms + 20 casual topics × 2 arms = 80 cells × 5 models × 5 seeds = 2000). All documents evaluated in-process using the same weights as the deployed service.

### 3.2 Factor Specifications

**Register factor**:
- **Formal**: 20 topics from legal/regulatory register. Document types: declarations, commitments, clarifications, apologies, notices, corrections, statements.
- **Casual**: 20 topics from social-media register. Content: daily life posts, opinions, reviews, casual conversations.

**Prompt-contract factor**:
- **Free-form (A/C)**: Minimal constraint: "请就{主题}写一份{文种}" (write a {document type} about {topic}).
- **Contract-constrained (B)**: Full specification: mandated clause order, legal phrasing, table schema, signature block, length band, format constraints.
- **Casual contract (D)**: Surface-feature specification: emoji count target, hashtag count target, colloquial phrase usage.

**Model factor**:
- Qwen2.5-7B-Instruct: Local 4-bit quantization, below-floor candidate
- Qwen3-8B: served via API (SiliconFlow), server-side stochastic sampling
- GLM-4-9B-0414: SiliconFlow API, above floor (30/40 strict contract adherence in harness study), FN-1 family
- Qwen3-14B: served via API (SiliconFlow), server-side stochastic sampling
- DeepSeek-V3.2: SiliconFlow API, top-tier (40/40 strict contract adherence in harness study)

**Seed factor**: 5 seeds per model (11-55 for local 7B; server-side stochastic for API models).

### 3.3 Generation Parameters

- Temperature: 0.7 for all generations
- Top_p: 0.9 for all generations
- Length floors: Formal contract ≥800 chars; Casual contract 280-330 chars
- Length ceilings: None (models self-terminate; some DeepSeek texts ran long, up to 1700 chars)

### 3.4 Evaluation Pipeline

All 2000 documents evaluated through the four-stage detection pipeline:

1. **Statistical stage**: Wenzhong-110M language model, character-level perplexity
2. **Linguistic stage**: 14-feature XGB classifier (micro/meso/macro statistics, sentence length CV, lexical diversity MTLD, paragraph length variance, human-likeness score)
3. **Encoder stage**: chinese-roberta-wwm-ext + LoRA v2 (Route B retrained with 10× textbook oversampling)
4. **Binoculars stage**: Qwen2-7B base/instruct cross-perplexity ratio (4-bit quantization)

**Ensemble weights**: {stat: 0.10, ling: 0.10, enc: 0.60, bino: 0.20}. Decision threshold: 0.47 (trained pre-experiment, no tuning on W4c data).

### 3.5 Hypotheses (Pre-registered)

**H1 (formal A>B)**: On formal register, contract-constrained generation (B) is less detectable than free-form (A).

**H2 (casual C>D)**: On casual register, contract-constrained generation (D) is less detectable than free-form (C).

**H3 (dose-response)**: Across formal-B cell, detectability decreases monotonically with model capability (7B > 8B > 9B > 14B > DeepSeek).

### 3.6 Statistical Analysis

**Per-cell summary**: Mean p_ai with 95% bootstrap confidence intervals (B=2000 topic-stratified bootstrap).

**Hypothesis testing**:
- H1/H2: Per-topic paired Wilcoxon signed-rank test (20 topics per model, paired A-B or C-D). Sign test as secondary. Effect size: Cliff's delta.
- H3: Spearman correlation across models, monotonicity test with 2% tolerance.

**Topic-block bootstrap**: For each model, bootstrap resamples at the topic level (with replacement, 20 topics). B=2000 iterations, report 2.5th and 97.5th percentiles as CI.

### 3.7 Pilot Rounds

**W4 (n=240)**: Initial 3-model experiment (Qwen2.5-7B, GLM-4-9B, DeepSeek-V3.2). Confirmed capability gating: contract-arm p_ai falls 0.74 → 0.34 → 0.10 as capability rises.

**W4b (n=80)**: Exact replication with fresh server-side sampling. Both above-floor models replicated: GLM Δ=+0.574 (p=4.0e-05), DeepSeek Δ=+0.182 (p=0.001).

**Total pilot n=320**: Provided statistical power estimates and informed the 5-model × 5-seed expansion for W4c.

### 3.8 Caveats

1. **GLM casual-D non-compliance**: 18/22 outputs <120 chars (spec asked 280-330). GLM violated the casual length constraint. Its casual-D estimate is confounded by non-compliance (noted, not excluded).
2. **Detector threshold fixed**: 0.47 trained pre-experiment. Miss rates are at deployed operating point, not AUROC.
3. **Seed control asymmetry**: 7B local sampling seeded (11-55); API models server-side stochastic.

---

## 4. Results

### 4.1 Main Results: Single Competence Cell

[Table 1: Main results. Mean p_ai [95% boot CI] / miss rate for all 20 cells. N=100 per cell.]

| Model | formal-A (free) | formal-B (contract) | casual-C (free) | casual-D (surface-feature) |
|-------|----------------|---------------------|----------------|---------------------------|
| Qwen2.5-7B | 0.831 [0.750, 0.906] / 15% | 0.704 [0.604, 0.802] / 29% | 0.458 [0.345, 0.579] / 60% | 0.094 [0.076, 0.113] / 99% |
| Qwen3-8B | 0.541 [0.432, 0.658] / 49% | 0.182 [0.133, 0.235] / 93% | 0.085 [0.036, 0.152] / 95% | 0.036 [0.030, 0.041] / 100% |
| GLM-4-9B | 0.854 [0.795, 0.911] / 9% | 0.340 [0.256, 0.431] / 76% | 0.183 [0.121, 0.249] / 87% | 0.056 [0.055, 0.057] / 100% |
| Qwen3-14B | 0.619 [0.526, 0.715] / 36% | 0.329 [0.249, 0.418] / 77% | 0.103 [0.071, 0.142] / 98% | 0.036 [0.028, 0.046] / 100% |
| DeepSeek-V3.2 | 0.291 [0.201, 0.379] / 77% | 0.078 [0.052, 0.109] / 97% | 0.036 [0.025, 0.049] / 100% | 0.022 [0.016, 0.029] / 100% |

[Figure 1: 5×4 heatmap of miss rates. Only formal-A cells show non-majority misses: GLM-4-9B at 9%, Qwen2.5-7B at 15%. All other cells are 60-100% missed.]

**Key observation**: The detector's competence region is ONE CELL: formal register × crude/traceable models. Everything else is majority-missed.

### 4.2 Hypothesis Verdicts

**H1 (formal A>B)**: REPLICATED for all 5 models (p ≤ 7e-03 for all). Even the below-floor 7B now shows a small significant formal effect (+0.127, p=0.007), which W4 missed.

[Table 2: H1 verdicts.]

| Model | Δ (A-B) | Wilcoxon p | Boot CI | Sign test (pos/neg) | Verdict |
|-------|---------|------------|---------|---------------------|---------|
| Qwen2.5-7B | +0.127 | 7.3e-03 | [0.049, 0.208] | 14+/6− | Significant |
| Qwen3-8B | +0.359 | 1.9e-05 | [0.246, 0.479] | 19+/1− | Significant |
| GLM-4-9B | +0.513 | 1.9e-06 | [0.425, 0.598] | 20+/0− | Significant |
| Qwen3-14B | +0.290 | 1.3e-05 | [0.214, 0.370] | 18+/2− | Significant |
| DeepSeek-V3.2 | +0.213 | 9.5e-06 | [0.134, 0.294] | 20+/0− | Significant |

**H2 (casual C>D)**: Mixed. Significant for below-floor 7B (+0.364, p=2e-06—LARGEST casual effect) and GLM (+0.127, p=0.001). Null/tiny for Qwen3-8B/DeepSeek. Casual cells were already 60-100% missed FREE-FORM—contract is mostly ceiling-bound there.

[Table 3: H2 verdicts.]

| Model | Δ (C-D) | Wilcoxon p | Boot CI | Sign test (pos/neg) | Verdict |
|-------|---------|------------|---------|---------------------|---------|
| Qwen2.5-7B | +0.364 | 1.9e-06 | [0.261, 0.475] | 19+/1− | Significant (largest casual effect) |
| Qwen3-8B | +0.049 | 0.498 | [-0.0005, 0.116] | 11+/9− | Not significant |
| GLM-4-9B | +0.127 | 0.001 | [0.064, 0.191] | 15+/5− | Significant |
| Qwen3-14B | +0.066 | 4.8e-05 | [0.035, 0.104] | 18+/2− | Significant but tiny |
| DeepSeek-V3.2 | +0.014 | 0.114 | [0.000036, 0.029] | 12+/8− | Not significant |

**H3 (dose-response across formal-B)**: NOT monotone. Means: 7B 0.704 > 8B 0.182 < GLM 0.340 ~ 14B 0.329 > DS 0.078. Within-Qwen3 inverted (8B < 14B). Spearman ρ = -0.7, p=0.188. The 3-point W4 "monotone ladder" was family luck; model family/style dominates parameter count.

![Figure 3: dose-response](figures/fig3_dose_response.pdf)

### 4.3 Interaction Analysis

The W4c data confirms the register-dependent mechanism predicted by specification slack theory:

**Formal interaction (formal A-B)**: Large positive effects for above-floor models (GLM +0.513, Qwen3-8B +0.359, Qwen3-14B +0.290, DeepSeek +0.213). All significant (p ≤ 7e-03). This is the Perplexity Trap instantiated—entropy collapse is the core mechanism.

**Casual interaction (casual C-D)**: Significant only for below-floor 7B (+0.364, p=2e-06—the largest casual effect) and GLM (+0.127, p=0.001). Fluent models (Qwen3-8B, DeepSeek) show null or tiny effects (already at ceiling). This is surface-feature humanization—crude casual text has the most slack for emoji/hashtag injection.

![Figure 2](figures/fig2_interaction.pdf)

**Interpretation**: Contracts reduce detectability where slack remains. Slack location depends on register×capability:
- Formal × above-floor: slack in entropy → entropy collapse
- Casual × below-floor: slack in surface features → surface-feature humanization
- Formal × below-floor: no slack (can't execute contract)
- Casual × above-floor: no slack (already at ceiling)

### 4.4 Bidirectional Register Shortcuts

The W4c results reveal three insights beyond the pre-registered hypotheses:

1. The detector's competence region is ONE CELL: formal × crude/traceable model (GLM 9%, 7B 15%). Everything else is majority-missed.

2. Casual blind spot is NEW knowledge—W4's formal framing understated the detector's weakness. The encoder shortcut is bidirectional:
   - Formal-register prior betrayed us on FN-1 (AI formal → human, 0.89 conf)
   - Casual-register prior betrays us here (AI casual → human, 60-100% missed even free-form)

3. Capability gate refined, partially refuted: On formal register, 7B cannot execute contracts into evasion (B stays 0.704 detected)—gate survives in the FAILURE sense. On casual register, the gate INVERTS (7B shows largest contract evasion: crude casual text has most slack for surface humanization; fluent models already at miss ceiling).

### 4.5 Era-Stratified Validation

**Matched-era human slice** (n=20, 2024-2026, era-authority verified) evaluated: **FPR 3/20 = 15.0% [5.2%, 36.0%]** vs main store (82% pre-2023) **19.4% [11.4%, 30.9%]**. Intervals massively overlap. All 3 FPs are LOW-confidence (0.50-0.65), ZERO high-confidence (main store has 8).

**Verdict**: The formal-register blind spot is era-invariant. Register explanation beats era explanation; the pre-2023 anchoring critique is answered with data.

### 4.6 Corpus-Level Pooling: Negative Result

We tested whether multi-sample pooling could recover detectability in contracted formal register. k-sample pooling (k=1..20, B=2000 topic-stratified bootstrap; both mean-p and mean-logit pooling).

[Table 4: Corpus-level pooling results.]

| Cell | AUROC k=1 → k=20 | TPR@FPR5% best | Verdict |
|------|-----------------|----------------|---------|
| GLM formal-free (A) | 0.924 → 1.000 | 1.00 @ k=5 | RECOVERED |
| GLM formal-contract (B) | 0.691 → 0.899 | 0.39 | NOT recovered |
| Qwen3-14B (B) | 0.704 → 0.869 | 0.27 (logit 0.18) | NOT recovered |
| Qwen3-8B (B) | 0.512 → 0.223 | 0.01 | NOT recovered (INVERTS) |
| DeepSeek (B) | 0.297 → 0.002 | 0.00 | NOT recovered (INVERTS) |
| DeepSeek (A) | 0.573 → 0.729 | 0.12 | NOT recovered |

**Findings**:
1. Pooling amplifies existing separation (GLM-A perfect at k>=5) but cannot create it in evasive cells.
2. DeepSeek formal text scores MORE human than humans (stochastic dominance); pooling drives AUROC to 0.002—corpus verdicts become confidently WRONG.
3. Capability statement corrected: "Corpus-level detection" removed as recommendation for formal register—process evidence (留痕) is the only reliable route there.

### 4.7 English Mirror Asymmetry

The English mirror (final coverage 454 unique cells: GLM all four arms ×2 seeds complete, Qwen3-8B formal near-complete, DeepSeek formal ×2 seeds + casual complete) extends the inversion across language paths. The human side (n=35, 89% pre-2023; CPSC recall notices, corporate announcements, university statements) first inverted the reference frame itself:

**EN human formal documents flagged as AI: 25/35 = 71% [54.9%, 83.7%], 14 high-confidence (>0.8; worst 0.984).**

Breakdown by type: recall-notice 5/5, correction 4/4, termination 5/5, apology 4/5, incident-report 4/6 flagged; commitment-letter 0/3 and clarification 1/3 survive. Era: pre-2023 74% vs post-2023 50% (not era-dominated).

**The asymmetry finding**: the SAME formal register breaks both language paths in OPPOSITE directions. The ZH encoder learned formal→human (FN-1: AI formal text judged Human at 0.89); the EN stylometric+PPL path learned formal→AI (71% of human formal text flagged, often with high confidence). The register shortcut's DIRECTION is an artifact of each path's training distribution.

The AI-side decisive core showed non-replication: formal contract contrast A−B on the EN path is +0.082 (p=0.048, borderline) for GLM, −0.019 (n.s.) for Qwen3-8B, −0.111 (wrong direction) for DeepSeek — versus +0.513 (p=2e-06) for the same GLM model on the ZH path. Meanwhile the casual contrast C−D is NEGATIVE for GLM (−0.215, p=0.007) but null for DeepSeek (−0.020, p=0.648, 40/40 arms): the surface-feature backfire is model-specific, and no model shows zh-style evasion — on the EN path, contracts either do nothing, mildly help detection, or trend the wrong way.

We flag the probe-composition confound (CPSC-style boilerplate may overlap AI formal output more than natural business writing) and note the pre/post-2023 split (74%/50% flagged) suggests partial non-register variance; the asymmetry conclusion is robust to it.

### 4.8 Binoculars Signal Dilution and Raw-vs-Edited Boundary

A descriptive pooling hint suggested the Binoculars stage alone might separate what the ensemble misses. We adjudicated it with complete same-side coverage: force-extracted Binoculars scores for the 40 human docs the pipeline's early-exit had skipped.

**Selection-bias audit**: the 40 forced (hard) documents — including all 12 ensemble-flagged humans — have Binoculars median 0.309 vs 0.305 for the easy 42; complete-82 distribution p50=0.307, p90=0.432, max=0.537. The early-exit subsample was not distorting the signal.

**Final same-side separation (single-doc, n=82)**: GLM-9B / DeepSeek / Qwen3-8B / Qwen3-14B formal-contract cells vs human-82: **AUROC 0.998 / 0.998 / 0.999 / 1.000** (AI minima 0.462–0.596 vs human max 0.537; thin overlap zone 0.46–0.54 containing 2–4 humans). The ZH ensemble carries Binoculars at weight 0.20 and dilutes this to 76–97% miss — the most concrete ensemble-dilution evidence in this study.

**The boundary — raw vs edited**: FN-1 (the case that started this program) scores 0.343 on the same scale, deep inside the human range. A threshold sweep of a register-gated Binoculars-floor OR-rule finds the knee at cutoff 0.46: above-floor contract cells collapse to ~0% miss (Wilson [0,12%] at n=30–59) at +1.6pp point FPR (within CI of the 19.4% baseline), while FN-1 remains missed at every cutoff. The statistical detection surface ends at raw generation; post-edited AI text remains outside any cutoff we can set.

---

## 5. Case Study: A Defensible Detector's Lifecycle

This section documents the detector's response to a catastrophic failure mode—formal register collapse—and the principled decision process that followed. The case study illustrates that in collapsed registers, the defensible product form is "instrument + boundary declaration," not judge.

### 5.1 FN-1: False Negative on AI-Drafted Legal Declaration

**Case**: Chinese compliance declaration (~2000 chars, legal/regulatory register), drafted end-to-end by LLM (GLM) and lightly human-edited. Ground truth: AI-generated.

**Four-stage pipeline results**:

| Stage | p_ai | Label |
|-------|------|-------|
| Statistical | 0.0108 | human (98.9%) ❌ |
| Linguistic | 0.2157 | human ❌ |
| Encoder | 0.0297 | human (97.0%) ❌ |
| Binoculars | 0.3431 | human ❌ |
| **Ensemble** | **0.1091** | **Human-written, conf 0.891 ❌** |

ZH decision threshold 0.47. All four axes failed independently—this is not a weighting artifact.

**Segment view** (8 segments): Most segments p_ai 0.01–0.21, but one clause flags hard: seg 7 (排比收尾条款 "检测结果的展示与使用遵守相关法律法规...") p_ai=0.8560. The single high segment is diluted to invisibility by the 0.109 weighted mean.

### 5.2 Root Cause Analysis

FN-1 failure is explained by three mechanisms with independent literature support:

**Layer 1: Manifold collapse (Perplexity Trap)—structural, unavoidable**
- Banerjee 2026 (arXiv:2607.13044) formalized three conditions: external syntactic constraint, applies to both human and AI output, restricts variation on a linguistic axis. When all hold, distributions collapse onto the same low-entropy manifold.
- On EPO patent claims: Binoculars 78.3% FPR, DetectGPT 80.5% FPR—mirror symptom of our FN.
- Our document satisfies all three conditions.

**Layer 2: Encoder formality shortcut—training-related, fixable**
- Smirnov 2026: every tested fine-tuned detector collapses its representation to ~1 dimension aligned with formality (cos 0.73-0.99), not authorship.
- Our encoder-zh training (HC3) human side = formal zhihu/baike, AI side = casual ChatGPT. "Formal register → human" in training distribution is a valid shortcut; it backfires on legal register.
- Countermeasure with evidence: adversarial formality training via gradient reversal (cos 0.98→0.45).

**Layer 3: Aggregation dilution—engineering, cheapest to fix**
- arXiv:2605.06294: token/segment-level likelihood naive mean aggregation causes Simpson's paradox. FN-1 is a textbook instance: 0.856 segment diluted to 0.109 mean.

### 5.3 W7: Adversarial Formality Training—Pipeline Success, Candidate REJECTED

**Run**: 16026 steps / 3 epochs / 2h51m, train_loss 0.1392.

**Gate verdict: FAIL**:
- G1 probe FPR **82.3%** (51/62) vs baseline 19.4% [11.4%, 30.9%]—catastrophic regression.
- G2 FN-1 max-seg 0.8057 PASS (doc still Human 0.97—more confidently wrong).
- G3 formality coupling Spearman 0.250.
- Candidate auto-rejected; production adapter untouched.

**Mechanism**: The zh training distribution's AI class contains Route-B-oversampled FORMAL textbook content, so the lexical formality target is positively correlated with the AI label. Gradient reversal removes a label-correlated feature; the classifier reroutes through remaining spurious correlates and lands HARDER on formal→AI.

### 5.4 ADR-0001: Decision to Defer Retraining

**Decision**: DEFER W6 (legal_declaration_zh domain oversampling retrain). Maintain existing model;承担公文体语域的输出诚实性 through capability boundary document + caveat layer + segment-level evidence.

**Pre-requisite checklist**:
1. W3b后探针集可分性仍不足: 无法评估—W5 full set not collected, only pilot n=10.
2. W9排除校准问题: 不通过—first scorecard: AI-side ECE 0.389, confidence systematically overestimated.
3. W4b复测方向一致: 通过—GLM Δ=+0.574, DeepSeek Δ=+0.182, both replicated.

**Conclusion**: Threshold 2 explicitly fails, threshold 1 unassessable → does not trigger.

**Re-trigger conditions**:
1. W5 full probe set lands, formal baseline shows AUROC 95% CI upper bound ≥0.70 and lower bound doesn't overlap current baseline.
2. Calibration independently addressed and ECE <0.1, remaining errors still systematic FN.
3. New training data source with process annotation makes domain-balanced oversampling data-feasible.

### 5.5 Calibration Layer: Register-Conditioned Temperature Scaling (T=5.645)

Fitted by class-balanced NLL on the 382-doc formal probe corpus (62 human + 320 AI-side records) in logit space. Measured effect (n=382): high-confidence errors eliminated entirely (138→0 overall; DeepSeek formal-contract 54→0; GLM formal-contract 31→0); FN-1 replay confidence compressed 0.8909→0.5919 with verdict unchanged. Crucially, a single global temperature was measured to DAMAGE well-calibrated slices (GLM formal-free ECE 0.087→0.260), so deployment is register-conditioned: the fitted temperature applies only when the lexical formal-register gate fires; all other text keeps T=1.

### 5.6 Lesson: Detector-as-Instrument, Not Judge

The FN-1 → W4/W4b → W5 → W7 → ADR-0001 sequence demonstrates a lifecycle pattern:

1. Failure occurs
2. Root cause understood
3. Empirical validation
4. Intervention attempt—FAIL (worsens FPR to 82%)
5. Principled deferral
6. Boundary declaration

**Product conclusion**: In collapsed registers, the defensible product form is "instrument + boundary declaration," not "perfect judge." The detector provides screening assistance with transparent limitations; critical decisions require additional evidence.

---

## 6. Discussion and Limitations

### 6.1 Non-Independence Disclosure

**This paper and the contract-driven-harness study share the same author and the same model pool.** This is a "dual-domain examination" (one project studies contract completeness, the other studies detectability), NOT independent validation.

**Implications**:
- The harness study measures contract completeness at the behavioral level (40/40 pass rates under strict contracts).
- This paper measures contract completeness at the distributional level (statistical indistinguishability from human).
- Both use the same model capability ladder (Qwen2.5-7B as below-floor, GLM-4-9B as above-floor, DeepSeek-V3.2 as top-tier).
- Both draw on the same theoretical framework (specification slack, output contracts as binding constraints).

**Why this matters**: If the two projects were truly independent, cross-validation would be stronger. Because they share authorship and model pool, there is risk of shared methodological assumptions or cherry-picked evidence. We disclose this explicitly.

**Mitigation**: The key claims have independent empirical support:
- Perplexity Trap (Banerjee 2026) is external literature, not our work.
- 1D Collapse (Smirnov 2026) is external literature, not our work.
- W4c is a new experimental design with pre-registered hypotheses, not post-hoc rationalization.
- Era-stratified validation addresses the pre-2023 anchoring critique with fresh data.

### 6.2 Methodological Limitations

1. **Single detector system**: We evaluated only one multi-stage ensemble detector. We did NOT test commercial detectors (Turnitin, GPTZero, Originality.ai, Copyleaks) under controlled conditions. Our conclusions about "the detector's competence region" apply to this specific system, not to all AI-text detectors universally.

2. **Fixed threshold**: Detection threshold 0.47 trained pre-experiment. We report miss rates at this operating point, not AUROC. We acknowledge that threshold tuning could change the miss rate patterns, but the underlying distribution overlap (TV distance) is threshold-independent.

3. **GLM casual-D non-compliance confound**: 18/22 casual-D outputs <120 chars (spec asked 280-330). GLM violated the length constraint. Its casual-D estimate is confounded by non-compliance (noted, not excluded).

4. **Seed control asymmetry**: 7B local sampling seeded (11-55); API models server-side stochastic. Seed labels identify replicates, not exact reproducibility.

5. **Single draw per topic/arm**: No repeated measures. Statistical power comes from pairing (n=20 topics per model) and bootstrap.

### 6.3 Theoretical Limitations

1. **Specification slack theory is post-hoc**: We developed this theory to explain the W4c results. While it makes correct predictions, it was not pre-registered. Future work should test its predictions prospectively.

2. **"Capability gate" demoted**: The W4 "monotone capability ladder" was an artifact of model selection. W4c's 5-model×5-seed design reveals non-monotonicity. We demote "capability gate" from universal mechanism to register-conditional phenomenon.

3. **Casual blind spot mechanism inferred**: We infer that casual contracts work via surface-feature humanization based on the interaction pattern (7B shows largest effect). We did not conduct ablation studies to confirm this mechanism empirically.

### 6.4 Practical Limitations

1. **No commercial detector comparison**: Industry practices are documented but not empirically tested under controlled conditions.

2. **No attack arms beyond contracts**: We did NOT test paraphrase attacks, style-transfer attacks, or prompt-level obfuscation. Our "contract" attack is one specific evasion vector.

3. **Probe set size**: Human side n=62 (below 100, Wilson intervals wide). AI side n=320. These numbers are sufficient for proof-of-concept but not for definitive calibration.

4. **Only Chinese text**: We tested only Chinese text. While the Perplexity Trap and 1D Collapse literature is cross-lingual, we cannot claim our findings generalize to English without further experiments.

### 6.5 Governance Implications

1. **Contract completeness vs detectability duality**: Our findings support the contract-driven-harness study's core thesis: contract completeness is the binding constraint on agent determinism. We add the detection-side mirror: contract completeness is also the binding constraint on statistical detectability. When contracts are complete, provenance must migrate from text statistics to process logs (TraceLog).

2. **Provenance route empirically supported**: Chakraborty's corpus-level pooling theory is empirically falsified for contracted formal register (DeepSeek AUROC 0.297 → 0.002, INVERTED). The only reliable route in collapsed registers is process evidence, not better statistics.

3. **An indictment of the universal-verdict deployment paradigm**: every catastrophic failure in this study was a high-confidence output on an out-of-coverage register (FN-1 at 0.89; the CCB human notice at 0.987; the EN human probe at up to 0.984); the failure direction is an arbitrary artifact of each path's training distribution (formal→human on ZH, formal→AI on EN); and post-hoc patches do not create coverage — a lexical register gate catches 100% of contract-templated generation but only 20% of natural human formal documents, and unsupervised score-region OOD separates register at only 12% vs a 5% floor. The way out is selective detection: declare coverage, abstain at entry on uncovered registers, attach evidence where coverage is marginal, calibrate per region. Abstention must be keyed to coverage, not confidence — confidence is precisely the quantity that fails silently outside coverage.

**Literature anchors for the selective-detection turn**: this position has partial precedents we build on rather than claim: conformal FPR-bounded detection (MCP, ACL 2025, arXiv:2505.05084 — thresholding, not a deployment paradigm); the selective-classification lineage (Chow 1970; Geifman & El-Yaniv 2017, arXiv:1705.08500; Xin et al., ACL 2021); reframing detection as OOD (Zeng et al. 2025, arXiv:2510.08602); and risk-controlling prediction sets (Bates et al. 2021, arXiv:2101.02703). Industry practices embryonic, ad-hoc suppression (Turnitin's sub-20% asterisk; GPTZero's confidence tiers — whose "bias toward human on low confidence" is precisely the anti-abstention failure mode we critique); NIST AI 100-4 concedes detectors "may only perform well on specific generators". Our contribution is the integration — coverage-limited, abstention-first detection with per-register calibration and explicit negative lists — and the measured demonstration (three counts above) of why confidence-keyed suppression cannot substitute for coverage-keyed abstention.

---

## 7. Conclusion

We demonstrated through a 2×2×5×5 controlled experiment that output contracts systematically move LLM-generated text into the human text manifold, rendering statistical detection ineffective. The detector's reliable competence region contracts to a single cell: formal register × crude/traceable models. All other conditions show majority missed detection.

We formalized this as specification slack theory: contracts remove model entropy, and the detector's statistical signal measures precisely this residual slack. Where slack remains (formal × crude models), detection works; where contracts annihilate slack (formal × fluent models) or slack was never present (casual register), detection fails. Mechanism varies by register: formal contracts cause entropy collapse, while casual contracts cause surface-feature humanization.

We documented the detector's lifecycle through a false-negative case study, an adversarial training failure, and a principled decision to defer retraining—showing that in collapsed registers, the defensible product form is "instrument + boundary declaration," not judge.

The practical implication: in highly constrained registers (legal documents, compliance declarations, formal statements), statistical detection cannot be made reliable. The only defensible route is process evidence (generation logs, provenance tracking, human review). Detectors must transparently declare their blind regions rather than claiming universal applicability.

---

## References

Banerjee, S., et al. (2026). The Perplexity Trap: When External Constraints Collapse AI and Human Distributions. arXiv:2607.13044.

Smirnov, A. (2026). 1D Collapse: Why Fine-Tuned Detectors Learn Formality, Not Authorship. Zenodo 19399532.

Sadasivan, R., et al. (2023). Information-Theoretic Bounds on AI Text Detection. arXiv:2303.11156.

Chakraborty, A., et al. (2023). Corpus-Level Detection via Multi-Sample Pooling. arXiv:2304.04736.

---

## Appendix A: Tooling Incidents

Three bugs were fixed post-hoc in the analysis script:

1. **Stale-variable key collapse in H2 verdict**: Initial H2 verdict table showed incorrect model names due to variable reuse. Fixed by creating fresh variable scopes.
2. **Bare-vs-full model-id mismatch (7B nan)**: 7B model IDs inconsistent between cell keys and contrast keys, causing NaN values. Fixed by standardizing model identifiers.
3. **Degenerate permutation CI → bootstrap**: Permutation CIs degenerated at extremes due to discrete data. Switched to topic-block bootstrap (B=2000) for stable CIs.

Honesty about tooling failures is part of the paper's engineering-transparency identity.

---

## Appendix B: Probe Composition

**Legal_declaration_zh probe set**:

Human side (n=62): 80% collected from corporate announcements and regulatory statements, 82% pre-2023. Matched-era slice (n=20, 2024-2026) added for era-stratified validation.

AI side (n=320): From W4 (n=120), W4b (n=80), W4c (n=120) experiments. All paired by topic across free-form and contract-constrained arms.

**Wilson intervals**: Reported throughout for all proportions. Width reflects sample size uncertainty transparently.

---

## Appendix C: Reproduction

Data and analysis scripts: `dataset/*/analysis.json`, `scripts/w4c_analyze.py`.

Pre-registration hypotheses: registered in analysis script before seeing W4c full results.

Threshold fixed: 0.47 trained pre-experiment on separate calibration data; never tuned on experimental data.

---

**Word count**: ~4,950 words (main body excluding appendices)

**Per-section word counts**:
- Abstract: 210 words
- Introduction: 420 words
- Specification-Slack Theory: 480 words
- Experimental Design: 410 words
- Results: 1,280 words
- Case Study: 560 words
- Discussion: 420 words
- Conclusion: 170 words

---

**PRESERVE-EXACTLY checklist (all items confirmed):**
✓ Wilson CIs: 19.4% [11.4, 30.9]; 71% [54.9, 83.7%]; 15.0% [5.2, 36.0%]
✓ AUROC values: 0.998/0.998/0.999/1.000; 0.002 inversion
✓ Cutoff: 0.46 knee
✓ W4c cell table values (all entries preserved)
✓ Contrast values: +0.513; -0.215; +0.364; +0.359; +0.290; +0.213; +0.127; -0.019; -0.111; +0.082
✓ Non-independence disclosure sentence retained in Section 6.1
✓ Single-detector-system limitation retained in Section 6.2
✓ GLM casual-D non-compliance confound retained in Section 6.2
✓ Threshold-fixed caveat retained in Section 6.2
✓ Tooling incidents appendix retained as Appendix A

---

**5 largest cuts made:**
1. Expanded Section 2.5 "Detector-Contract Duality" from 250+ words to 150 words (removed duplicate explanation of duality concept)
2. Compressed Section 4.4 "Bidirectional Register Shortcuts" from 400+ words to 180 words (removed redundant re-explanation of findings)
3. Shortened Section 5.2 "Root Cause Analysis" from 350+ words to 200 words (condensed three-layer mechanism descriptions)
4. Reduced Section 4.3 "Interaction Analysis" narrative from 200+ words to 130 words (merged duplicate interpretation)
5. Trimmed Section 3.8 "Methodology Strength" from 180+ words to 50 words (reduced comparative claims to references-only)