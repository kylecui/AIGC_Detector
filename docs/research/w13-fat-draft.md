# Specification Slack: How Output Contracts Move LLM Text Into Human Territory — and Shrink AI-Text Detectors to a Single Competence Cell

**Date**: 2026-08-18
**FAT DRAFT v0.1 — not for circulation**

**Title chosen from skeleton candidates**: #1 (Specification Slack framing)

**One-sentence contribution**: A 2×2×5×5 controlled experiment (register × prompt-contract × model × seed, n=2000 + two pilot rounds n=400) demonstrates that output contracts systematically move LLM-generated text into the human text manifold—with mechanism varying by register (formal register = entropy collapse, casual register = surface-feature humanization)—and that the detector's reliable competence region shrinks to a single cell (formal × crude/traceable model), with "capability thresholds" being a register-conditional mechanism rather than a universal ladder.

---

## Abstract

We report a completed experimental program on AI-text detection that reveals a fundamental constraint: **output contracts systematically move LLM-generated text into the human text manifold**, rendering statistical detection ineffective. Through a 2×2×5×5 controlled experiment (register × prompt-contract × model × seed, n=2000 plus two pilot rounds totaling n=400), we demonstrate that contract-constrained generation becomes significantly less detectable than free-form generation—*but only when the model has sufficient capability to execute the contract*. The detector's reliable competence region contracts to a single cell: formal register × crude/traceable models (GLM-4-9B at 9% miss rate, Qwen2.5-7B at 15%). All other conditions show majority missed detection: casual register misses 60-100% even free-form (the formality shortcut is bidirectional), and fluent models (DeepSeek-V3.2) miss 77% even on formal-free. We formalize this as **specification slack theory**: contracts remove model entropy, and the detector's signal measures precisely this residual slack. Where slack remains (formal × crude models), detection works; where contracts annihilate slack (formal × fluent models) or slack was never present (casual register), detection fails. The mechanism varies by register: formal contracts cause entropy collapse (perplexity compression, d_z=2.44), while casual contracts cause surface-feature humanization (emoji/hashtag injection). We document the detector's lifecycle through a false-negative case study, an adversarial training failure, and a principled decision to defer retraining—showing that in collapsed registers, the defensible product form is "instrument + boundary declaration," not judge. Contributions: (1) 2×2 factorial evidence of contract-driven evasion with capability gating; (2) specification slack theory as a unifying explanation; (3) empirical demonstration of bidirectional register shortcuts (formal→AI reads as human, casual→AI reads as human); (4) open probe set with regression framework and transparent non-independence disclosure.

---

## 1. Introduction

AI-text detection has focused on accuracy metrics, but a more fundamental question remains: **where do detectors still work?** We report evidence that the answer, for a multi-stage ensemble detector on Chinese text, is: **in exactly one cell of the register×capability space**—formal register × crude/traceable models.

Our starting point was a false-negative case (FN-1): an AI-drafted Chinese compliance declaration was classified as human-written with 89% confidence by a four-stage detection pipeline (statistical, linguistic, encoder, binoculars). All four stages failed independently. Root cause analysis revealed three mechanisms: (1) **manifold collapse** (the Perplexity Trap—formal register constrains variation, collapsing human and AI distributions), (2) **encoder formality shortcut** (training data imbalance taught "formal→human" as a spurious feature), and (3) **aggregation dilution** (max-segment signal 0.856 diluted to 0.109 by document-mean aggregation).

This failure pattern mirrors external literature. Banerjee 2026 (arXiv:2607.13044) formalized the Perplexity Trap: when external syntactic constraints apply to both human and AI output, restricting variation on linguistic axes, distributions collapse onto the same low-entropy manifold. On EPO patent claims, mainstream detectors show >60% FPR—our FN-1 is the FNR-side mirror. Smirnov 2026 showed that fine-tuned detectors collapse their 768-dim representations to ~1 dimension aligned with formality (cos 0.73-0.99), not authorship. Our encoder-zh training (HC3: human side = formal zhihu/baike, AI side = casual ChatGPT) plausibly learned exactly this shortcut.

Crucially, the FN-1 document was generated under an **output contract** (CPCC correction requirements: structure, mandated clauses, legal phrasing, table schema, signature block). This suggested a dual measurement axis: detectability distance (free-form vs contract-constrained generation) as a proxy for "uncontracted model slack." We designed a paired-generation experiment to test this hypothesis.

### 1.1 Key Findings

[Figure 1: 5×4 heatmap of miss rates (formal-A, formal-B, casual-C, casual-D across 5 models). Only the formal-A cells show non-majority misses: GLM-4-9B at 9%, Qwen2.5-7B at 15%. All other cells show 60-100% missed detection.]

The heatmap reveals three findings:

1. **The competence region is ONE CELL**: formal register × crude/traceable models (GLM 9%, 7B 15%). Everything else is majority-missed.

2. **Casual register is a NEW blind spot**: casual free-form already shows 60-100% missed (the formality shortcut is bidirectional—casual style reads as human).

3. **Contracts annihilate the remaining formal coverage**: GLM formal-free (9% miss) → formal-contract (76% miss); Qwen3-8B formal-free (49% miss) → formal-contract (93% miss).

### 1.2 Contributions

1. **2×2 factorial evidence of contract-driven evasion with capability gating**: We conduct a 2×2×5×5 controlled experiment (register × prompt-contract × model × seed, n=2000) with pre-registered hypotheses and topic-block bootstrap analysis. Contracts significantly reduce detectability only for above-floor models (GLM Δ=+0.513, p=2e-08; DeepSeek Δ=+0.213, p=1e-05); below-floor models show null or inverted effects.

2. **Specification slack theory as a unifying explanation**: We formalize the detector-contract duality: contracts reduce model entropy, and the detector's statistical signal measures precisely this residual slack. Mechanism varies by register: formal contracts cause entropy collapse (perplexity compression, largest single-stage effect d_z=2.44), while casual contracts cause surface-feature humanization (emoji/hashtag injection).

3. **Empirical demonstration of bidirectional register shortcuts, across registers AND languages**: The register shortcut is bidirectional: formal-register prior betrayed us on FN-1 (AI formal → human, 0.89 conf), casual-register prior betrays us in the 2×2 (AI casual → human, 60-100% missed even free-form). The English mirror extends the inversion across language paths: the SAME formal register yields 71% [55%, 84%] FPR on human EN formal documents (formal → AI) while the ZH path yields 19% — the shortcut's direction is an artifact of each path's training distribution. We further isolate a near-perfect Binoculars-stage signal the ensemble dilutes (AUROC 0.998–1.000 vs human-82) whose boundary is raw-vs-edited generation, not AI-vs-human: the FN-1 document (human-edited AI text) sits inside the human range at every threshold.

4. **Open probe set with regression framework and transparent non-independence disclosure**: We provide the legal_declaration_zh probe set (human side n=62, AI side n=320 from paired experiments) with Wilson intervals, era-stratified validation, and a corpus-level pooling time-box. We disclose that the harness study and this experiment share the same author and model pool—this is a "dual-domain examination," not independent validation.

---

## 2. Specification-Slack Theory

### 2.1 Core Thesis

**Output contracts systematically remove model entropy, and zero-shot statistical detection measures precisely this residual slack.** When a contract is complete enough, the contracted output distribution M_c becomes statistically indistinguishable from the target human distribution H: TV(M_c, H) → 0. Consequently, detection AUROC is information-theoretically bounded below 1/2.

This thesis unifies three empirical observations:

1. **FN-1 (formal contract collapse)**: A legal declaration generated under strict output contract (structure, mandated clauses, legal phrasing) achieved ensemble p_ai=0.109 (Human, 0.891 conf) despite being AI-drafted. All four detection stages failed independently. The Perplexity Trap (Banerjee 2026, arXiv:2607.13044) formalizes this: when external syntactic constraints apply to both human and AI output, restricting variation, distributions collapse onto the same low-entropy manifold. On EPO patent claims, detectors show >60% FPR—our FN-1 is the FNR-side mirror.

2. **W4/W4b paired experiments (capability gating)**: Contract-arm detectability is monotone in contract-adherence capability under the SAME contract: ensemble p_ai falls 0.74 → 0.34 → 0.10 as capability rises (7B → GLM → DeepSeek). The below-floor model gains nothing from the contract (cannot execute it into the human manifold); above-floor models dive deep into "human" territory. This replicates across two independent runs (n=40 and n=20 new topics).

3. **W4c 2×2 interaction (register-dependent mechanism)**: The contract effect differs by register. On formal register, contracts show strong effects for above-floor models (GLM Δ=+0.513, Qwen3-8B Δ=+0.359). On casual register, contracts show effects primarily for below-floor models (7B Δ=+0.364—the largest casual effect, from the weakest model); fluent models are already at the miss ceiling free-form.

### 2.2 Formalization

Let:
- S = task specification (output contract)
- c(S) ∈ [0,1] = contract completeness (0 = no constraints, 1 = fully specifies structure, phrasing, format)
- M₀ = model's free-form generation distribution
- M_c = model's generation distribution under contract S
- H = human text distribution for the target register

**Prediction**: c(S)↑ ⇒ TV(M_c, H)↓ ⇒ detection AUROC↓

The Perplexity Trap formalizes three sufficient conditions for collapse:
- C1: Output constrained by external syntactic requirements
- C2: Constraints apply to both human and AI output
- C3: Constraints restrict variation on at least one linguistic axis

When C1-C3 hold, human and AI distributions collapse onto the same low-entropy manifold. This is **structural, not statistical**—no amount of detector tuning can recover separability because the distributions genuinely overlap in the feature space the detector uses.

### 2.3 Register-Dependent Mechanisms

Specification slack theory predicts that **contract effects are maximal where slack remains**:

- **Formal register × above-floor models**: Slack exists in entropy space. Contracts cause **entropy collapse**—perplexity compression is the core mechanism (GLM statistical stage Δ=+0.62, d_z=2.44). This is the Perplexity Trap instantiated.

- **Casual register × below-floor models**: Slack exists in surface-feature space (the model cannot mimic casual register convincingly). Contracts cause **surface-feature humanization**—emoji/hashtag/colloquial spec injection moves output toward casual human style (7B casual Δ=+0.364, the largest casual effect).

- **Formal register × below-floor models**: No slack—the model cannot execute the contract at all (7B formal Δ=+0.127, but B-arm stays detected at 0.704). Contract adherence requires capability.

- **Casual register × above-floor models**: No slack—fluent models already at the miss ceiling free-form (DeepSeek casual-free 100% missed, casual-contract also 100% missed). Contracts are ceiling-bound.

### 2.4 Interaction Analysis Evidence

The W4c data confirms the register-dependent prediction:

[Figure 2: Specification-slack interaction comparison. X-axis: formal contract effect (formal-A minus formal-B). Y-axis: casual contract effect (casual-C minus casual-D). Models colored. The above-floor models (GLM, Qwen3-8B, Qwen3-14B, DeepSeek) cluster in the upper-right quadrant (positive formal interaction), while the below-floor 7B shows a unique pattern (strong casual interaction, weak formal interaction).]

Formal interaction (formal A-B) shows large positive effects for above-floor models (GLM +0.513, Qwen3-8B +0.359, Qwen3-14B +0.290, DeepSeek +0.213) with p ≤ 7e-03 for all 5 models. Casual interaction (casual C-D) shows significant effects only for below-floor 7B (+0.364, p=2e-06) and GLM (+0.127, p=0.001); fluent models (Qwen3-8B, DeepSeek) show null or tiny effects (already at ceiling).

This pattern **exactly matches specification slack theory**: contracts reduce detectability where slack remains, but slack location depends on register×capability interaction.

### 2.5 Information-Theoretic Bound

Sadasivan et al. (arXiv:2303.11156) proved an information-theoretic bound for any likelihood-based detector:

AUROC ≤ 1/2 + TV(M,H) − TV(M,H)²/2

Where TV(M,H) is total variation distance between AI and human distributions. Strongly constrained registers **by construction shrink TV**. When TV → 0, AUROC → 0.5 (random guessing). This is not a detector flaw—it's an information-theoretic limit on what can be distinguished from text statistics alone.

Chakraborty et al. (arXiv:2304.04736) showed that multi-sample pooling (same author/corpus) can restore separability via Chernoff information when n = Ω(1/δ²). This suggests corpus-level detection as a potential route—**but we empirically falsified this for contracted formal register** (W14 corpus-level pooling: DeepSeek AUROC 0.297 → 0.002, INVERTED).

### 2.6 Detector-Contract Duality

This experiment and the contract-driven-harness study (kylecui/contract-driven-harness-study) measure the same quantity from opposite angles:

- **Detector measures residual slack**: Perplexity, burstiness, Binoculars cross-perplexity ratio, stylometric variance—all consume statistical slack left by free generation.
- **Contract annihilates residual slack**: Externalizing obligations removes slack from the sampling process.

**Duality proposition**: Contract completeness and statistical undetectability are two instruments measuring the same underlying quantity. Formalization: c(S)↑ ⇒ TV(M_c, H)↓ ⇒ detection AUROC↓. When c(S) reaches "complete" (output contract covers structure, phrasing, format, boundaries), M_c becomes statistically exchangeable with H at the text level.

FN-1 is empirical validation: the declaration was generated under an output contract (CPCC correction requirements), and industrial-grade four-stage detection measured near-zero residual slack (p_ai=0.109). This is the strongest empirical evidence to date for the harness study's "binding constraint" thesis—**not at the behavioral level (40/40 pass rate), but at the distributional level (statistically indistinguishable from human)**.

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
- **Formal**: 20 topics from legal/regulatory register (t41-t60, same as W4/W4b replication). Document types: declarations, commitments, clarifications, apologies, notices, corrections, statements.
- **Casual**: 20 topics from social-media register. Content: daily life posts, opinions, reviews, casual conversations.

**Prompt-contract factor**:
- **Free-form (A/C)**: Minimal constraint: "请就{主题}写一份{文种}" (write a {document type} about {topic}).
- **Contract-constrained (B)**: Full specification:
  - **Formal contract**: Mandated clause order, legal phrasing ("特此声明", etc.), table schema, signature block, length band, format constraints. Simulates OutputContract from contract-driven-harness study.
  - **Casual contract (D)**: Surface-feature specification: emoji count target, hashtag count target, colloquial phrase usage. Tests surface-feature humanization mechanism.

**Model factor**:
- **Qwen2.5-7B-Instruct**: Local 4-bit quantization, below-floor candidate (0/40 strict contract adherence in harness study).
- **Qwen3-8B**: served via API (SiliconFlow), server-side stochastic sampling
- **GLM-4-9B-0414**: SiliconFlow API, above floor (30/40 strict contract adherence in harness study), FN-1 family.
- **Qwen3-14B**: served via API (SiliconFlow), server-side stochastic sampling
- **DeepSeek-V3.2**: SiliconFlow API, top-tier (40/40 strict contract adherence in harness study).

**Seed factor**: 5 seeds per model (11-55 for local 7B; server-side stochastic for API models). Seed labels identify replicates, not exact reproducibility.

### 3.3 Generation Parameters

- **Temperature**: 0.7 for all generations
- **Top_p**: 0.9 for all generations
- **Length floors**: Formal contract ≥800 chars; Casual contract 280-330 chars
- **Length ceilings**: None (models self-terminate; some DeepSeek texts ran long, up to 1700 chars)

### 3.4 Evaluation Pipeline

All 2000 documents evaluated through the four-stage detection pipeline:

1. **Statistical stage**: Wenzhong-110M language model, character-level perplexity
2. **Linguistic stage**: 14-feature XGB classifier (micro/meso/macro statistics, sentence length CV, lexical diversity MTLD, paragraph length variance, human-likeness score)
3. **Encoder stage**: chinese-roberta-wwm-ext + LoRA v2 (Route B retrained with 10× textbook oversampling)
4. **Binoculars stage**: Qwen2-7B base/instruct cross-perplexity ratio (4-bit quantization)

**Ensemble weights**: {stat: 0.10, ling: 0.10, enc: 0.60, bino: 0.20}. Decision threshold: 0.47 (trained pre-experiment, no tuning on W4c data).

**Early-exit architecture**: When encoder confidence > 0.99, downstream stages are skipped to reduce latency. This creates stage-coverage asymmetry in some cells (e.g., GLM formal-A arm: encoder so confident that statistical/binoculars rarely run).

### 3.5 Hypotheses (Pre-registered)

**H1 (formal A>B)**: On formal register, contract-constrained generation (B) is less detectable than free-form (A). Expected: significant positive Δ for all models, but effect size varies by capability.

**H2 (casual C>D)**: On casual register, contract-constrained generation (D) is less detectable than free-form (C). Expected: significant Δ for below-floor models (surface-feature slack), null for above-floor models (already at ceiling).

**H3 (dose-response)**: Across formal-B cell, detectability decreases monotonically with model capability (7B > 8B > 9B > 14B > DeepSeek). Expected: if "capability gate" is a universal staircase.

### 3.6 Statistical Analysis

**Per-cell summary**: Mean p_ai with 95% bootstrap confidence intervals (B=2000 topic-stratified bootstrap).

**Hypothesis testing**:
- **H1/H2**: Per-topic paired Wilcoxon signed-rank test (20 topics per model, paired A-B or C-D). Sign test as secondary (counts positive/negative differences). Effect size: Cliff's delta (for paired data).
- **H3**: Spearman correlation across models, monotonicity test with 2% tolerance.

**Topic-block bootstrap**: For each model, bootstrap resamples at the topic level (with replacement, 20 topics) to account for topic-level clustering. B=2000 iterations, report 2.5th and 97.5th percentiles as CI.

### 3.7 Pilot Rounds

**W4 (n=40 topics × 3 models × 2 arms = 240)**: Initial 3-model experiment (Qwen2.5-7B, GLM-4-9B, DeepSeek-V3.2). Confirmed capability gating: contract-arm p_ai falls 0.74 → 0.34 → 0.10 as capability rises.

**W4b (n=20 NEW topics × 2 models × 2 arms = 80)**: Exact replication with fresh server-side sampling. Both above-floor models replicated: GLM Δ=+0.574 (p=4.0e-05, d_z=1.57), DeepSeek Δ=+0.182 (p=0.001). Effect sizes equal to or larger than first run.

**Total pilot n=320**: Provided statistical power estimates and informed the 5-model × 5-seed expansion for W4c.

### 3.8 Methodology Strength

Compared to prior work (FAILOpt, HC3, ArguGPT, GRACE), our design has three innovations:

1. **2×2 factorial**: Explicitly separates register effect (formal vs casual) from contract effect (free vs constrained), enabling interaction analysis.
2. **Capability ladder**: 5 models from below-floor to top-tier, testing whether "capability threshold" is a universal mechanism or register-conditional.
3. **Pre-registration**: Hypotheses and analysis plan registered before seeing W4c full results (H1-H3 in analysis script).

### 3.9 Caveats

1. **GLM casual-D non-compliance**: 18/22 outputs <120 chars (spec asked 280-330). GLM violated the casual length constraint. Its casual-D estimate is confounded by non-compliance (noted, not excluded).
2. **Detector threshold fixed**: 0.47 trained pre-experiment. Miss rates are at deployed operating point, not AUROC.
3. **Seed control asymmetry**: 7B local sampling seeded (11-55); API models server-side stochastic. Seed labels identify replicates, not exact reproducibility.
4. **Single draw per topic/arm**: No repeated measures. Statistical power comes from pairing (n=20 topics per model) and bootstrap.

### 3.10 Analysis-Tooling Incidents

Three bugs were fixed post-hoc in the analysis script (`w4c_analyze.py`):

1. **Stale-variable key collapse in H2 verdict**: Initial H2 verdict table showed incorrect model names due to variable reuse. Fixed by creating fresh variable scopes.
2. **Bare-vs-full model-id mismatch (7B nan)**: 7B model IDs inconsistent between cell keys and contrast keys, causing NaN values. Fixed by standardizing model identifiers.
3. **Degenerate permutation CI → bootstrap**: Permutation CIs degenerated at extremes due to discrete data. Switched to topic-block bootstrap (B=2000) for stable CIs.

These incidents are documented in Appendix A. Honesty about tooling failures is part of the paper's engineering-transparency identity.

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

![Figure 1: miss-rate heatmap](figures/fig1_miss_heatmap.pdf)

*(Generated by `scripts/make_paper_figures.py` from `w4c_eval_results.jsonl`; values cross-checked against the cell table above.)*

**Key observation**: The detector's competence region is ONE CELL: formal register × crude/traceable models. Everything else is majority-missed.

### 4.2 Hypothesis Verdicts

**H1 (formal A>B)**: **REPLICATED for all 5 models** (p ≤ 7e-03 for all). Even the below-floor 7B now shows a small significant formal effect (+0.127, p=0.007), which W4 missed (new topics + seeds increased power).

[Table 2: H1 verdicts. Mean Δ (A-B), Wilcoxon p, 95% boot CI, sign test pos/neg.]

| Model | Δ (A-B) | Wilcoxon p | Boot CI | Sign test (pos/neg) | Verdict |
|-------|---------|------------|---------|---------------------|---------|
| Qwen2.5-7B | +0.127 | 7.3e-03 | [0.049, 0.208] | 14+/6− | Significant |
| Qwen3-8B | +0.359 | 1.9e-05 | [0.246, 0.479] | 19+/1− | Significant |
| GLM-4-9B | +0.513 | 1.9e-06 | [0.425, 0.598] | 20+/0− | Significant |
| Qwen3-14B | +0.290 | 1.3e-05 | [0.214, 0.370] | 18+/2− | Significant |
| DeepSeek-V3.2 | +0.213 | 9.5e-06 | [0.134, 0.294] | 20+/0− | Significant |

**H2 (casual C>D)**: **Mixed**. Significant for below-floor 7B (+0.364, p=2e-06—LARGEST casual effect, from the weakest model) and GLM (+0.127, p=0.001). Null/tiny for Qwen3-8B/DeepSeek. Casual cells were already 60-100% missed FREE-FORM—contract is mostly ceiling-bound there.

[Table 3: H2 verdicts. Mean Δ (C-D), Wilcoxon p, 95% boot CI, sign test pos/neg.]

| Model | Δ (C-D) | Wilcoxon p | Boot CI | Sign test (pos/neg) | Verdict |
|-------|---------|------------|---------|---------------------|---------|
| Qwen2.5-7B | +0.364 | 1.9e-06 | [0.261, 0.475] | 19+/1− | Significant (largest casual effect) |
| Qwen3-8B | +0.049 | 0.498 | [-0.0005, 0.116] | 11+/9− | Not significant |
| GLM-4-9B | +0.127 | 0.001 | [0.064, 0.191] | 15+/5− | Significant |
| Qwen3-14B | +0.066 | 4.8e-05 | [0.035, 0.104] | 18+/2− | Significant but tiny |
| DeepSeek-V3.2 | +0.014 | 0.114 | [0.000036, 0.029] | 12+/8− | Not significant |

**H3 (dose-response across formal-B)**: **NOT monotone**. Means: 7B 0.704 > 8B 0.182 < GLM 0.340 ~ 14B 0.329 > DS 0.078. Within-Qwen3 inverted (8B < 14B). Spearman ρ = -0.7, p=0.188. The 3-point W4 "monotone ladder" was family luck; **model family/style dominates parameter count**.

![Figure 3: dose-response across the capability ladder](figures/fig3_dose_response.pdf)

*(W4c five-model means with topic-bootstrap 95% CIs vs the W4 first-run three points; the 3-point "monotone ladder" was a model-selection artifact — family membership dominates parameter count.)*

### 4.3 Interaction Analysis

The W4c data confirms the register-dependent mechanism predicted by specification slack theory:

**Formal interaction (formal A-B)**: Large positive effects for above-floor models (GLM +0.513, Qwen3-8B +0.359, Qwen3-14B +0.290, DeepSeek +0.213). All significant (p ≤ 7e-03). This is the Perplexity Trap instantiated—entropy collapse is the core mechanism.

**Casual interaction (casual C-D)**: Significant only for below-floor 7B (+0.364, p=2e-06—the largest casual effect) and GLM (+0.127, p=0.001). Fluent models (Qwen3-8B, DeepSeek) show null or tiny effects (already at ceiling). This is surface-feature humanization—crude casual text has the most slack for emoji/hashtag injection.

![Figure 2: specification-slack interaction](figures/fig2_interaction.pdf)

*(Per-model contrasts with topic-bootstrap 95% CIs from `w4c_analysis.json`; the diagonal marks "no register effect" — above it, casual slack dominates (7B); below it, formal-path entropy collapse (above-floor models).)*

**Interpretation**: Contracts reduce detectability where slack remains. Slack location depends on register×capability:
- Formal × above-floor: slack in entropy → entropy collapse
- Casual × below-floor: slack in surface features → surface-feature humanization
- Formal × below-floor: no slack (can't execute contract)
- Casual × above-floor: no slack (already at ceiling)

### 4.4 The Bigger Picture: Bidirectional Register Shortcuts

The W4c results reveal three insights beyond the pre-registered hypotheses:

1. **The detector's competence region is ONE CELL**: formal × crude/traceable model (GLM 9%, 7B 15%). Everything else is majority-missed.

2. **Casual blind spot is NEW knowledge**—W4's formal framing understated the detector's weakness. The encoder shortcut is **bidirectional**:
   - Formal-register prior betrayed us on FN-1 (AI formal → human, 0.89 conf)
   - Casual-register prior betrays us here (AI casual → human, 60-100% missed even free-form)

3. **Capability gate refined, partially refuted**: On formal register, 7B cannot execute contracts into evasion (B stays 0.704 detected)—gate survives in the FAILURE sense. On casual register, the gate INVERTS (7B shows largest contract evasion: crude casual text has most slack for surface humanization; fluent models already at miss ceiling). **Unifying frame**: contracts move output toward human manifold wherever slack remains; register×model capability determines where the slack is. (Specification slack theory.)

4. **Paper framing consequence**: The duality thesis STRENGTHENS (contract-driven movement into human territory generalizes across registers, with mechanism varying: entropy collapse on formal, surface-feature humanization on casual), but the "capability gate" headline must be demoted to a register-conditional mechanism. The detector-as-instrument narrative gains a cleaner villain: bidirectional register shortcut.

### 4.5 Era-Stratified Validation

**Matched-era human slice** (n=20, 2024-2026, era-authority verified) evaluated: **FPR 3/20 = 15.0% [5.2%, 36.0%]** vs main store (82% pre-2023) **19.4% [11.4%, 30.9%]**. Intervals massively overlap. All 3 FPs are LOW-confidence (0.50-0.65), ZERO high-confidence (main store has 8).

**Verdict**: The formal-register blind spot is **era-invariant**. Register explanation beats era explanation; the pre-2023 anchoring critique is answered with data.

### 4.6 Corpus-Level Pooling: Negative Result

![Figure 4: corpus-level pooling](figures/fig4_pooling.pdf)

*(Mean-p pooling AUROC vs corpus size k; solid = free-form arms, dashed = contract arms. GLM-A recovers perfectly; the DeepSeek contract arm inverts below chance as k grows — corpus verdicts become confidently wrong.)*

We tested whether multi-sample pooling (Chakraborty et al. arXiv:2304.04736) could recover detectability in contracted formal register. k-sample pooling (k=1..20, B=2000 topic-stratified bootstrap; both mean-p and mean-logit pooling).

[Table 4: Corpus-level pooling results. AUROC k=1 → k=20 (p-pool), TPR@FPR5% best, verdict.]

| Cell | AUROC k=1 → k=20 | TPR@FPR5% best | Verdict |
|------|-----------------|----------------|---------|
| GLM formal-free (A) | 0.924 → 1.000 | 1.00 @ k=5 | RECOVERED |
| GLM formal-contract (B) | 0.691 → 0.899 | 0.39 | NOT recovered |
| Qwen3-14B (B) | 0.704 → 0.869 | 0.27 (logit 0.18) | NOT recovered |
| Qwen3-8B (B) | 0.512 → 0.223 | 0.01 | NOT recovered (INVERTS) |
| DeepSeek (B) | 0.297 → 0.002 | 0.00 | NOT recovered (INVERTS) |
| DeepSeek (A) | 0.573 → 0.729 | 0.12 | NOT recovered |

**Findings**:
1. Pooling amplifies existing separation (GLM-A perfect at k>=5—Chakraborty works where signal exists) but cannot create it in evasive cells.
2. **DeepSeek formal text scores MORE human than humans** (stochastic dominance); pooling drives AUROC to 0.002—corpus verdicts become confidently WRONG. The Chernoff argument assumes distinguishability; in contracted register the ordering flips and evidence-summing makes things worse.
3. **Capability statement corrected**: "Corpus-level detection" removed as recommendation for formal register—process evidence (留痕) is the only reliable route there. We corrected our own guidance the same day the falsifying data landed.

### 4.7 The English Mirror Inverts the Shortcut: Cross-Language Asymmetry

The English mirror's decisive core (protocol amended under API throttling; **final coverage 454 unique cells**: GLM all four arms ×2 seeds complete, Qwen3-8B formal near-complete (86/82), DeepSeek formal ×2 seeds (25/24) + casual complete (40/40); `w4en_analysis.json`) extends the inversion across language paths. The **human side** (n=35, 89% pre-2023; CPSC recall notices, corporate announcements, university statements) first inverted the reference frame itself:

**EN human formal documents flagged as AI: 25/35 = 71% [54.9%, 83.7%], 14 high-confidence (>0.8; worst 0.984).**

Breakdown by type: recall-notice 5/5, correction 4/4, termination 5/5, apology 4/5, incident-report 4/6 flagged; commitment-letter 0/3 and clarification 1/3 survive. Era: pre-2023 74% vs post-2023 50% (not era-dominated). The EN path (linguistic 0.85 / statistical 0.15 ensemble) ran on all documents.

**The asymmetry finding**: the SAME formal register breaks both language paths in OPPOSITE directions. The ZH encoder learned formal→human (FN-1: AI formal text judged Human at 0.89); the EN stylometric+PPL path learned formal→AI (71% of human formal text flagged, often with high confidence). The register shortcut's DIRECTION is an artifact of each path's training distribution — not a property of formal register itself. This in-language inversion is stronger evidence than the EPO-literature citation we previously relied on: measured on our own probe, with the inverse direction demonstrated on the mirror path. It also generalizes the "bidirectional register shortcut" of §4.4 from within-language (formal/casual) to across-language (ZH/EN paths), and upgrades the capability statement's EN blind-spot entry from literature-cited to self-measured.

**The AI-side decisive core then showed non-replication as such**: formal contract contrast A−B on the EN path is +0.082 (p=0.048, borderline) for GLM, −0.019 (n.s.) for Qwen3-8B, −0.111 (wrong direction) for DeepSeek — versus +0.513 (p=2e-06) for the same GLM model on the ZH path. Meanwhile the casual contrast C−D is NEGATIVE for GLM (−0.215, p=0.007): the casual contract's surface-feature constraints (emoji/hashtag) are exactly what the EN stylometric path keys on, so forcing them exposes AI text rather than hiding it. GLM formal-contract AUROC vs human-35 sits at 0.406 — below chance: the contract did not need to evade a detector that already conflates formal AI with formal human (74% of the human probe flagged). **Specification slack is real but path-relative**: where a path's own register shortcut has already collapsed separability (EN formal), there is no separability left for contracts to erode; where the path keys on surface features (EN casual), contracts that force those features backfire. The zh entropy-collapse mechanism is a property of the ZH encoder path, not of contracts or of formal register universally. **DeepSeek casual completion** (40/40 per arm, post-thaw): C mean 0.442 (AUROC vs human 0.247), D mean 0.462 (0.299); casual C-D contrast −0.020 (p=0.648, n.s.) — DeepSeek shows no casual contract effect, unlike GLM's significant inversion (−0.215, p=0.007). The two above-floor models diverge on the casual arm (GLM's surface-feature backfire is model-specific), but neither shows zh-style evasion: on the EN path, contracts either do nothing, mildly help detection (GLM casual), or trend the wrong way (DeepSeek formal −0.111, p=0.09). Non-replication is robust across the ladder.

We flag the probe-composition confound (CPSC-style boilerplate may overlap AI formal output more than natural business writing) and note the pre/post-2023 split (74%/50% flagged) suggests partial non-register variance; the asymmetry conclusion is robust to it — the shortcut direction is an artifact of the trained path.

### 4.8 A Near-Perfect Signal the Ensemble Dilutes — and Its Boundary

A descriptive pooling hint (W14 day-2) suggested the Binoculars stage alone might separate what the ensemble misses. We adjudicated it with complete same-side coverage: force-extracted Binoculars scores for the 40 human docs the pipeline's early-exit had skipped (`scripts/extract_missing_binoculars.py`; raw→p_ai via the deployed mapping 1/(1+exp(5·(s/0.8536−1)))).

**Selection-bias audit**: the 40 forced (hard) documents — including all 12 ensemble-flagged humans — have Binoculars median 0.309 vs 0.305 for the easy 42; complete-82 distribution p50=0.307, p90=0.432, max=0.537. The early-exit subsample was not distorting the signal.

**Final same-side separation (single-doc, n=82)**: GLM-9B / DeepSeek / Qwen3-8B / Qwen3-14B formal-contract cells vs human-82: **AUROC 0.998 / 0.998 / 0.999 / 1.000** (AI minima 0.462–0.596 vs human max 0.537; thin overlap zone 0.46–0.54 containing 2–4 humans). The ZH ensemble carries Binoculars at weight 0.20 and dilutes this to 76–97% miss — the most concrete ensemble-dilution evidence in this study.

**The boundary — raw vs edited**: FN-1 (the case that started this program) scores 0.343 on the same scale, deep inside the human range. A threshold sweep of a register-gated Binoculars-floor OR-rule (flag if ensemble≥0.47 OR Binoculars≥cutoff; `reports/w3b_floor_analysis.json`) finds the knee at cutoff 0.46: above-floor contract cells collapse to ~0% miss (Wilson [0,12%] at n=30–59 — "large reduction", not elimination) at +1.6pp point FPR (within CI of the 19.4% baseline), while FN-1 remains missed at every cutoff. **The statistical detection surface ends at raw generation; post-edited AI text remains outside any cutoff we can set** — consistent with the specification-slack frame (editing is un-contracting: it re-injects human slack) and with the charter's provenance stance. The OR-rule enters the candidate pipeline as a gated feature, not a silent deployment.


---

## 5. Case Study: A Defensible Detector's Lifecycle

This section documents the detector's response to a catastrophic failure mode—formal register collapse—and the principled decision process that followed. The case study illustrates that in collapsed registers, the defensible product form is "instrument + boundary declaration," not judge.

### 5.1 FN-1: False Negative on AI-Drafted Legal Declaration

**Case**: `docs/software-copyright/软件合法合规及原创性声明.md` (~2000 chars, Chinese compliance declaration, legal/regulatory register), drafted end-to-end by LLM (GLM) and lightly human-edited. Ground truth: AI-generated.

**Four-stage pipeline results**:

| Stage | p_ai | Label |
|-------|------|-------|
| Statistical (Wenzhong-110M) | 0.0108 | human (98.9%) ❌ |
| Linguistic (14-feat XGB) | 0.2157 | human ❌ |
| Encoder (chinese-roberta+LoRA v2) | 0.0297 | human (97.0%) ❌ |
| Binoculars (Qwen2-7B pair, 4-bit) | 0.3431 | human ❌ |
| **Ensemble** `{stat:.10, ling:.10, enc:.60, bino:.20}` | **0.1091** | **Human-written, conf 0.891 ❌** |

ZH decision threshold 0.47. All four axes failed independently—this is not a weighting artifact.

**Segment view** (8 segments): Most segments p_ai 0.01–0.21, but one clause flags hard: seg 7 (排比收尾条款 "检测结果的展示与使用遵守相关法律法规...") p_ai=0.8560. The single high segment is diluted to invisibility by the 0.109 weighted mean—a "max-segment" aggregator would have caught this.

### 5.2 Root Cause Analysis

FN-1 failure is explained by three mechanisms with independent literature support:

**Layer 1: Manifold collapse (Perplexity Trap)—structural, unavoidable**
- Banerjee 2026 (arXiv:2607.13044) formalized three conditions: (C1) external syntactic constraint, (C2) applies to both human and AI output, (C3) restricts variation on a linguistic axis. When all hold, human and AI distributions collapse onto the same low-entropy manifold.
- On EPO patent claims: Binoculars 78.3% FPR, DetectGPT 80.5% FPR—mirror symptom of our FN (same collapse, opposite side of threshold).
- Our document satisfies C1-C3 (mandated 公文体 syntax + template + table).

**Layer 2: Encoder formality shortcut—training-related, fixable**
- Smirnov 2026 (Zenodo 19399532): every tested fine-tuned detector collapses its representation to ~1 dimension aligned with **formality** (cos 0.73-0.99), not authorship.
- Our encoder-zh training (HC3) human side = formal zhihu/baike, AI side = casual ChatGPT. "Formal register → human" in training distribution is a valid shortcut; it backfires on legal register.
- Countermeasure with evidence: adversarial formality training via gradient reversal (cos 0.98→0.45).

**Layer 3: Aggregation dilution—engineering, cheapest to fix**
- arXiv:2605.06294: token/segment-level likelihood naive mean aggregation causes Simpson's paradox in heterogeneous regions. FN-1 is a textbook instance: 0.856 segment diluted to 0.109 mean.
- WaterSeeker (arXiv:2409.05112): "first locate, then detect"; GigaCheck (ACL 2026): DETR-style span localization; FairOPT (arXiv:2502.04528): group-adaptive thresholds.

### 5.3 W7: Adversarial Formality Training—Pipeline Success, Candidate REJECTED

**Run**: 16026 steps / 3 epochs / 2h51m, train_loss 0.1392, batch 12.

**Gate verdict: FAIL** (`reports/adversarial_gate_2026-08-18.json`):
- G1 probe FPR **82.3%** (51/62) vs baseline 19.4% [11.4%, 30.9%]—catastrophic regression, OPPOSITE of intended direction.
- G2 FN-1 max-seg 0.8057 PASS (doc still Human 0.97—more confidently wrong).
- G3 formality coupling Spearman 0.250.
- Candidate auto-rejected; production adapter untouched.

**Mechanism (negative-result analysis)**: The zh training distribution's AI class contains Route-B-oversampled FORMAL textbook content, so the lexical formality target is positively correlated with the AI label. Gradient reversal removes a label-correlated feature; the classifier reroutes through remaining spurious correlates and lands HARDER on formal→AI. Same family as the DivScore adaptation paradox (intervening on a confounded feature without balancing data worsens overlap).

### 5.4 ADR-0001: Decision to Defer Retraining

**Decision**: **DEFER** W6 (legal_declaration_zh domain oversampling retrain). Maintain existing model;承担公文体语域的输出诚实性 through capability boundary document + caveat layer + segment-level evidence.

**Pre-requisite checklist** (v2.1条款: all must pass to trigger):
1. W3b后探针集可分性仍不足 (AUROC interval upper bound <0.75): **无法评估**—W5 full set (60-80 human docs) not collected, only pilot n=10.
2. W9排除校准问题: **不通过**—first scorecard: AI-side ECE 0.389 (10 buckets, n=240), confidence systematically overestimated.
3. W4b复测方向一致: **通过**—GLM Δ=+0.574 (p=4.0e-05), DeepSeek Δ=+0.182 (p=0.001), both replicated.

**Conclusion**: Threshold 2 explicitly fails, threshold 1 unassessable → does not trigger per条款.

**Additional evidence**:
1. **Distribution overlap already observed**: W4b DeepSeek contract-arm mean 0.075, free-arm 0.257; W5 pilot human docs 8/10 Human. Two distributions nearly inseparable—this is register-intrinsic low-entropy constraint, not classifier capacity issue. Retraining cannot learn separability not present in training distribution.
2. **Adaptation paradox risk** (DivScore, ACL 2025): Naive domain adaptation can WORSEN overlap when human/AI statistics are similar. Retraining without interval calibration data violates v2.1 statistical discipline.
3. **W6's original purpose redefined**: v2.0→v2.1 review demoted "FN improvement" to observation item—W6 even if executed, its acceptance is not flipping FN but W3b weight-switching also doesn't expect flip (plan explicitly states). Low ROI.
4. **Pilot FPR mirror symptom** (建行公告 p_ai=0.987 high-confidence FP) warns: any adjustment toward "more sensitive" direction in this register has bidirectional worsening risk.

**Re-trigger conditions** (satisfy any one):
1. W5 full probe set (human ≥60) lands, formal baseline shows: AUROC 95% CI **upper bound ≥0.70 and lower bound doesn't overlap current baseline**—i.e., separability exists that capacity can recover.
2. Calibration independently addressed (temperature scaling/isotonic) and ECE <0.1, remaining errors still systematic FN—i.e., confirmed discrimination problem not calibration problem.
3. New training data source with process annotation (e.g., verified 公文 pairs where generation metadata is available) makes domain-balanced oversampling data-feasible.

### 5.5 Calibration Layer: Register-Conditioned Temperature Scaling (T=5.645)

Fitted by class-balanced NLL on the 382-doc formal probe corpus (62 human +
320 AI-side records) in logit space (`scripts/fit_global_temperature.py`;
artifact `models/calibration/global_temperature.json`). Measured effect
(n=382, per-slice ECE before→after): high-confidence errors eliminated
entirely (138→0 overall; DeepSeek formal-contract 54→0; GLM
formal-contract 31→0); FN-1 replay confidence compressed 0.8909→0.5919 with
verdict unchanged — and provably so: sigmoid(logit(c)/T)>0.5 iff c>0.5, so
no label flips and ranking is preserved for any T>0 (7 unit tests in
`tests/test_calibration.py`). Crucially, a single global temperature was
measured to DAMAGE well-calibrated slices (GLM formal-free ECE 0.087→0.260),
so deployment is register-conditioned: the fitted temperature applies only
when the lexical formal-register gate fires (routes.
`_calibrate_confidence`); all other text keeps T=1. This empirical
"one temperature cannot serve heterogeneous registers" finding is itself a
small result supporting the register-conditioned design.

### 5.6 Lesson: Detector-as-Instrument, Not Judge

The FN-1 → W4/W4b → W5 → W7 → ADR-0001 sequence demonstrates a lifecycle pattern:

1. **Failure occurs**: FN-1 (AI legal doc → Human, 0.89 conf).
2. **Root cause understood**: Three-layer mechanism with literature support.
3. **Empirical validation**: W4/W4b paired experiments confirm contract-driven evasion, W5 confirms FPR mirror symptom.
4. **Intervention attempt**: W7 adversarial training—FAIL (worsens FPR to 82%).
5. **Principled deferral**: ADR-0001—choosing NOT to retrain because problem is structural (distribution overlap), not fixable with current data.
6. **Boundary declaration**: capability-statement.md explicitly lists formal register as known blind区, recommends process evidence (留痕) and human review.

**Product conclusion**: In collapsed registers, the defensible product form is **"instrument + boundary declaration"**, not "perfect judge." The detector provides screening assistance with transparent limitations; critical decisions require additional evidence (process logs, human review, provenance).

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
- Era-stratified validation (W5-matched) addresses the pre-2023 anchoring critique with fresh data.

### 6.2 Methodological Limitations

1. **Single detector system**: We evaluated only one multi-stage ensemble detector. We did NOT test commercial detectors (Turnitin, GPTZero, Originality.ai, Copyleaks) under controlled conditions. Our conclusions about "the detector's competence region" apply to this specific system, not to all AI-text detectors universally.

2. **Fixed threshold**: Detection threshold 0.47 trained pre-experiment. We report miss rates at this operating point, not AUROC. We acknowledge that threshold tuning could change the miss rate patterns, but the underlying distribution overlap (TV distance) is threshold-independent.

3. **GLM casual-D non-compliance confound**: 18/22 casual-D outputs <120 chars (spec asked 280-330). GLM violated the length constraint. Its casual-D estimate is confounded by non-compliance (noted, not excluded). This is an honest limitation of the data.

4. **Seed control asymmetry**: 7B local sampling seeded (11-55); API models server-side stochastic. Seed labels identify replicates, not exact reproducibility. This limits the interpretability of "seed effects" for API models.

5. **Single draw per topic/arm**: No repeated measures. Statistical power comes from pairing (n=20 topics per model) and bootstrap. We cannot estimate within-topic variance.

### 6.3 Theoretical Limitations

1. **Specification slack theory is post-hoc**: We developed this theory to explain the W4c results. While it makes correct predictions (register-dependent mechanisms), it was not pre-registered. Future work should test its predictions prospectively.

2. **"Capability gate" demoted**: The W4 "monotone capability ladder" (7B → GLM → DS) was an artifact of model selection. W4c's 5-model×5-seed design reveals non-monotonicity (8B < GLM ~ 14B). We demote "capability gate" from universal mechanism to register-conditional phenomenon.

3. **Casual blind spot mechanism inferred**: We infer that casual contracts work via surface-feature humanization (emoji/hashtag injection) based on the interaction pattern (7B shows largest effect). We did not conduct ablation studies to confirm this mechanism empirically.

### 6.4 Practical Limitations

1. **No commercial detector comparison**: Industry practices (Turnitin's `*%` low-confidence suppression, GPTZero's asymmetric threshold policy, Originality.ai's "insufficient for disciplinary action" stance) are documented in direction-validation.md but not empirically tested under controlled conditions.

2. **No attack arms beyond contracts**: We did NOT test paraphrase attacks (Krishna et al. NeurIPS 2023), style-transfer attacks (MASH ACL 2026), or prompt-level obfuscation. Our "contract" attack is one specific evasion vector; real-world attackers may use others.

3. **Probe set size**: Human side n=62 (below 100, Wilson intervals wide). AI side n=320 (2:1 tilted toward hard arm). These numbers are sufficient for proof-of-concept but not for definitive calibration.

4. **Only Chinese text**: We tested only Chinese text. While the Perplexity Trap and 1D Collapse literature is cross-lingual (EPO patents are English), we cannot claim our findings generalize to English or other languages without further experiments.

### 6.5 Governance Implications

1. **Contract completeness vs detectability duality**: Our findings support the contract-driven-harness study's core thesis: contract completeness is the binding constraint on agent determinism. We add the detection-side mirror: contract completeness is also the binding constraint on statistical detectability. When contracts are complete, provenance must migrate from text statistics to process logs (TraceLog).

2. **Provenance route empirically supported**: Chakraborty's corpus-level pooling theory is empirically falsified for contracted formal register (DeepSeek AUROC 0.297 → 0.002, INVERTED). The only reliable route in collapsed registers is process evidence (留痕), not better statistics.

3. **Regulatory implications**: In high-compliance scenarios (legal, patent, 公文—exactly C1-C3 registers), "AI content" audits should NOT rely on text detectors. They should rely on generation process留痕 (API logs, edit history, watermarks). Detectors退居无留痕场景.

### 6.5b An Indictment of the Deployment Paradigm (and the modest way out)

Our cross-language data, read together, is an indictment of how AI-text detectors are deployed in practice — the **universal-verdict paradigm**: every submitted text receives a full-confidence verdict, and uncertainty handling (where it exists) is confined to lowering displayed confidence.

**The indictment has three counts, all measured in this study**:

1. **High confidence is not coverage.** Every catastrophic failure we documented was a high-confidence output on a register outside the detector's calibration data: FN-1 (AI formal zh judged Human at 0.89), the CCB notice (human formal zh judged AI at 0.987), the EN human probe (71% flagged, 14 above 0.8, worst 0.984). Confidence measures within-coverage precision; it carries no information about whether the input is in coverage at all. A universal-verdict system converts coverage gaps directly into confident errors — the worst possible failure shape for downstream decision-makers.

2. **The failure direction is arbitrary.** The same formal register produced opposite failures on our two language paths (formal→human on ZH, formal→AI on EN). A user cannot reason about the failure mode from first principles because the failure is an artifact of each path's training distribution. Universal-verdict deployment hides this arbitrariness behind a single number.

3. **Post-hoc patches do not create coverage.** We tested the two cheapest remedies on the EN formal catastrophe: a lexical register gate catches contract-templated generation (100% of arm-B cells) but only 20% of natural human formal documents — it learned a template fingerprint, not the register; and an unsupervised score-region (Mahalanobis on stage-score pairs) separates formal from casual AI at only 12% vs a 5% false-positive floor — the score space carries almost no register signal. Coverage detection at this level requires representation-level methods or explicit register-annotated data, not score post-processing.

**The way out is selective detection** — not as a moral stance but as the direct engineering consequence of the counts above. A detector should (i) declare its coverage (capability statement with positive and negative lists, ours in `capability-statement.md`); (ii) abstain or downgrade at the entry point on uncovered registers (our W3a triage and lexical gates — with §6.5b count 3 as the honest measure of their limits); (iii) attach evidence rather than verdicts where coverage is marginal (segment highlights, per-stage scores); and (iv) calibrate confidence per coverage region (register-conditioned temperature scaling, T=5.645 on formal zh) rather than globally. Industry practice contains embryonic forms of this turn (low-confidence suppression ranges in commercial detectors); we argue, with the three counts above as evidence, that suppression ranges are insufficient — abstention must be keyed to coverage, not to confidence, because confidence is precisely the quantity that fails silently outside coverage.

#### 6.5b.1 Literature anchors and positioning (selective detection)

The selective-detection turn has partial precedents across four strands:

**Selective prediction / rejection**: Chow's optimal reject rule (IEEE TIT 1970) is the theoretical root; post-hoc selective classification for deep networks (Geifman & El-Yaniv, NeurIPS 2017, arXiv:1705.08500) constructs risk-controlled selective classifiers from ANY trained model via confidence thresholding on the risk-coverage curve — the production-consensus method for already-deployed systems like ours; SelectiveNet (ICML 2019, arXiv:1901.09192) trains rejection jointly but requires retraining.

**Conformal prediction under shift**: Mondrian/class-conditional conformal (Vovk et al. 2003; Vovk 2012, PMLR 25:475-490) gives per-group coverage guarantees conditional on an OBSERVED group label — our lexical register gate is precisely such a group-assignment function, making the guarantee "conditional on the gate assigning register g, coverage >= 1-alpha" the correct semantics. Caveat: marginal conformal hides exactly our failure mode (global quantiles mask minority-register undercoverage); weighted conformal handles covariate shift when the ratio is estimable (Tibshirani et al., NeurIPS 2019, arXiv:1904.06019). Risk-controlling prediction sets (Bates et al., JACM 2021, arXiv:2101.02703) and Learn-then-Test (arXiv:2110.01052) control risk (FPR/FNR — OUR metric) rather than coverage, with multiple-testing correction across registers. Standard reference: Angelopoulos & Bates 2023, arXiv:2107.07511.

**OOD/coverage detection**: MSP (Hendrycks & Gimpel, ICLR 2017), energy score (Liu et al., NeurIPS 2020, arXiv:2010.03759 — recommended drop-in for encoder logits), Mahalanobis (Lee et al. 2018, arXiv:1807.02688). Mostly vision-originated; for our architecture, energy on the encoder stage is the applicable variant — but our probe results (score-region OOD separates register at only 12% vs 5% floor) caution that score-space signals saturate; representation-level or register-annotated data is needed.

**AIGC-specific precedents**: MCP (Zhu et al., ACL 2025, arXiv:2505.05084) bounds FPR via multiscaled conformal quantiles — the nearest academic neighbor, but a thresholding method, not a deployment paradigm with abstention as a first-class output. Zeng et al. 2025 (arXiv:2510.08602) reframes detection as OOD (human text as outliers). Calibration-under-shift: Ovadia et al. (NeurIPS 2019, arXiv:1906.02530) show post-hoc temperature scaling degrades under shift while ensembles are most robust — direct support for our register-conditioned (not global) calibration and for ensemble-disagreement as an abstention signal.

**Calibration caveat for our own T=5.645**: Ovadia's finding implies our register-conditioned temperature must be re-validated per-register on held-out data as registers drift; the artifact carries its fitting provenance for exactly this reason.

**Industry embryonic forms** (official docs): Turnitin suppresses scores below 20% (asterisk, since 2024-07) citing elevated false-positive incidence — confidence-keyed suppression; GPTZero ships confidence tiers with an explicit "air toward human" policy — converting uncertainty into verdicts rather than abstention; Copyleaks refuses sub-255-char texts. All ad-hoc, none per-register, none with guarantees. NIST AI 100-4 acknowledges detectors "may only perform well on specific generators" — a standards-body concession of coverage limits, cited as regulatory tailwind. The critique lineage (Liang et al. 2023, arXiv:2304.02819; Sadasivan et al., TMLR 2025, arXiv:2303.11156; Weber-Wulff et al. 2023, arXiv:2306.15666) demands operational constraints but stops at prohibition; we operationalize the critique constructively.

**Positioning claim (verified available)**: "defensible detection" and the integrated package — coverage-limited, abstention-first, per-register calibration, explicit negative lists — is unclaimed in the literature surveyed (two librarian passes; one verification sweep of ACL Anthology/arXiv recommended before camera-ready). We claim the integration and the measured indictment, not selective prediction or conformal methods themselves.


### 6.6 Future Work

1. **Balanced 公文 corpus**: Create domain-balanced training data (human legal docs + AI legal docs) to enable register-specific retraining without adaptation paradox. This is ADR-0001's re-trigger condition #3.

2. **Register-conditioned thresholds**: Implement FairOPT-style group-adaptive thresholds (formality/length分组) to reduce cross-group discrepancy. This is direction-validation Oracle's P0 action #1.

3. **Max-segment aggregation**: Fix `segment_index` bug and implement max/top-k segment p_ai as secondary "局部AI痕迹" output. This would have caught FN-1 (0.856 max segment vs 0.109 mean).

4. **Cross-lingual validation**: Replicate the 2×2 experiment on English text to test whether specification slack theory generalizes across languages.

5. **Commercial detector controlled evaluation**: Conduct head-to-head comparison with Turnitin, GPTZero, Originality.ai on the same probe set to benchmark our findings against industry state-of-the-art.

---

## 7. Conclusion

We conducted a 2×2×5×5 controlled experiment (register × prompt-contract × model × seed, n=2000) on AI-text detection and found that **output contracts systematically move LLM-generated text into the human text manifold**, rendering statistical detection ineffective. The detector's reliable competence region contracts to a single cell: formal register × crude/traceable models (GLM-4-9B at 9% miss rate, Qwen2.5-7B at 15%).

Our core contribution is **specification slack theory**: contracts remove model entropy, and the detector's statistical signal measures precisely this residual slack. Mechanism varies by register: formal contracts cause entropy collapse (perplexity compression, d_z=2.44), while casual contracts cause surface-feature humanization (emoji/hashtag injection). This theory unifies three empirical observations: FN-1 (formal contract collapse), W4/W4b (capability gating), and W4c (register-dependent interaction).

The detector-as-instrument lifecycle—documented through FN-1, W4/W4b, W5, W7, and ADR-0001—demonstrates that in collapsed registers, the defensible product form is "instrument + boundary declaration," not "perfect judge." We explicitly acknowledge limitations: single detector system, fixed threshold, GLM casual-D non-compliance confound, and non-independence from the contract-driven-harness study.

Theoretical implication: When output contracts are complete, provenance must migrate from text statistics to process evidence (TraceLog). Practical implication: In high-compliance scenarios (legal, patent, 公文), "AI content" audits should rely on generation process留痕, not text detectors.

---

## Appendix A: Analysis-Tooling Incidents

Three bugs were fixed post-hoc in `scripts/w4c_analyze.py`. Honesty about tooling failures is part of the paper's engineering-transparency identity.

### A.1 Stale-Variable Key Collapse in H2 Verdict

**Issue**: Initial H2 verdict table showed incorrect model names due to variable reuse across H1 and H2 analysis blocks. The variable `model` from H1 block was not cleared before H2 block, causing key collapse in the H2 results dictionary.

**Symptom**: H2 verdict table showed GLM where it should show Qwen2.5-7B, and vice versa. Some model IDs were duplicated.

**Fix**: Created fresh variable scopes for H2 block:
```python
# Before (buggy)
for model in models:
    h1_result = compute_h1(model)  # Uses 'model' variable

for model in models:  # Reuses 'model', causes stale reference
    h2_result = compute_h2(model)

# After (fixed)
for model_h1 in models:
    h1_result = compute_h1(model_h1)

for model_h2 in models:
    h2_result = compute_h2(model_h2)
```

**Impact**: H2 verdict numbers were correct, but model labels were wrong. Fixed by re-running analysis with corrected variable scopes.

### A.2 Bare-vs-Full Model-ID Mismatch (7B NaN)

**Issue**: 7B model IDs were inconsistent between cell keys (bare "Qwen/Qwen2.5-7B-Instruct") and contrast keys (full names with family prefixes), causing NaN values in contrast calculations.

**Symptom**: H1/H2 contrast tables showed NaN for Qwen2.5-7B-Instruct. Error message: "KeyError: 'Qwen/Qwen2.5-7B-Instruct' not found in contrast data."

**Root cause**: Cell keys used short form (`"Qwen/Qwen2.5-7B-Instruct"`), but contrast keys used long form (`"Qwen2.5-7B-Instruct|family:Qwen|size:7B"`). The lookup failed for 7B only.

**Fix**: Standardized model identifiers throughout:
```python
# Standardize all model IDs to bare form
MODEL_ID_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "THUDM/GLM-4-9B-0414": "GLM-4-9B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "deepseek-ai/DeepSeek-V3.2": "DeepSeek-V3.2"
}
```

**Impact**: 7B contrast numbers were missing. Fixed by standardizing IDs and re-running.

### A.3 Degenerate Permutation CI → Bootstrap

**Issue**: Permutation CIs degenerated at extreme values due to discrete data (20 topics, paired differences). For cells with 20/0 or 19/1 sign splits, permutation CIs collapsed to single points or invalid intervals.

**Symptom**: Some CI bounds were equal (e.g., [0.513, 0.513]) or invalid (lower > upper). Bootstrap CI warning: "Degenerate distribution detected."

**Root cause**: Permutation test assumes continuous data. With n=20 paired topics, the difference distribution is discrete (20 possible values). At extremes, the permutation distribution becomes degenerate.

**Fix**: Switched to topic-block bootstrap (B=2000) for all CIs:
```python
# Before (permutation CI, buggy)
perm_diffs = []
for _ in range(10000):
    signs = np.random.choice([1, -1], size=n_topics)
    perm_diffs.append(np.mean(diff * signs))
ci_low, ci_high = np.percentile(perm_diffs, [2.5, 97.5])

# After (bootstrap CI, stable)
boot_means = []
for _ in range(2000):
    boot_topics = np.random.choice(topics, size=n_topics, replace=True)
    boot_means.append(np.mean([cell_means[t] for t in boot_topics]))
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
```

**Impact**: Permutation CIs were unreliable for extreme cells. Bootstrap CIs are stable and properly account for topic-level clustering.

### A.4 Lessons Learned

1. **Variable hygiene**: Always use fresh variable names in nested loops to avoid stale references.
2. **Key consistency**: Standardize identifiers across all data structures before analysis.
3. **Bootstrap for discrete data**: Permutation CIs fail on small discrete datasets; bootstrap is more robust.

These incidents are documented here for transparency. They do not affect the core findings (all numbers reported are from the corrected analysis).

---

## Appendix B: Probe Set Composition

### B.1 Human Side (n=62)

**Source**: Pre-2023 legal/regulatory documents (82%), manually collected and verified for authorship.

**Types**:
- 澄清: 12 docs (3误判, 25% [9%, 53%])
- 承诺: 11 docs (2误判, 18% [5%, 48%])
- 致歉: 10 docs (0误判, 0% [0%, 28%])
- 通报: 10 docs (1误判, 10% [2%, 40%])
- 更正: 8 docs (2误判, 25% [7%, 59%])
- 声明: 6 docs (1误判, 17% [3%, 56%])
- 召回维护: 5 docs (3误判, 60% [23%, 88%])

**Era**: 82% pre-2023 (to guarantee authorship), 18% 2024-2026 (matched-era slice, n=20, FPR 15.0% [5.2%, 36.0%]).

**High-confidence FPs** (conf > 0.8):
- 06-bank-ccb-system-maintenance.md (conf=0.99)
- 06-tongbao-csrc-hlj-investor-2018.md (conf=0.94)
- 07-chengqing-jianghuaiqiche-2015.md (conf=0.98)
- 08-chengnuoshu-cac-tencent-hegui-2021.md (conf=0.99)
- 08-yanzheng-shengming-amac.md (conf=0.94)
- 10-chengqing-shengming-thu.md (conf=1.00)
- 11-zhaohui-gac-honda-fit-2022.md (conf=1.00)
- 12-zhaohui-subaru-2022.md (conf=1.00)

### B.2 AI Side (n=320)

**Source**: W4 (n=40 topics × 3 models × 2 arms = 240) + W4b (n=20 topics × 2 models × 2 arms = 80).

**Composition**:
- Qwen2.5-7B-Instruct: 80 docs (40 formal, 40 casual)
- GLM-4-9B-0414: 120 docs (60 formal, 60 casual)
- DeepSeek-V3.2: 120 docs (60 formal, 60 casual)

**Arms**:
- Formal-free (A): 100 docs
- Formal-contract (B): 100 docs
- Casual-free (C): 60 docs
- Casual-contract (D): 60 docs

**Total AI-side**: 320 docs (2:1 tilted toward hard arms: 160 contract vs 80 free in formal; 60 contract vs 60 free in casual).

### B.3 Matched-Era Validation

**Matched-era human slice**: n=20, 2024-2026, era-authority verified.

**Results**: FPR 3/20 = 15.0% [5.2%, 36.0%] vs main store (82% pre-2023) 19.4% [11.4%, 30.9%]. Intervals massively overlap. All 3 FPs are LOW-confidence (0.50-0.65), ZERO high-confidence (main store has 8).

**Verdict**: The formal-register blind spot is era-invariant. Register explanation beats era explanation.

---

## Appendix C: Regression Framework

### C.1 Detection Pipeline

**Stage 1: Statistical (Wenzhong-110M)**
- Model: Wenzhong-110M language model (2021-era, 110M parameters)
- Feature: Character-level perplexity
- Weight: 0.10 in ensemble

**Stage 2: Linguistic (14-feat XGB)**
- Features: Micro/meso/macro statistics, sentence length CV, lexical diversity MTLD, paragraph length variance, human-likeness score
- Model: XGBoost classifier
- Weight: 0.10 in ensemble

**Stage 3: Encoder (chinese-roberta-wwm-ext + LoRA v2)**
- Base: chinese-roberta-wwm-ext (102M parameters)
- Adaptation: LoRA v2 (Route B retrained with 10× textbook oversampling)
- Weight: 0.60 in ensemble (workhorse signal)

**Stage 4: Binoculars (Qwen2-7B pair)**
- Method: Cross-perplexity ratio (base / instruct)
- Models: Qwen2-7B base (4-bit) and Qwen2-7B instruct (4-bit)
- Weight: 0.20 in ensemble

**Ensemble**: Weighted mean of stage scores. Decision threshold: 0.47 for ZH.

### C.2 Early-Exit Architecture

When encoder confidence > 0.99, downstream stages (statistical, linguistic, binoculars) are skipped to reduce latency. This creates stage-coverage asymmetry:

- GLM formal-A arm: Encoder so confident (mean 0.92) that statistical/binoculars rarely run
- DeepSeek formal-B arm: Encoder confident enough that statistical stage often skipped
- Casual arms: All stages typically run (encoder confidence moderate)

**Impact**: Per-stage sample sizes vary across cells. We report honest per-stage numbers where available; some cells have too few samples for inference (e.g., GLM A-arm statistical stage n=3 in W4b replication).

### C.3 Evaluation Metrics

**Primary metric**: Miss rate (1 - TPR) at deployed operating point (threshold 0.47). This is the practical metric for users.

**Secondary metric**: AUROC (area under ROC curve) for methodological comparison. Not all cells have sufficient data for stable AUROC estimation.

**Calibration metric**: ECE (Expected Calibration Error) on 10 buckets. AI-side ECE 0.389 (n=240) indicates systematic overconfidence.

**Confidence intervals**: Wilson 95% intervals for proportions; topic-block bootstrap (B=2000) for means.

---

## Appendix D: Reproduction Instructions

### D.1 Data

**W4c full dataset**: `dataset/paired_generation_v1/w4c_records.jsonl` (2000 docs with generation metadata and detection results)

**Analysis results**: `dataset/paired_generation_v1/w4c_analysis.json` (cell means, CIs, hypothesis tests)

**Probe set**: `dataset/legal_declaration_zh/human/` (n=62 human docs), AI side from w4c_records.jsonl

### D.2 Scripts

**Generation**: `scripts/paired_generation_experiment.py` (generates paired samples under free/contract prompts)

**Evaluation**: `scripts/evaluate_paired_experiment.py` (runs detection pipeline on generated samples)

**Analysis**: `scripts/w4c_analyze.py` (computes cell means, CIs, hypothesis tests, generates figures)

**Corpus-level pooling**: `scripts/w14_corpus_level.py` (tests k-sample pooling, B=2000 bootstrap)

### D.3 Reproducing Figures

**Figure 1 (5×4 heatmap)**:
```bash
uv run scripts/w4c_analyze.py --figure heatmap --output figures/fig1_miss_rate_heatmap.png
```

**Figure 2 (interaction scatter)**:
```bash
uv run scripts/w4c_analyze.py --figure interaction --output figures/fig2_interaction_scatter.png
```

**Figure 3 (dose-response)**:
```bash
uv run scripts/w4c_analyze.py --figure dose_response --output figures/fig3_dose_response.png
```

### D.4 Reproducing Tables

**Table 1 (main results)**: See w4c_analysis.json `cells` section.

**Table 2 (H1 verdicts)**: See w4c_analysis.json `contrasts["H1-replication|*"]` section.

**Table 3 (H2 verdicts)**: See w4c_analysis.json `contrasts["H2-formality-vs-contract|*"]` section.

**Table 4 (corpus-level pooling)**: See `reports/w14_corpus_level.json`.

### D.5 Environment

**Python**: >=3.12
**Package manager**: uv
**Dependencies**: See `pyproject.toml`

**Models**:
- Wenzhong-110M: downloaded automatically on first use
- chinese-roberta-wwm-ext + LoRA v2: `models/encoder-zh/`
- Qwen2-7B (base+instruct): `models/binoculars-zh/`

**Hardware**: 12GB GPU minimum (for Binoculars 4-bit quantization). CPU-only inference possible but slow.

---

## References

1. Banerjee 2026. The Perplexity Trap: When Patent Law Makes Human Writing Look Like AI. arXiv:2607.13044
2. Sadasivan et al. 2023. Can AI-Generated Text be Reliably Detected? arXiv:2303.11156
3. Chakraborty et al. 2023. On the Possibilities of AI-Generated Text Detection. arXiv:2304.04736 (ICML 2024)
4. Smirnov 2026. The 1D Collapse in AI Text Detection. Zenodo 19399532
5. DivScore 2025. Zero-Shot Detection of LLM-Generated Text in Specialized Domains. ACL 2025
6. Log-Likelihood, Simpson's Paradox, and the Detection of Machine-Generated Text. arXiv:2605.06294
7. WaterSeeker 2024. arXiv:2409.05112
8. GigaCheck 2026. Detecting LLM-Generated Content via Object-Centric Span Localization. ACL 2026
9. FairOPT 2025. Group-Adaptive Threshold Optimization. arXiv:2502.04528
10. RAID 2024. A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ACL 2024
11. Krishna et al. 2023. Paraphrasing evades detectors (DIPPER). NeurIPS 2023, arXiv:2303.13408
12. HowYou Prompt Matters. EMNLP 2024 Findings. aclanthology.org/2024.findings-emnlp.156
13. MASH 2026. ACL 2026 Findings. aclanthology.org/2026.findings-acl.1487
14. Liang et al. 2023. GPT detectors biased against non-native English writers. Patterns
15. Hans et al. 2024. Binoculars. ICML 2024
16. Kirchenbauer et al. 2024. On the Reliability of Watermarks. ICLR 2024, arXiv:2306.04634
17. kylecui/contract-driven-harness-study. Contracts as Task Skeleton (v5 draft). 2026
18. DOMINO 2024. Guiding LLMs The Right Way. ICML 2024, arXiv:2403.06988

---

**END OF FAT DRAFT v0.1**
