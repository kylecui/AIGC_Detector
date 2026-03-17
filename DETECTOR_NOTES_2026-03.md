# AIGC Detector Notes — 2026-03

## Recent findings

### 1. Formal Chinese false negative was a domain issue, not just routing

- A formal/proposal-style Chinese sample was initially judged as human.
- We fixed the zh pipeline so Stage 1 statistical results no longer early-exit for zh.
- We added a zh-specific arbitration path in `src/aigc_detector/detection/pipeline.py`:
  - if statistical says human
  - and encoder `p_ai >= 0.35`
  - combine encoder-only with `ZH_DECISION_THRESHOLD = 0.47`
- This improved the earlier `formal_zh` case.

### 2. The new `formal_zh_proxy` encoder improved the academic/proposal case

- We built a proxy zh adaptation subset from finance + healthcare.
- Remote experiment output:
  - `/data/aigc/models/formal-zh-proxy/encoder-zh`
- Local end-to-end validation through the existing API succeeded after swapping the local `models/encoder-zh` path.
- Result on the formal Chinese sample:
  - previous generic zh encoder: human
  - `formal_zh_proxy`: AI-generated with high confidence

### 3. Security/BP rhetoric Chinese remains a separate failure mode

- A long Chinese security incident + BP / persuasive rhetoric sample was still predicted as human with high confidence.
- Breakdown:
  - statistical `p_ai ≈ 0.0039`
  - encoder `p_ai ≈ 0.00415`
- This means the sample did **not** hit the zh arbitration path.
- The current pipeline therefore behaved as designed; this is not primarily an ensemble bug.

### 4. Current theoretical status

The repo still partially follows the classic conjecture:

> AI-generated text tends to be smoother / more predictable to a language model.

This is explicitly implemented in the statistical stage via:

- perplexity
- average entropy
- entropy standard deviation
- burstiness
- max/min entropy

But this conjecture is no longer sufficient as the sole detection axis.

### 5. Updated architecture judgment

The best current strategy is:

- **Primary axis:** domain-aware encoder + small-sample adaptation
- **Auxiliary axis:** statistical predictability features
- **Fallback / evidence axis:** training-free contrastive methods (e.g. binoculars)

### 6. Dataset / prompt coverage gap identified

Current zh prompts cover:

- domains: technology, education, healthcare, law, finance
- styles: news_report, argumentative, popular_science, commentary

Current automated zh data does **not** explicitly cover:

- security incident retrospectives
- vendor pitch / BP rhetoric
- social-media longform persuasive Chinese
- emoji-heavy and slogan-heavy structured prose
- security architecture + business persuasion hybrids

### 7. Third-party PDF report insight

Two external PaperPass reports suggest a useful product and modeling direction:

- They produce **document-level** and **segment-level** suspiciousness.
- The formal architecture sample scored high across many segments.
- The security/BP sample had only moderate document-level suspicion, but several local segments still scored as suspicious.

Main takeaway:

> We should add segment-level detection rather than relying only on document-level output.

## P1 decision

P1 = add backend segment-level detection support.

Minimal goals:

1. Keep existing `/api/v1/detect` fields backward-compatible.
2. Add optional segment-level results to the response.
3. Run the existing detection pipeline on the full text and on segments.
4. Surface suspicious spans for long zh texts, especially formal and security/BP rhetoric styles.

## Next domain iteration after P1

Likely next zh adaptation bucket:

- `security_bp_zh`
- or `structured_persuasive_zh`

Target text patterns:

- security incident narration
- attack/defense retrospectives
- architecture + acronym-heavy exposition
- commercial BP / sales-pitch rhetoric
- emoji + bullet + slogan structure

## Dedicated technology_article_zh findings

### Why a dedicated domain pack was created

After expanding the broader `professional_zh` corpus and repeatedly running local baseline evaluation, `technology_article` emerged as the strongest and most stable remaining professional-domain blind spot.

It was then split into a dedicated domain pack with:

- `dataset/seeds/technology_article_zh/seed_samples.jsonl`
- `dataset/seeds/technology_article_zh/hard_case_eval_v1.jsonl`
- `dataset/technology_article_zh_adaptation_v1/train.jsonl`
- `dataset/technology_article_zh_adaptation_v1/val.jsonl`
- `dataset/technology_article_zh_adaptation_v1/test.jsonl`

### Current domain-level conclusion

The detector is not generally unstable on technology writing. Instead, it is specifically under-sensitive to a cluster of calm, professional, ToB-style article subtypes.

### Subtypes that remain the main blind spot

These subtype buckets repeatedly stayed at **0 AI recall** even after local sample expansion:

- `industry_analysis`
- `product_trend`
- `platform_engineering`
- `infra_commentary`
- `industry_outlook`

Common characteristics:

- professional and stable tone
- technology industry analysis
- enterprise / platform / infrastructure perspective
- low emotional volatility
- low overt sales pressure
- high similarity to real industry media or enterprise research prose

### Subtypes that are less problematic

These were comparatively easier for the detector on the current local set:

- `workflow_transformation`
- `vendor_ecosystem`
- `organizational_change`

These partially improved but were not fully solved:

- `cio_advisory`
- `enterprise_adoption`

### Latest local stopping-point judgment

At this point, continued local sample expansion is still useful for documentation and future training input quality, but it is **no longer enough by itself** to materially improve the hardest `technology_article_zh` subtypes.

That means the project has reached the point where the next meaningful step is:

> targeted `technology_article_zh` domain adaptation (LoRA) once remote compute is available again.

### Operational note

The user explicitly paused remote-server usage for cost reasons.

Therefore the correct current stop state is:

1. keep local corpus/domain packs ready
2. do not assume any remote host is still available
3. wait for the user to reopen a server and provide fresh connection info
4. then run the dedicated `technology_article_zh` adaptation workflow
