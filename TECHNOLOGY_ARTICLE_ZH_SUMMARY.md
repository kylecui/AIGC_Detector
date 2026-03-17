# technology_article_zh summary

## What this domain is

`technology_article_zh` is the dedicated Chinese domain pack for professional technology writing extracted from the broader `professional_zh` corpus.

It currently includes:

- `dataset/seeds/technology_article_zh/seed_samples.jsonl`
- `dataset/seeds/technology_article_zh/hard_case_eval_v1.jsonl`
- `dataset/technology_article_zh_adaptation_v1/train.jsonl`
- `dataset/technology_article_zh_adaptation_v1/val.jsonl`
- `dataset/technology_article_zh_adaptation_v1/test.jsonl`

## Current corpus size

- seed / hard-case total: `44`
- AI: `22`
- Human: `22`

Adaptation subset:

- train: `28` (`14/14`)
- val: `8` (`4/4`)
- test: `8` (`4/4`)

## Latest dedicated baseline

Script:

- `scripts/eval_technology_article_zh.py`

Report:

- `reports/technology_article_zh_baseline_v1.json`

Latest result:

- overall accuracy: `27 / 44 = 0.614`

## Main conclusion

The detector is not generally unstable on technology writing.

Instead, it is specifically under-sensitive to calm, professional, ToB-style technology commentary and industry-analysis prose.

## Hardest subtype buckets

These subtype groups remained at **0/3 AI recall** in the latest local baseline:

- `industry_analysis`
- `product_trend`
- `platform_engineering`
- `infra_commentary`
- `industry_outlook`

In all of these buckets, human specificity remained strong on the current set.

That means the main issue is:

> low AI recall for specific technology-commentary subtypes,
> not broad human false positives.

## Less problematic subtype buckets

These are comparatively easier for the current detector:

- `workflow_transformation`
- `vendor_ecosystem`
- `organizational_change`

These showed partial improvement but are not fully solved:

- `cio_advisory`
- `enterprise_adoption`

## Practical interpretation

Local sample expansion helped make the blind spot clearer, but it is no longer producing meaningful recall improvement on the hardest subtype cluster.

So the next meaningful step is not endless local expansion.

The next meaningful step is:

> targeted `technology_article_zh` domain adaptation (LoRA)

once remote compute is available again.

## When to resume remote work

The user explicitly paused remote-server usage to save cost.

When the user reopens a server and provides fresh SSH info, use:

- `TECHNOLOGY_ARTICLE_ZH_REMOTE_PLAN.md`

as the execution checklist.
