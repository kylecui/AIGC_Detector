# technology_article_zh remote execution checklist

Use this checklist **only after the user reopens a remote server and provides fresh SSH information**.

## Goal

Run a dedicated `technology_article_zh` LoRA adaptation experiment using the already prepared local subset:

- `dataset/technology_article_zh_adaptation_v1/`

## Preconditions

Before starting, confirm all of the following:

1. User has provided new SSH host / port / password.
2. Remote server has not been assumed from old credentials.
3. `tmux` is available on the remote host.
4. The torch/Python training environment exists on the remote host.
5. A valid local zh base-model cache is available remotely, or can be mapped to:
   - `/root/.cache/modelscope/hub/hfl/chinese-roberta-wwm-ext-large`

## Local assets to use

- training subset:
  - `dataset/technology_article_zh_adaptation_v1/train.jsonl`
  - `dataset/technology_article_zh_adaptation_v1/val.jsonl`
  - `dataset/technology_article_zh_adaptation_v1/test.jsonl`
- training script:
  - `scripts/train_cloud.py`

## Suggested isolated remote paths

- remote dataset dir:
  - `/data/aigc/dataset/technology_article_zh_adaptation_v1`
- remote script path:
  - `/data/aigc/experiments/technology_article_zh/train_cloud.py`
- remote output dir:
  - `/data/aigc/models/technology-article-zh`
- tmux session name:
  - `technology_article_zh_train`

These paths should remain isolated from generic encoder outputs.

## Step-by-step execution

### 1. Update helper credentials

Update both files with the fresh SSH host/port/password:

- `scripts/remote_cmd.py`
- `scripts/upload_cloud.py`

Do **not** reuse stale host information.

### 2. Verify remote environment

Check:

- GPU type and free memory
- `tmux -V`
- `/data/miniconda/envs/torch/bin/python -V`
- existence of `/data/aigc`
- existence of a valid local zh base model cache

If only `/data/.rootcache/modelscope/hub/hfl/chinese-roberta-wwm-ext-large` exists, copy it into:

- `/root/.cache/modelscope/hub/hfl/chinese-roberta-wwm-ext-large`

before training.

### 3. Upload dataset and training script

Use:

```bash
uv run python scripts/upload_cloud.py --dataset-dir dataset/technology_article_zh_adaptation_v1 --remote-dataset-dir /data/aigc/dataset/technology_article_zh_adaptation_v1 --remote-script-path /data/aigc/experiments/technology_article_zh/train_cloud.py
```

Verify:

- dataset files exist remotely
- line counts match local train/val/test counts

### 4. Launch training in tmux

Run remotely via `scripts/remote_cmd.py`:

```bash
tmux new-session -d -s technology_article_zh_train "cd /data/aigc && /data/miniconda/envs/torch/bin/python /data/aigc/experiments/technology_article_zh/train_cloud.py --lang zh --data-dir /data/aigc/dataset/technology_article_zh_adaptation_v1 --output-dir /data/aigc/models/technology-article-zh > /data/aigc/models/technology-article-zh/train.log 2>&1"
```

### 5. Verify early startup

Check that logs show all of these:

- language filtering succeeded
- local model path resolved
- tokenizer/model loaded
- LoRA modules initialized
- train/val dataset loaded
- no immediate Hugging Face / cache path failure

### 6. After completion

Check:

- `adapter_config.json`
- `adapter_model.safetensors`
- `eval_results.json`

Expected output directory:

- `/data/aigc/models/technology-article-zh/encoder-zh`

### 7. Download for local validation

After the run finishes, download to a local staging dir first, not directly over `models/encoder-zh`.

Suggested local target:

- `models/experiments/technology-article-zh/encoder-zh`

### 8. Local validation after download

1. back up current local `models/encoder-zh`
2. temporarily swap in the new experiment adapter
3. run:
   - dedicated `technology_article_zh` baseline
   - representative professional zh samples
   - segment-level API checks in the current web/backend flow
4. compare against the current baseline report before deciding whether to keep the new adapter active

## Success criteria

The run should be considered promising only if it improves the hardest `technology_article_zh` subtypes, especially:

- `industry_analysis`
- `product_trend`
- `platform_engineering`
- `infra_commentary`
- `industry_outlook`

Success is **not** just producing an adapter; it must improve recall on these subtype buckets without introducing broad human false positives.

## Current local artifacts already prepared

- `dataset/seeds/technology_article_zh/seed_samples.jsonl`
- `dataset/seeds/technology_article_zh/hard_case_eval_v1.jsonl`
- `dataset/technology_article_zh_adaptation_v1/train.jsonl`
- `dataset/technology_article_zh_adaptation_v1/val.jsonl`
- `dataset/technology_article_zh_adaptation_v1/test.jsonl`
- `dataset/technology_article_zh_adaptation_v1/metadata.json`

## Important reminder

The user explicitly said the remote server is turned off for now.

Therefore:

- do not attempt remote commands until the user provides fresh access info
- when remote use becomes necessary, ask for / use the new link info first
