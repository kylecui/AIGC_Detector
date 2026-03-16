# AIGC_Detector Runbook

> Hands-on operating manual for setup, training, evaluation, and service checks.

---

## 1. Common Commands

## 1.1 Install dependencies

```bash
uv sync
```

## 1.2 Sanity check

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run ruff check src scripts tests
```

---

## 2. Local Model Preparation

Required local directories:

```text
models/statistical-en/
models/statistical-zh/
models/encoder-en/
models/encoder-zh/
models/base/deberta-v3-large/
models/base/chinese-roberta-wwm-ext-large/
```

If base models are missing, download them locally:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/deberta-v3-large', local_dir='models/base/deberta-v3-large')"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hfl/chinese-roberta-wwm-ext-large', local_dir='models/base/chinese-roberta-wwm-ext-large')"
```

---

## 3. Cloud Training Workflow

## 3.1 Upload / execute remote commands

```bash
uv run python scripts/remote_cmd.py "nvidia-smi"
uv run python scripts/upload_cloud.py
```

## 3.2 Launch cloud training

Cloud helper:

```bash
uv run python scripts/train_cloud.py --help
```

Actual cloud training in this project was orchestrated remotely via tmux and helper scripts.

---

## 4. Download Trained Adapters From Cloud

```bash
uv run python scripts/download_cloud.py --remote /data/aigc/models --local models
```

Expected:

- `models/encoder-en/adapter_config.json`
- `models/encoder-zh/adapter_config.json`

---

## 5. Evaluate Models

## 5.1 Full evaluation

```bash
uv run python scripts/evaluate.py
```

## 5.2 Only EN encoder

```bash
uv run python scripts/evaluate.py --lang en --detectors encoder
```

## 5.3 Only ZH encoder

```bash
uv run python scripts/evaluate.py --lang zh --detectors encoder
```

Reports are written to:

```text
models/eval_reports/
```

---

## 6. Start API Locally

```bash
uv run python -m uvicorn aigc_detector.api.main:app --host 127.0.0.1 --port 8000
```

---

## 7. Smoke Tests

## 7.1 Health

```bash
uv run python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/api/v1/health', timeout=30).text)"
```

## 7.2 Real detect request

```bash
uv run python -c "import httpx; payload={'text':'This is a sufficiently long English passage for endpoint smoke testing. It verifies end-to-end request handling and structured output.', 'models':['all']}; r=httpx.post('http://127.0.0.1:8000/api/v1/detect', json=payload, timeout=180); print(r.status_code); print(r.text)"
```

---

## 8. Troubleshooting

## 8.1 ZH encoder load error

Symptom:

- `storage has wrong byte size`

Action:

1. delete local corrupted base model dir
2. redownload `models/base/chinese-roberta-wwm-ext-large`
3. retry evaluation/runtime

Command:

```bash
uv run python -c "from pathlib import Path; import shutil; shutil.rmtree(Path('models/base/chinese-roberta-wwm-ext-large'), ignore_errors=True)"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hfl/chinese-roberta-wwm-ext-large', local_dir='models/base/chinese-roberta-wwm-ext-large', allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','added_tokens.json','.gitattributes'], force_download=True)"
```

## 8.2 API starts but `/detect` fails

Check:

- `models/statistical-*` exist
- `models/encoder-*` exist
- `models/base/*` exist
- GPU available

## 8.3 First request is slow

Expected. Model wrappers are created at startup, but weights are lazily loaded on first actual use.

---

## 9. Recommended Release Checklist

- [ ] `uv sync`
- [ ] `uv run ruff check src scripts tests`
- [ ] `uv run python scripts/evaluate.py`
- [ ] verify `models/eval_reports/comparison.json`
- [ ] start API
- [ ] verify `/api/v1/health`
- [ ] verify `/api/v1/detect`
- [ ] confirm `models/base/` exists for EN and ZH

---

## 10. Current Known Good Metrics

| Detector | F1 |
|---|---:|
| statistical-en | 0.9913 |
| encoder-en | 0.9987 |
| statistical-zh | 0.9688 |
| encoder-zh | 0.9704 |
