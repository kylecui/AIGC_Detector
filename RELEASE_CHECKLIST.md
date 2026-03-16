# Release Checklist

Use this checklist before handing the project to another engineer or running a demo/release.

---

## 1. Environment

- [ ] `uv sync` completed successfully
- [ ] CUDA visible: `uv run python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `.env` created from `.env.example` when needed

---

## 2. Required Model Assets

- [ ] `models/statistical-en/classifier.joblib` exists
- [ ] `models/statistical-zh/classifier.joblib` exists
- [ ] `models/encoder-en/adapter_config.json` exists
- [ ] `models/encoder-zh/adapter_config.json` exists
- [ ] `models/base/deberta-v3-large/` exists
- [ ] `models/base/chinese-roberta-wwm-ext-large/` exists

---

## 3. Code Quality

- [ ] `uv run ruff check src scripts tests`
- [ ] `uv run python -m py_compile src/aigc_detector/detection/encoder.py scripts/evaluate.py`

---

## 4. Evaluation

- [ ] `uv run python scripts/evaluate.py`
- [ ] `models/eval_reports/comparison.json` refreshed
- [ ] latest metrics reviewed

Current known-good F1 targets:

- [ ] EN statistical around `0.9913`
- [ ] EN encoder around `0.9987`
- [ ] ZH statistical around `0.9688`
- [ ] ZH encoder around `0.9704`

---

## 5. API Validation

- [ ] start API with uvicorn
- [ ] `/api/v1/health` returns 200
- [ ] `/api/v1/detect` returns 200

---

## 6. Git / Delivery

- [ ] working tree clean
- [ ] docs updated (`DEPLOYMENT.md`, `RUNBOOK.md`)
- [ ] commits created
- [ ] pushed to remote successfully
