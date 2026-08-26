# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-21

First tagged release. Positioning: self-hosted, verifiable bilingual
(ZH/EN) AI-generated text detection instrument — every verdict ships with
segment evidence, confidence, and declared blind spots. See
docs/capability-statement.md for the reliability boundary and
docs/sla-statement.md for the operating envelope.

### Added

- **Detection pipeline (four-stage ensemble)** — statistical LM-probability
  stage, 14-feature linguistic stylometry, LoRA encoder classifiers, and a
  zero-shot Binoculars fallback, orchestrated by a language-routed cascade
  with early exits (see docs/architecture.md).
- **Register caveat (W3a/W3b)** — lexical formal-register (公文体) gate for
  Chinese: formal-document texts get an explicit `caveat` payload with
  action guidance instead of a silent high-confidence verdict.
- **Segment highlights (W2)** — `segment_highlights` response field
  surfacing the strongest local AI traces as an auxiliary review signal,
  decoupled from the document-level verdict.
- **Register-conditioned confidence calibration (W11)** — deployed formal
  temperature T=5.645 (models/calibration/global_temperature.json,
  fitted on the 382-doc formal probe corpus); label-flip-free and
  ranking-preserving by construction; applied to formal-register texts only.
- **W15 binoculars floor OR-rule** — register-gated upgrade rule deployed
  enabled (models/calibration/binoculars_floor.json, cutoff 0.46) after
  gate review PASS (reports/w15_gate_review.md): raw contract-generated
  formal documents are now caught at ~0% miss.
- **EN formal-register downgrade guard (W16)** — English notice/
  announcement texts hit a measured 71% human-misclassification blind spot;
  the product-level guard caps confidence at 0.49 and attaches a strong
  warning instead of issuing a normal-confidence verdict.
- **WebUI evidence blocks** — caveat bar, segment highlights, and
  calibrated-confidence display in the frontend.
- **Service logging with sanitizing guard** — structured logs with a
  hygiene gate that keeps submitted text out of log files.
- **Docker packaging** — single-stage GPU runtime image, no baked-in
  models (HF cache volume); deployment guide at docs/deploy-docker.md.
- **SLA statement** — single-tenant self-hosted operating envelope with
  measured latency/VRAM numbers (docs/sla-statement.md).
- **Model pins** — all runtime models pinned to exact Hugging Face
  revisions (models/calibration/model_pins.json).
- **License** — Apache-2.0 (LICENSE), with an AGPL-avoidance note for the
  optional `pymupdf` dependency.
- **Console entry point + self-contained wheel** — `aigc-detector
  serve|doctor`; the wheel force-includes static assets and the model
  registry so installed layouts work without the repo checkout.
- **Release verification gate** — `scripts/verify_release.ps1` (10/10 PASS
  on the final run: license, README entry, wheel self-containment, CLI
  doctor, calibration artifacts, model pins, test suite).
- **Test suite** — 310 automated tests covering the API contract, segment
  contract, highlights, calibration, floor rule, register gates, log
  hygiene, and model-cache behavior.
- **Documentation set** — capability statement, ADR-0001 (formal-register
  encoder retraining deferral, docs/adr/), evaluation reports and
  validation dossier (reports/, docs/research/), lab notebook
  (DETECTOR_NOTES_*.md).

### Changed

- README quick start now documents the real entry (`aigc-detector serve` /
  `uvicorn aigc_detector.api.main:app`) plus the model bootstrap scripts
  (scripts/download_models.py, scripts/prefetch_binoculars.py).
- Runtime models resolved exclusively through the pinned registry
  (exact HF revisions) for reproducible deployments.

### Fixed

- README quick start previously referenced the stub `main.py` entry;
  replaced with the real uvicorn command.
- Release verification gate check semantics (first PASS run recorded after
  the wheel-layout calibration-path fix).
