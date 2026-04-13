# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-04-12

### Added
- `CERTGroundingEvaluator` — LangSmith RunEvaluator implementing SGI and DGI
  scoring. Method chosen automatically: SGI when context is present,
  DGI when it is not.
- `CERTCallbackHandler` — LangChain callback for automatic trace logging
  to the CERT dashboard without modifying pipeline code.
- Self-contained SGI/DGI scoring in `langchain_cert/_scoring.py`.
  No cert-framework dependency. Implements arXiv:2512.13771 (SGI) and
  arXiv:2602.13224 (DGI).
- `reference_csv` parameter on `CERTGroundingEvaluator` for domain-specific
  DGI calibration. Provide a path to a CSV of verified grounded
  (question, response) pairs. Generic (bundled) calibration achieves
  AUROC ~0.76; domain-specific calibration typically reaches 0.90+.
- Bundled reference dataset: 212 grounded pairs across finance, medical,
  science, history, geography, law, coding, and general domains.
- DGI reference direction cache keyed by `(model_name, reference_csv)`
  to prevent collisions when switching datasets in the same process.
- `_load_user_csv()` with auto-delimiter detection (comma or semicolon)
  and explicit error messages for missing columns or empty files.
- Context manager support on both classes (`with` statement / `close()`).
- CI workflow: lint, type check, and tests across Python 3.9, 3.11, 3.12.
