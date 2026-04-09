# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-04-09

### Added
- `CERTGroundingEvaluator` — LangSmith RunEvaluator implementing SGI and DGI
- `CERTCallbackHandler` — LangChain callback for automatic trace logging
- Self-contained SGI/DGI scoring in `langchain_cert/_scoring.py`
  (no cert-framework dependency; implements arXiv:2512.13771 and arXiv:2602.13224)
- Type-correct context manager support on both classes (`with` / `close()`)
- CI workflow: tests on Python 3.9, 3.11, 3.12
- Tests for the geometric computation without mocking
