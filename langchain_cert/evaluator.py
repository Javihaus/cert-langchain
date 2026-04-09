"""
CERTGroundingEvaluator — LangSmith RunEvaluator for hallucination detection.

Usage:
    from langchain_cert import CERTGroundingEvaluator
    from langsmith.evaluation import evaluate

    evaluator = CERTGroundingEvaluator()
    results = evaluate(
        target_pipeline,
        data="my-langsmith-dataset",
        evaluators=[evaluator],
    )
    evaluator.close()   # flush pending CERT traces if api_key was provided
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from langchain_cert._scoring import (
    DGI_PASS,
    DGIResult,
    SGI_REVIEW,
    SGI_STRONG_PASS,
    SGIResult,
    compute_dgi,
    compute_sgi,
)

logger = logging.getLogger(__name__)


class CERTGroundingEvaluator:
    """
    LangSmith-compatible evaluator that detects LLM hallucinations using
    embedding geometry — no second LLM required.

    Scoring modes:
        SGI (with context): dist(response, question) / dist(response, context)
        DGI (no context):   cosine(normalize(r - q), mu_hat)

    The method is chosen automatically based on whether context is present.

    Scores are normalized to [0, 1] for LangSmith compatibility.
    The raw SGI/DGI values appear in the ``comment`` field of each result.

    Args:
        api_key:       CERT API key for logging traces to the CERT dashboard.
                       If omitted, scoring runs locally — no network calls.
        project:       CERT project name for trace grouping.
        threshold:     Normalized score [0, 1] below which a run is marked
                       failing. Default 0.45 maps approximately to SGI=0.95
                       (the review threshold from arXiv:2512.13771).
        model_name:    Sentence transformer model for local scoring.
        dashboard_url: Override for self-hosted CERT deployments.

    Thresholds reference (raw scores):
        SGI >= 1.20  : strong context engagement (green)
        SGI 0.95-1.20: partial engagement, review recommended (amber)
        SGI < 0.95   : confabulation risk (red)
        DGI >= 0.30  : displacement matches grounded patterns (green)
        DGI < 0.30   : unusual displacement, flag for review (red)
    """

    key = "cert_grounding"   # LangSmith result key

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: str = "langsmith-evaluation",
        threshold: float = 0.45,
        model_name: str = "all-MiniLM-L6-v2",
        dashboard_url: str = "https://cert-framework.com",
    ) -> None:
        self.api_key       = api_key
        self.project       = project
        self.threshold     = threshold
        self.model_name    = model_name
        self.dashboard_url = dashboard_url
        self._client       = None

    # ── LangSmith RunEvaluator protocol ──────────────────────────────────────

    def evaluate_run(self, run: Any, example: Any = None) -> Any:
        """
        Evaluate a LangSmith Run for grounding.

        Called automatically by LangSmith during evaluate() calls.
        Also usable directly in custom evaluation pipelines.

        Args:
            run:     LangSmith Run object.
                     Expected input keys: "input", "question", "query", or "prompt".
                     Expected output keys: "output", "answer", "response", or "text".
            example: Optional LangSmith Example. If it contains a "context" or
                     "reference" key in its outputs, SGI is used instead of DGI.

        Returns:
            EvaluationResult with key="cert_grounding", score in [0, 1], and
            a comment describing method, raw score, and interpretation.
        """
        try:
            from langsmith.evaluation.evaluator import EvaluationResult
        except ImportError as e:
            raise ImportError(
                "langsmith is required: pip install langsmith"
            ) from e

        input_text = self._extract_input(run.inputs)
        if not input_text:
            return EvaluationResult(
                key=self.key,
                score=None,
                comment=(
                    "Could not extract input from run. "
                    "Expected key: 'input', 'question', 'query', or 'prompt'."
                ),
            )

        output_text = self._extract_output(run.outputs or {})
        if not output_text:
            return EvaluationResult(
                key=self.key,
                score=None,
                comment=(
                    "Could not extract output from run. "
                    "Expected key: 'output', 'answer', 'response', or 'text'."
                ),
            )

        context = self._extract_context(run.inputs, example)

        # Compute grounding score
        result: Union[SGIResult, DGIResult]
        if context:
            result = compute_sgi(
                question=input_text,
                context=context,
                response=output_text,
                model_name=self.model_name,
            )
            method_desc = (
                f"SGI={result.raw_score:.3f} "
                f"(review<{SGI_REVIEW}, strong>={SGI_STRONG_PASS})"
            )
            score = result.normalized
            flagged = result.flag
        else:
            result = compute_dgi(
                question=input_text,
                response=output_text,
                model_name=self.model_name,
            )
            method_desc = f"DGI={result.raw_score:.3f} (flag<{DGI_PASS})"
            score = result.normalized
            flagged = result.flag

        passed = score >= self.threshold

        comment = (
            f"Method: {result.method.upper()} | "
            f"{method_desc} | "
            f"Normalized: {score:.3f} | "
            f"{'PASS' if passed else 'FAIL — potential hallucination'}"
        )

        if flagged and not passed:
            comment += (
                " | NOTE: Type III hallucinations (factual errors in correct "
                "semantic frame) are not geometrically detectable."
            )

        # Log trace to CERT dashboard asynchronously (non-blocking)
        if self.api_key:
            self._log_trace(
                input_text=input_text,
                output_text=output_text,
                context=context,
                model=getattr(run, "name", "unknown"),
                score=score,
                method=result.method,
            )

        return EvaluationResult(key=self.key, score=score, comment=comment)

    # ── Context manager support ───────────────────────────────────────────────

    def close(self) -> None:
        """Flush pending CERT traces. Call after evaluate() completes."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.warning("Failed to close CERT client: %s", exc)
            finally:
                self._client = None

    def __enter__(self) -> "CERTGroundingEvaluator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_input(self, inputs: Dict[str, Any]) -> Optional[str]:
        for key in ("input", "question", "query", "prompt", "human_input"):
            if key in inputs and isinstance(inputs[key], str):
                return str(inputs[key])
        return None

    def _extract_output(self, outputs: Dict[str, Any]) -> Optional[str]:
        for key in ("output", "answer", "response", "text", "result"):
            if key in outputs and isinstance(outputs[key], str):
                return str(outputs[key])
        return None

    def _extract_context(
        self, inputs: Dict[str, Any], example: Any = None
    ) -> Optional[str]:
        # Check run inputs first
        for key in ("context", "documents", "retrieved_docs", "knowledge"):
            if key in inputs:
                val = inputs[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list):
                    return "\n\n".join(
                        d.page_content if hasattr(d, "page_content") else str(d)
                        for d in val
                    )
        # Fall back to example outputs (dataset-level context)
        if example and hasattr(example, "outputs") and example.outputs:
            for key in ("context", "reference", "ground_truth"):
                if key in example.outputs:
                    return str(example.outputs[key])
        return None

    def _log_trace(
        self,
        input_text: str,
        output_text: str,
        context: Optional[str],
        model: str,
        score: float,
        method: str,
    ) -> None:
        """Log trace to CERT dashboard. Non-fatal on failure."""
        try:
            client = self._get_client()
            # Verified against cert-sdk 1.0.3: CertClient.trace() requires
            # provider, model, input_text, output_text.
            client.trace(
                provider="langchain",
                model=model,
                input_text=input_text,
                output_text=output_text,
                knowledge_base=context,
                evaluation_mode="grounded" if context else "ungrounded",
                task_type="evaluation",
                metadata={"score": score, "method": method},
            )
        except Exception as exc:
            logger.warning("CERT trace logging failed (non-fatal): %s", exc)

    def _get_client(self) -> Any:
        """Lazy-load CERT SDK client."""
        if self._client is None:
            try:
                from cert import CertClient
            except ImportError as e:
                raise ImportError(
                    "cert-sdk is required for dashboard logging: "
                    "pip install langchain-cert[dashboard]"
                ) from e
            self._client = CertClient(
                api_key=self.api_key,
                project=self.project,
                dashboard_url=self.dashboard_url,
            )
        return self._client
