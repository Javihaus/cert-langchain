"""
CERTCallbackHandler — LangChain callback for automatic trace logging.

Logs every LLM call in a LangChain chain to CERT without pipeline modification.

Usage:
    from langchain_cert import CERTCallbackHandler
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        callbacks=[CERTCallbackHandler(api_key="cert_xxx", project="prod")]
    )

Note:
    This handler logs traces to the CERT dashboard for async evaluation.
    It does NOT compute grounding scores synchronously — evaluation runs in
    the CERT worker pipeline. For synchronous scoring, use CERTGroundingEvaluator.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class CERTCallbackHandler:
    """
    LangChain callback handler that logs LLM calls to CERT for evaluation.

    Args:
        api_key:       CERT API key (required for dashboard logging).
        project:       CERT project name for trace grouping.
        dashboard_url: Override for self-hosted CERT deployments.
    """

    def __init__(
        self,
        api_key: str,
        project: str = "langchain",
        dashboard_url: str = "https://cert-framework.com",
    ) -> None:
        self.api_key       = api_key
        self.project       = project
        self.dashboard_url = dashboard_url
        self._client       = None
        self._pending: dict[str, str] = {}  # run_id -> input_text

    # ── LangChain callback protocol ───────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Store input prompt, keyed by run_id, for pairing with output."""
        if prompts:
            self._pending[str(run_id)] = prompts[0]

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Log completed LLM call to CERT."""
        run_id_str   = str(run_id)
        input_text   = self._pending.pop(run_id_str, "")

        try:
            output_text = response.generations[0][0].text
        except (IndexError, AttributeError):
            logger.debug("Could not extract output text from LLM response.")
            return

        model = (kwargs.get("invocation_params") or {}).get("model_name", "unknown")

        self._log(input_text=input_text, output_text=output_text, model=model)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Remove pending input on error — prevents memory leak."""
        self._pending.pop(str(run_id), None)

    def close(self) -> None:
        """Flush pending CERT traces."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.warning("Failed to close CERT client: %s", exc)
            finally:
                self._client = None

    def __enter__(self) -> "CERTCallbackHandler":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Private ───────────────────────────────────────────────────────────────

    def _log(self, input_text: str, output_text: str, model: str) -> None:
        try:
            client = self._get_client()
            # Verified against cert-sdk 1.0.3: CertClient.trace() requires
            # provider, model, input_text, output_text.
            client.trace(
                provider="langchain",
                model=model,
                input_text=input_text,
                output_text=output_text,
            )
        except Exception as exc:
            logger.warning("CERT callback logging failed (non-fatal): %s", exc)

    def _get_client(self) -> Any:
        if self._client is None:
            from cert import CertClient
            self._client = CertClient(
                api_key=self.api_key,
                project=self.project,
                dashboard_url=self.dashboard_url,
            )
        return self._client
