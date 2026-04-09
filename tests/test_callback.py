"""Tests for CERTCallbackHandler."""

import uuid
from unittest.mock import MagicMock, patch

from langchain_cert.callback import CERTCallbackHandler


class TestCERTCallbackHandler:

    def test_on_llm_start_stores_input(self):
        handler = CERTCallbackHandler(api_key="cert_test")
        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["Hello world"], run_id=run_id)
        assert str(run_id) in handler._pending
        assert handler._pending[str(run_id)] == "Hello world"

    def test_on_llm_error_cleans_up(self):
        handler = CERTCallbackHandler(api_key="cert_test")
        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["test"], run_id=run_id)
        handler.on_llm_error(Exception("fail"), run_id=run_id)
        assert str(run_id) not in handler._pending

    def test_on_llm_end_cleans_up_pending(self):
        handler = CERTCallbackHandler(api_key="cert_test")
        run_id = uuid.uuid4()
        handler.on_llm_start({}, ["What is AI?"], run_id=run_id)

        response = MagicMock()
        generation = MagicMock()
        generation.text = "AI is artificial intelligence."
        response.generations = [[generation]]

        with patch.object(handler, "_log") as mock_log:
            handler.on_llm_end(response, run_id=run_id)
            mock_log.assert_called_once()

        assert str(run_id) not in handler._pending

    def test_context_manager(self):
        with CERTCallbackHandler(api_key="cert_test") as handler:
            assert handler is not None
