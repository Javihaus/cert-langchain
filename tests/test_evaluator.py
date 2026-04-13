"""
Tests for CERTGroundingEvaluator — mock LangSmith, real scoring.
"""

from unittest.mock import MagicMock

from langchain_cert.evaluator import CERTGroundingEvaluator


def make_run(input_val: str, output_val: str, context_val: str = None):
    run = MagicMock()
    run.name = "test-run"
    run.inputs = {"input": input_val}
    if context_val:
        run.inputs["context"] = context_val
    run.outputs = {"output": output_val}
    return run


class TestCERTGroundingEvaluator:

    def test_init_defaults(self):
        ev = CERTGroundingEvaluator()
        assert ev.threshold == 0.45
        assert ev.project == "langsmith-evaluation"
        assert ev.api_key is None
        assert ev.reference_csv is None
        assert ev.model_name == "all-MiniLM-L6-v2"

    def test_evaluate_run_returns_evaluation_result(self):
        from langsmith.evaluation.evaluator import EvaluationResult

        ev = CERTGroundingEvaluator()
        run = make_run(
            "What does the policy cover?",
            "The policy covers flood damage up to $50,000.",
            context_val="Flood coverage limit is $50,000.",
        )
        result = ev.evaluate_run(run)
        assert isinstance(result, EvaluationResult)
        assert result.key == "cert_grounding"
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0
        assert "SGI" in result.comment

    def test_evaluate_run_dgi_mode_when_no_context(self):
        from langsmith.evaluation.evaluator import EvaluationResult

        ev = CERTGroundingEvaluator()
        run = make_run(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        result = ev.evaluate_run(run)
        assert isinstance(result, EvaluationResult)
        assert "DGI" in result.comment

    def test_evaluate_run_missing_input_returns_none_score(self):
        ev = CERTGroundingEvaluator()
        run = MagicMock()
        run.inputs = {"unknown_key": "something"}
        run.outputs = {"output": "something"}
        result = ev.evaluate_run(run)
        assert result.score is None

    def test_evaluate_run_missing_output_returns_none_score(self):
        ev = CERTGroundingEvaluator()
        run = MagicMock()
        run.inputs = {"input": "question"}
        run.outputs = {"unknown_key": "something"}
        result = ev.evaluate_run(run)
        assert result.score is None

    def test_extract_input_keys(self):
        ev = CERTGroundingEvaluator()
        assert ev._extract_input({"input": "x"}) == "x"
        assert ev._extract_input({"question": "x"}) == "x"
        assert ev._extract_input({"query": "x"}) == "x"
        assert ev._extract_input({"prompt": "x"}) == "x"
        assert ev._extract_input({"unknown": "x"}) is None

    def test_extract_output_keys(self):
        ev = CERTGroundingEvaluator()
        assert ev._extract_output({"output": "x"}) == "x"
        assert ev._extract_output({"answer": "x"}) == "x"
        assert ev._extract_output({"response": "x"}) == "x"
        assert ev._extract_output({}) is None

    def test_extract_context_from_inputs(self):
        ev = CERTGroundingEvaluator()
        assert ev._extract_context({"context": "docs"}) == "docs"
        assert ev._extract_context({"no_context": "x"}) is None

    def test_extract_context_list_of_strings(self):
        ev = CERTGroundingEvaluator()
        result = ev._extract_context({"context": ["doc1", "doc2"]})
        assert "doc1" in result
        assert "doc2" in result

    def test_extract_context_from_example_outputs(self):
        ev = CERTGroundingEvaluator()
        example = MagicMock()
        example.outputs = {"reference": "ground truth"}
        assert ev._extract_context({}, example) == "ground truth"

    def test_context_manager(self):
        with CERTGroundingEvaluator() as ev:
            assert ev is not None
        # close() should not raise even with no client

    def test_no_api_call_without_api_key(self):
        """Scoring must work locally with no api_key — no network calls."""
        ev = CERTGroundingEvaluator(api_key=None)
        run = make_run("Q?", "A.")
        # Should not attempt to create a CertClient
        result = ev.evaluate_run(run)
        assert result.score is not None
        assert ev._client is None
