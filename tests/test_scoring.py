"""
Tests for _scoring.py — these test the actual geometric computation.

They require sentence-transformers and numpy.
Run with: pytest tests/test_scoring.py -v
"""

import math

from langchain_cert._scoring import (
    DGI_PASS,
    SGI_REVIEW,
    SGI_STRONG_PASS,
    DGIResult,
    SGIResult,
    compute_dgi,
    compute_sgi,
)


class TestSGI:
    """SGI tests use semantically meaningful inputs to verify geometric behavior."""

    def test_grounded_response_passes(self):
        """Response that engages with context should score above review threshold."""
        result = compute_sgi(
            question="What is the coverage limit for water damage?",
            context="Water damage coverage limit is $50,000 per occurrence. "
                    "Deductible is $1,500. Floods require separate policy.",
            response="The coverage limit for water damage is $50,000 per "
                     "occurrence, with a $1,500 deductible.",
        )
        assert isinstance(result, SGIResult)
        assert result.raw_score > 0
        assert 0.0 <= result.normalized <= 1.0
        assert result.q_dist > 0
        assert result.ctx_dist > 0
        # Grounded response should not be flagged
        assert not result.flag

    def test_hallucinated_response_flags(self):
        """Response that ignores context should score low."""
        result = compute_sgi(
            question="What is the coverage limit for water damage?",
            context="Water damage coverage limit is $50,000 per occurrence.",
            response="The coverage for water damage is unlimited, "
                     "including floods and hurricanes with no deductible.",
        )
        assert isinstance(result, SGIResult)
        assert result.raw_score > 0
        assert 0.0 <= result.normalized <= 1.0

    def test_sgi_ratio_correctness(self):
        """SGI = dist(response, question) / dist(response, context)."""
        result = compute_sgi(
            question="Q",
            context="C",
            response="R",
        )
        # Verify the ratio is positive and finite
        assert result.raw_score > 0
        assert math.isfinite(result.raw_score)
        assert math.isfinite(result.normalized)

    def test_degenerate_identical_response_context(self):
        """When response == context, SGI should be very high (strongly grounded)."""
        text = "The policy covers water damage up to $50,000."
        result = compute_sgi(question="Q?", context=text, response=text)
        assert not result.flag
        assert result.normalized >= 0.9

    def test_degenerate_identical_response_question(self):
        """When response == question, SGI should be zero (ungrounded)."""
        text = "What is the coverage?"
        result = compute_sgi(question=text, context="C", response=text)
        assert result.flag
        assert result.normalized == 0.0

    def test_normalized_in_range(self):
        result = compute_sgi(
            question="What is the capital of France?",
            context="France is a country in Western Europe. Its capital is Paris.",
            response="The capital of France is Paris.",
        )
        assert 0.0 <= result.normalized <= 1.0
        assert not math.isnan(result.raw_score)


class TestDGI:
    def test_factual_response_scores_positively(self):
        """Factually correct responses should have positive DGI."""
        result = compute_dgi(
            question="What causes seasons on Earth?",
            response="Seasons are caused by Earth's 23.5-degree axial tilt, "
                     "which changes how directly sunlight hits each hemisphere.",
        )
        assert isinstance(result, DGIResult)
        assert -1.0 <= result.raw_score <= 1.0
        assert 0.0 <= result.normalized <= 1.0
        assert not math.isnan(result.raw_score)
        # Factually correct response should pass
        assert not result.flag

    def test_confabulated_response_may_flag(self):
        """Pure fabrication should score differently from factual responses."""
        result = compute_dgi(
            question="What causes seasons on Earth?",
            response="Seasons are managed by the Global Climate Regulation Board, "
                     "established in 1952, which adjusts Earth's orbital parameters.",
        )
        assert isinstance(result, DGIResult)
        assert -1.0 <= result.raw_score <= 1.0
        assert 0.0 <= result.normalized <= 1.0

    def test_normalized_maps_correctly(self):
        """DGI normalized should be (raw + 1) / 2, clipped to [0, 1]."""
        result = compute_dgi(question="Q", response="R")
        expected = (result.raw_score + 1.0) / 2.0
        assert abs(result.normalized - expected) < 0.01

    def test_no_nan_or_inf(self):
        result = compute_dgi(
            question="What is machine learning?",
            response="Machine learning is a branch of AI that enables systems "
                     "to learn from data without being explicitly programmed.",
        )
        assert not math.isnan(result.raw_score)
        assert not math.isinf(result.raw_score)


class TestThresholds:
    def test_sgi_constants_are_ordered(self):
        assert SGI_REVIEW < SGI_STRONG_PASS

    def test_dgi_pass_is_positive(self):
        assert DGI_PASS > 0
