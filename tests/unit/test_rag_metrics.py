"""
Unit tests for RAG-specific metrics.
"""

import pytest

from llm_eval.metrics.answer_relevancy import AnswerRelevancyMetric
from llm_eval.metrics.context_relevancy import ContextRelevancyMetric
from llm_eval.metrics.faithfulness import FaithfulnessMetric


class TestFaithfulnessMetric:
    """Tests for FaithfulnessMetric."""

    @pytest.fixture(scope="class")
    def metric(self):
        """Create Faithfulness metric instance."""
        return FaithfulnessMetric()

    def test_faithful_answer(self, metric):
        """Test faithfulness with answer grounded in context."""
        result = metric.compute(
            prediction="Paris is the capital of France.",
            reference="Paris is the capital.",
            contexts=["Paris is the capital and largest city of France."],
        )

        assert result.is_valid
        assert result.score > 0.6

    def test_unfaithful_answer(self, metric):
        """Test faithfulness with answer NOT grounded in context."""
        result = metric.compute(
            prediction="London is the capital of France.",
            reference="Paris is the capital.",
            contexts=["Paris is the capital of France."],
        )

        assert result.is_valid
        # Should still have some score due to word overlap, but lower

    def test_no_context(self, metric):
        """Test faithfulness with no context."""
        result = metric.compute(
            prediction="Paris is the capital.", reference="Paris is the capital.", contexts=[]
        )

        assert result.is_valid
        assert result.score == 0.0
        assert "No context" in result.details.get("reason", "")

    def test_empty_prediction(self, metric):
        """Test faithfulness with empty prediction."""
        result = metric.compute(
            prediction="",
            reference="Paris is the capital.",
            contexts=["Paris is the capital of France."],
        )

        assert result.is_valid
        assert result.score == 0.0

    def test_sentence_scores_in_details(self, metric):
        """Test that sentence scores are in details."""
        result = metric.compute(
            prediction="Paris is the capital. It is a beautiful city.",
            reference="Paris is the capital.",
            contexts=["Paris is the capital of France. Paris is known for its culture."],
        )

        assert "sentence_scores" in result.details
        assert "num_prediction_sentences" in result.details


class TestContextRelevancyMetric:
    """Tests for ContextRelevancyMetric."""

    @pytest.fixture(scope="class")
    def metric(self):
        """Create Context Relevancy metric instance."""
        return ContextRelevancyMetric()

    def test_relevant_context(self, metric):
        """Test with relevant context."""
        result = metric.compute(
            prediction="Paris is the capital.",
            reference="Paris is the capital.",
            query="What is the capital of France?",
            contexts=["Paris is the capital and largest city of France."],
        )

        assert result.is_valid
        assert result.score > 0.5

    def test_irrelevant_context(self, metric):
        """Test with irrelevant context."""
        result = metric.compute(
            prediction="Paris is the capital.",
            reference="Paris is the capital.",
            query="What is the capital of France?",
            contexts=["The weather today is sunny and warm."],
        )

        assert result.is_valid
        assert result.score < 0.6

    def test_no_query(self, metric):
        """Test with no query."""
        result = metric.compute(
            prediction="Paris is the capital.",
            reference="Paris is the capital.",
            query=None,
            contexts=["Paris is the capital of France."],
        )

        assert result.is_valid
        assert result.score == 0.0
        assert "No query" in result.details.get("reason", "")

    def test_no_context(self, metric):
        """Test with no context."""
        result = metric.compute(
            prediction="Paris is the capital.",
            reference="Paris is the capital.",
            query="What is the capital of France?",
            contexts=[],
        )

        assert result.is_valid
        assert result.score == 0.0

    def test_multiple_contexts(self, metric):
        """Test with multiple contexts."""
        result = metric.compute(
            prediction="Paris is the capital.",
            reference="Paris is the capital.",
            query="What is the capital of France?",
            contexts=[
                "Paris is the capital of France.",
                "France is a country in Europe.",
                "The Eiffel Tower is in Paris.",
            ],
        )

        assert result.is_valid
        assert "context_scores" in result.details
        assert len(result.details["context_scores"]) == 3


class TestAnswerRelevancyMetric:
    """Tests for AnswerRelevancyMetric."""

    @pytest.fixture(scope="class")
    def metric(self):
        """Create Answer Relevancy metric instance."""
        return AnswerRelevancyMetric()

    def test_relevant_answer(self, metric):
        """Test with relevant answer."""
        result = metric.compute(
            prediction="Paris is the capital of France.",
            reference="The capital of France is Paris.",
            query="What is the capital of France?",
        )

        assert result.is_valid
        assert result.score > 0.6

    def test_irrelevant_answer(self, metric):
        """Test with irrelevant answer."""
        result = metric.compute(
            prediction="Cats like to sleep.",
            reference="The capital of France is Paris.",
            query="What is the capital of France?",
        )

        assert result.is_valid
        assert result.score < 0.5

    def test_no_query(self, metric):
        """Test with no query."""
        result = metric.compute(
            prediction="Paris is the capital.", reference="Paris is the capital.", query=None
        )

        assert result.is_valid
        assert result.score == 0.0
        assert "No query" in result.details.get("reason", "")

    def test_empty_prediction(self, metric):
        """Test with empty prediction."""
        result = metric.compute(
            prediction="", reference="Paris is the capital.", query="What is the capital of France?"
        )

        assert result.is_valid
        assert result.score == 0.0

    def test_similarity_scores_in_details(self, metric):
        """Test that similarity scores are in details."""
        result = metric.compute(
            prediction="Paris is the capital of France.",
            reference="The capital of France is Paris.",
            query="What is the capital of France?",
        )

        assert "query_answer_similarity" in result.details
        assert "answer_reference_similarity" in result.details
