"""
Unit tests for ROUGE-L metric.
"""

import pytest

from llm_eval.metrics.rouge import ROUGELMetric


class TestROUGELMetric:
    """Tests for ROUGELMetric."""

    @pytest.fixture
    def metric(self):
        """Create ROUGE-L metric instance."""
        return ROUGELMetric()

    def test_perfect_match(self, metric):
        """Test ROUGE-L score for identical texts."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference="The capital of France is Paris.",
        )

        assert result.is_valid
        assert result.score == 1.0

    def test_partial_match(self, metric):
        """Test ROUGE-L score for partially matching texts."""
        result = metric.compute(
            prediction="Paris is the capital of France.",
            reference="The capital of France is Paris.",
        )

        assert result.is_valid
        assert 0.4 < result.score < 1.0

    def test_no_match(self, metric):
        """Test ROUGE-L score for completely different texts."""
        result = metric.compute(
            prediction="The sky is blue today.", reference="Cats enjoy sleeping all day."
        )

        assert result.is_valid
        assert result.score < 0.3

    def test_empty_prediction(self, metric):
        """Test ROUGE-L with empty prediction."""
        result = metric.compute(prediction="", reference="The capital of France is Paris.")

        assert result.is_valid
        assert result.score == 0.0

    def test_empty_reference(self, metric):
        """Test ROUGE-L with empty reference."""
        result = metric.compute(prediction="The capital of France is Paris.", reference="")

        assert result.is_valid
        assert result.score == 0.0

    def test_precision_recall_in_details(self, metric):
        """Test that precision and recall are in details."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference="Paris is the capital city of France.",
        )

        assert "precision" in result.details
        assert "recall" in result.details
        assert "f1" in result.details

    def test_long_text(self, metric):
        """Test ROUGE-L with longer texts."""
        result = metric.compute(
            prediction="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            reference="Machine learning is an AI subset enabling systems to learn from experience without explicit programming.",
        )

        assert result.is_valid
        assert result.score > 0.5

    def test_batch_compute(self, metric):
        """Test batch computation."""
        predictions = [
            "Paris is the capital.",
            "Shakespeare wrote plays.",
            "AI is machine learning.",
        ]
        references = [
            "Paris is the capital of France.",
            "Shakespeare was a playwright.",
            "AI stands for artificial intelligence.",
        ]

        results = metric.compute_batch(predictions, references)

        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_with_stemmer(self):
        """Test with stemmer enabled."""
        metric = ROUGELMetric(use_stemmer=True)

        result = metric.compute(
            prediction="The cats are running quickly.", reference="The cat runs quick."
        )

        assert result.is_valid
        # Stemmer should increase score by matching stems
