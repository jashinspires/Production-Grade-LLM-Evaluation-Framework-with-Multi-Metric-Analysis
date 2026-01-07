"""
Unit tests for BLEU metric.
"""

import pytest

from llm_eval.metrics.bleu import BLEUMetric


class TestBLEUMetric:
    """Tests for BLEUMetric."""
    
    @pytest.fixture
    def metric(self):
        """Create BLEU metric instance."""
        return BLEUMetric()
    
    def test_perfect_match(self, metric):
        """Test BLEU score for identical texts."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert result.score > 0.9  # Should be very high for identical texts
    
    def test_partial_match(self, metric):
        """Test BLEU score for partially matching texts."""
        result = metric.compute(
            prediction="Paris is the capital of France.",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert 0.3 < result.score < 1.0  # Should have some overlap
    
    def test_no_match(self, metric):
        """Test BLEU score for completely different texts."""
        result = metric.compute(
            prediction="The sky is blue.",
            reference="Cats like to sleep."
        )
        
        assert result.is_valid
        assert result.score < 0.3  # Should be low
    
    def test_empty_prediction(self, metric):
        """Test BLEU with empty prediction."""
        result = metric.compute(
            prediction="",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert result.score == 0.0
        assert "Empty prediction" in result.details.get("reason", "")
    
    def test_empty_reference(self, metric):
        """Test BLEU with empty reference."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference=""
        )
        
        assert result.is_valid
        assert result.score == 0.0
    
    def test_ngram_scores_in_details(self, metric):
        """Test that n-gram scores are in details."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference="The capital of France is Paris."
        )
        
        assert "ngram_scores" in result.details
        assert "bleu_1" in result.details["ngram_scores"]
    
    def test_short_text(self, metric):
        """Test BLEU with very short text."""
        result = metric.compute(
            prediction="Paris",
            reference="Paris is the capital."
        )
        
        assert result.is_valid
        assert result.score >= 0.0
    
    def test_batch_compute(self, metric):
        """Test batch computation."""
        predictions = [
            "Paris is the capital.",
            "Shakespeare wrote plays.",
            "AI is machine learning."
        ]
        references = [
            "Paris is the capital of France.",
            "Shakespeare was a playwright.",
            "AI stands for artificial intelligence."
        ]
        
        results = metric.compute_batch(predictions, references)
        
        assert len(results) == 3
        assert all(r.is_valid for r in results)
    
    def test_custom_ngram(self):
        """Test with custom n-gram order."""
        metric = BLEUMetric(max_ngram=2)
        
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert metric.max_ngram == 2
    
    def test_smoothing_disabled(self):
        """Test with smoothing disabled."""
        metric = BLEUMetric(smoothing=False)
        
        result = metric.compute(
            prediction="Paris",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
