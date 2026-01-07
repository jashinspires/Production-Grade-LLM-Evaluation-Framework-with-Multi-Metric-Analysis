"""
Unit tests for BERTScore metric.
"""

import pytest

from llm_eval.metrics.bertscore import BERTScoreMetric


class TestBERTScoreMetric:
    """Tests for BERTScoreMetric."""
    
    @pytest.fixture(scope="class")
    def metric(self):
        """Create BERTScore metric instance (cached for class)."""
        return BERTScoreMetric()
    
    def test_perfect_match(self, metric):
        """Test BERTScore for identical texts."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert result.score > 0.95
    
    def test_semantic_similarity(self, metric):
        """Test BERTScore for semantically similar texts."""
        result = metric.compute(
            prediction="Paris is France's capital city.",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert result.score > 0.7  # Should recognize semantic similarity
    
    def test_different_texts(self, metric):
        """Test BERTScore for different texts."""
        result = metric.compute(
            prediction="The weather is nice today.",
            reference="Cats like to sleep on soft surfaces."
        )
        
        assert result.is_valid
        assert result.score < 0.7  # Should be lower for unrelated texts
    
    def test_empty_prediction(self, metric):
        """Test BERTScore with empty prediction."""
        result = metric.compute(
            prediction="",
            reference="The capital of France is Paris."
        )
        
        assert result.is_valid
        assert result.score == 0.0
    
    def test_empty_reference(self, metric):
        """Test BERTScore with empty reference."""
        result = metric.compute(
            prediction="The capital of France is Paris.",
            reference=""
        )
        
        assert result.is_valid
        assert result.score == 0.0
    
    def test_raw_cosine_in_details(self, metric):
        """Test that raw cosine similarity is in details."""
        result = metric.compute(
            prediction="Machine learning is AI.",
            reference="Machine learning is artificial intelligence."
        )
        
        assert "raw_cosine_similarity" in result.details
        assert "model_name" in result.details
    
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
    
    def test_batch_with_empty_inputs(self, metric):
        """Test batch with some empty inputs."""
        predictions = ["Paris is the capital.", "", "AI"]
        references = ["The capital is Paris.", "Reference", ""]
        
        results = metric.compute_batch(predictions, references)
        
        assert len(results) == 3
        assert results[0].is_valid
        assert results[1].score == 0.0
        assert results[2].score == 0.0
