"""
BERTScore metric implementation.

Provides BERTScore calculation using sentence transformers
for semantic similarity assessment.
"""

from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_eval.metrics.base import Metric, MetricResult
from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class BERTScoreMetric(Metric):
    """
    BERTScore metric for semantic similarity evaluation.
    
    Uses sentence transformers to compute embedding-based
    similarity between predictions and references.
    """
    
    name = "bertscore"
    description = "BERTScore measuring semantic similarity using embeddings"
    score_range = (0.0, 1.0)
    
    # Class-level model cache
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs: Any
    ) -> None:
        """
        Initialize BERTScore metric.
        
        Args:
            model_name: Name of the sentence transformer model
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """Load the sentence transformer model if not already loaded."""
        if BERTScoreMetric._model is None or BERTScoreMetric._model_name != self.model_name:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            BERTScoreMetric._model = SentenceTransformer(self.model_name)
            BERTScoreMetric._model_name = self.model_name
    
    def _cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def compute(
        self,
        prediction: str,
        reference: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute BERTScore between prediction and reference.
        
        Args:
            prediction: The model's predicted answer
            reference: The expected/reference answer
            query: Not used for BERTScore
            contexts: Not used for BERTScore
            **kwargs: Additional arguments (ignored)
            
        Returns:
            MetricResult with BERTScore (cosine similarity)
        """
        try:
            # Handle empty inputs
            if not prediction or not reference:
                return MetricResult(
                    score=0.0,
                    details={"reason": "Empty prediction or reference"}
                )
            
            # Encode texts
            embeddings = BERTScoreMetric._model.encode(
                [prediction, reference],
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Compute cosine similarity
            score = self._cosine_similarity(embeddings[0], embeddings[1])
            
            # Normalize to 0-1 range (cosine similarity can be negative)
            normalized_score = (score + 1) / 2
            
            return MetricResult(
                score=round(normalized_score, 4),
                details={
                    "raw_cosine_similarity": round(score, 4),
                    "model_name": self.model_name,
                }
            )
            
        except Exception as e:
            return MetricResult(
                score=0.0,
                error=f"BERTScore computation failed: {str(e)}"
            )
    
    def compute_batch(
        self,
        predictions: List[str],
        references: List[str],
        queries: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any
    ) -> List[MetricResult]:
        """
        Compute BERTScore for a batch of examples efficiently.
        
        Batches the embedding computation for better performance.
        """
        try:
            results = []
            
            # Filter valid pairs
            valid_pairs = []
            invalid_indices = []
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                if pred and ref:
                    valid_pairs.append((i, pred, ref))
                else:
                    invalid_indices.append(i)
            
            if not valid_pairs:
                return [
                    MetricResult(score=0.0, details={"reason": "Empty input"})
                    for _ in predictions
                ]
            
            # Batch encode all texts
            all_texts = []
            for _, pred, ref in valid_pairs:
                all_texts.extend([pred, ref])
            
            embeddings = BERTScoreMetric._model.encode(
                all_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Compute similarities
            result_map = {}
            for idx, (i, pred, ref) in enumerate(valid_pairs):
                pred_emb = embeddings[idx * 2]
                ref_emb = embeddings[idx * 2 + 1]
                
                score = self._cosine_similarity(pred_emb, ref_emb)
                normalized_score = (score + 1) / 2
                
                result_map[i] = MetricResult(
                    score=round(normalized_score, 4),
                    details={
                        "raw_cosine_similarity": round(score, 4),
                        "model_name": self.model_name,
                    }
                )
            
            # Build final results list
            for i in range(len(predictions)):
                if i in result_map:
                    results.append(result_map[i])
                else:
                    results.append(MetricResult(
                        score=0.0,
                        details={"reason": "Empty prediction or reference"}
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch BERTScore failed: {e}")
            # Fall back to individual computation
            return super().compute_batch(
                predictions=predictions,
                references=references,
                queries=queries,
                contexts=contexts,
                **kwargs
            )
