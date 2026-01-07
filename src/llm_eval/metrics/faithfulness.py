"""
Faithfulness metric implementation.

Measures whether the model's answer is grounded in the provided context,
detecting hallucinations or fabricated information.
"""

from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_eval.metrics.base import Metric, MetricResult
from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class FaithfulnessMetric(Metric):
    """
    Faithfulness metric for RAG evaluation.
    
    Measures how well the answer is grounded in the provided context.
    A faithful answer should only contain information present in the context.
    """
    
    name = "faithfulness"
    description = "Faithfulness score measuring answer grounding in context"
    score_range = (0.0, 1.0)
    
    # Class-level model cache
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 2,
        **kwargs: Any
    ) -> None:
        """
        Initialize Faithfulness metric.
        
        Args:
            model_name: Sentence transformer model for embeddings
            chunk_size: Number of sentences to group for comparison
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """Load the sentence transformer model if not already loaded."""
        if FaithfulnessMetric._model is None or FaithfulnessMetric._model_name != self.model_name:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            FaithfulnessMetric._model = SentenceTransformer(self.model_name)
            FaithfulnessMetric._model_name = self.model_name
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
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
        Compute faithfulness score.
        
        Measures how well each claim in the prediction is supported
        by the provided contexts.
        
        Args:
            prediction: The model's predicted answer
            reference: The expected answer (not used directly)
            query: The original query (not used directly)
            contexts: List of retrieved context passages
            **kwargs: Additional arguments
            
        Returns:
            MetricResult with faithfulness score
        """
        try:
            # Handle empty inputs
            if not prediction:
                return MetricResult(
                    score=0.0,
                    details={"reason": "Empty prediction"}
                )
            
            if not contexts or all(not c for c in contexts):
                return MetricResult(
                    score=0.0,
                    details={"reason": "No context provided"}
                )
            
            # Split prediction into sentences (claims)
            pred_sentences = self._split_into_sentences(prediction)
            if not pred_sentences:
                return MetricResult(
                    score=0.0,
                    details={"reason": "No sentences in prediction"}
                )
            
            # Combine contexts
            combined_context = " ".join(contexts)
            context_sentences = self._split_into_sentences(combined_context)
            
            if not context_sentences:
                return MetricResult(
                    score=0.0,
                    details={"reason": "No sentences in context"}
                )
            
            # Encode all sentences
            all_sentences = pred_sentences + context_sentences
            embeddings = FaithfulnessMetric._model.encode(
                all_sentences,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            pred_embeddings = embeddings[:len(pred_sentences)]
            context_embeddings = embeddings[len(pred_sentences):]
            
            # For each prediction sentence, find max similarity with any context sentence
            sentence_scores = []
            for pred_emb in pred_embeddings:
                max_sim = 0.0
                for ctx_emb in context_embeddings:
                    sim = self._cosine_similarity(pred_emb, ctx_emb)
                    max_sim = max(max_sim, sim)
                # Normalize similarity to 0-1 range
                normalized_sim = (max_sim + 1) / 2
                sentence_scores.append(normalized_sim)
            
            # Average faithfulness across all sentences
            avg_score = sum(sentence_scores) / len(sentence_scores)
            
            return MetricResult(
                score=round(avg_score, 4),
                details={
                    "num_prediction_sentences": len(pred_sentences),
                    "num_context_sentences": len(context_sentences),
                    "sentence_scores": [round(s, 4) for s in sentence_scores],
                    "min_sentence_score": round(min(sentence_scores), 4),
                    "max_sentence_score": round(max(sentence_scores), 4),
                }
            )
            
        except Exception as e:
            return MetricResult(
                score=0.0,
                error=f"Faithfulness computation failed: {str(e)}"
            )
