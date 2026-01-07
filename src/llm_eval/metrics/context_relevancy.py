"""
Context Relevancy metric implementation.

Measures how relevant the retrieved context is to the query.
"""

from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_eval.metrics.base import Metric, MetricResult
from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class ContextRelevancyMetric(Metric):
    """
    Context Relevancy metric for RAG evaluation.
    
    Measures how relevant the retrieved context passages are
    to the original query.
    """
    
    name = "context_relevancy"
    description = "Context relevancy score measuring query-context alignment"
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
        Initialize Context Relevancy metric.
        
        Args:
            model_name: Sentence transformer model for embeddings
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """Load the sentence transformer model if not already loaded."""
        if ContextRelevancyMetric._model is None or ContextRelevancyMetric._model_name != self.model_name:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            ContextRelevancyMetric._model = SentenceTransformer(self.model_name)
            ContextRelevancyMetric._model_name = self.model_name
    
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
        Compute context relevancy score.
        
        Measures the semantic similarity between the query and
        each retrieved context passage.
        
        Args:
            prediction: The model's predicted answer (not used)
            reference: The expected answer (not used)
            query: The original query
            contexts: List of retrieved context passages
            **kwargs: Additional arguments
            
        Returns:
            MetricResult with context relevancy score
        """
        try:
            # Handle empty inputs
            if not query:
                return MetricResult(
                    score=0.0,
                    details={"reason": "No query provided"}
                )
            
            if not contexts or all(not c for c in contexts):
                return MetricResult(
                    score=0.0,
                    details={"reason": "No context provided"}
                )
            
            # Filter empty contexts
            valid_contexts = [c for c in contexts if c and c.strip()]
            if not valid_contexts:
                return MetricResult(
                    score=0.0,
                    details={"reason": "All contexts are empty"}
                )
            
            # Encode query and contexts
            all_texts = [query] + valid_contexts
            embeddings = ContextRelevancyMetric._model.encode(
                all_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            query_embedding = embeddings[0]
            context_embeddings = embeddings[1:]
            
            # Calculate similarity for each context
            context_scores = []
            for ctx_emb in context_embeddings:
                sim = self._cosine_similarity(query_embedding, ctx_emb)
                # Normalize to 0-1 range
                normalized_sim = (sim + 1) / 2
                context_scores.append(normalized_sim)
            
            # Calculate aggregate scores
            avg_score = sum(context_scores) / len(context_scores)
            max_score = max(context_scores)
            min_score = min(context_scores)
            
            return MetricResult(
                score=round(avg_score, 4),
                details={
                    "num_contexts": len(valid_contexts),
                    "context_scores": [round(s, 4) for s in context_scores],
                    "max_context_score": round(max_score, 4),
                    "min_context_score": round(min_score, 4),
                }
            )
            
        except Exception as e:
            return MetricResult(
                score=0.0,
                error=f"Context relevancy computation failed: {str(e)}"
            )
