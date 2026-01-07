"""
Answer Relevancy metric implementation.

Measures how well the answer addresses the original query.
"""

from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_eval.metrics.base import Metric, MetricResult
from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class AnswerRelevancyMetric(Metric):
    """
    Answer Relevancy metric for RAG evaluation.

    Measures how well the generated answer addresses and is
    relevant to the original query.
    """

    name = "answer_relevancy"
    description = "Answer relevancy score measuring query-answer alignment"
    score_range = (0.0, 1.0)

    # Class-level model cache
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs: Any) -> None:
        """
        Initialize Answer Relevancy metric.

        Args:
            model_name: Sentence transformer model for embeddings
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Load the sentence transformer model if not already loaded."""
        if (
            AnswerRelevancyMetric._model is None
            or AnswerRelevancyMetric._model_name != self.model_name
        ):
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            AnswerRelevancyMetric._model = SentenceTransformer(self.model_name)
            AnswerRelevancyMetric._model_name = self.model_name

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
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
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute answer relevancy score.

        Measures the semantic similarity between the query and
        the generated answer.

        Args:
            prediction: The model's predicted answer
            reference: The expected answer (used for additional comparison)
            query: The original query
            contexts: Not used for this metric
            **kwargs: Additional arguments

        Returns:
            MetricResult with answer relevancy score
        """
        try:
            # Handle empty inputs
            if not prediction:
                return MetricResult(score=0.0, details={"reason": "Empty prediction"})

            if not query:
                return MetricResult(score=0.0, details={"reason": "No query provided"})

            # Encode query and answer
            texts_to_encode = [query, prediction]
            if reference:
                texts_to_encode.append(reference)

            embeddings = AnswerRelevancyMetric._model.encode(
                texts_to_encode, convert_to_numpy=True, show_progress_bar=False
            )

            query_embedding = embeddings[0]
            answer_embedding = embeddings[1]

            # Calculate query-answer similarity
            qa_sim = self._cosine_similarity(query_embedding, answer_embedding)
            qa_normalized = (qa_sim + 1) / 2

            details = {
                "query_answer_similarity": round(qa_normalized, 4),
            }

            # If reference is provided, also compute reference-answer similarity
            if reference and len(embeddings) > 2:
                ref_embedding = embeddings[2]

                # Query-reference similarity (baseline)
                qr_sim = self._cosine_similarity(query_embedding, ref_embedding)
                qr_normalized = (qr_sim + 1) / 2

                # Answer-reference similarity
                ar_sim = self._cosine_similarity(answer_embedding, ref_embedding)
                ar_normalized = (ar_sim + 1) / 2

                details["query_reference_similarity"] = round(qr_normalized, 4)
                details["answer_reference_similarity"] = round(ar_normalized, 4)

                # Combined score: weighted average of query-answer and answer-reference
                combined_score = 0.6 * qa_normalized + 0.4 * ar_normalized
                details["combined_score"] = round(combined_score, 4)
            else:
                combined_score = qa_normalized

            return MetricResult(score=round(combined_score, 4), details=details)

        except Exception as e:
            return MetricResult(score=0.0, error=f"Answer relevancy computation failed: {str(e)}")
