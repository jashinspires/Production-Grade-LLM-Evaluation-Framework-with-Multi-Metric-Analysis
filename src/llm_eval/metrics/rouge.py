"""
ROUGE-L metric using google's rouge-score library.
"""

from typing import Any, List, Optional

from rouge_score import rouge_scorer

from llm_eval.metrics.base import Metric, MetricResult


class ROUGELMetric(Metric):
    """Computes ROUGE-L (longest common subsequence) F1 score."""

    name = "rouge_l"
    description = "ROUGE-L score measuring longest common subsequence"
    score_range = (0.0, 1.0)

    def __init__(self, use_stemmer: bool = False, **kwargs: Any) -> None:
        """
        Initialize ROUGE-L metric.

        Args:
            use_stemmer: Whether to use Porter stemmer for normalization
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.use_stemmer = use_stemmer
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    def compute(
        self,
        prediction: str,
        reference: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute ROUGE-L score between prediction and reference.

        Args:
            prediction: The model's predicted answer
            reference: The expected/reference answer
            query: Not used for ROUGE-L
            contexts: Not used for ROUGE-L
            **kwargs: Additional arguments (ignored)

        Returns:
            MetricResult with ROUGE-L F1 score
        """
        try:
            # Handle empty inputs
            if not prediction or not reference:
                return MetricResult(score=0.0, details={"reason": "Empty prediction or reference"})

            # Calculate ROUGE-L score
            scores = self._scorer.score(reference, prediction)
            rouge_l = scores["rougeL"]

            return MetricResult(
                score=round(rouge_l.fmeasure, 4),
                details={
                    "precision": round(rouge_l.precision, 4),
                    "recall": round(rouge_l.recall, 4),
                    "f1": round(rouge_l.fmeasure, 4),
                },
            )

        except Exception as e:
            return MetricResult(score=0.0, error=f"ROUGE-L computation failed: {str(e)}")

    def compute_batch(
        self,
        predictions: List[str],
        references: List[str],
        queries: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> List[MetricResult]:
        """
        Compute ROUGE-L scores for a batch of examples.

        Uses the default batch implementation.
        """
        return super().compute_batch(
            predictions=predictions,
            references=references,
            queries=queries,
            contexts=contexts,
            **kwargs,
        )
