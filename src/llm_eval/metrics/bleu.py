"""
BLEU score metric.

Calculates n-gram precision with brevity penalty using NLTK.
"""

from typing import Any, List, Optional

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from llm_eval.metrics.base import Metric, MetricResult


class BLEUMetric(Metric):
    """BLEU scorer - compares n-gram overlap between prediction and reference."""

    name = "bleu"
    description = "BLEU score measuring n-gram precision"
    score_range = (0.0, 1.0)

    def __init__(
        self,
        max_ngram: int = 4,
        weights: Optional[tuple] = None,
        smoothing: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize BLEU metric.

        Args:
            max_ngram: Maximum n-gram order (1-4)
            weights: Custom weights for each n-gram order
            smoothing: Whether to use smoothing for short texts
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.max_ngram = min(max_ngram, 4)
        self.smoothing = smoothing

        # Set default weights if not provided
        if weights is None:
            # Uniform weights for all n-gram orders
            self.weights = tuple(1.0 / self.max_ngram for _ in range(self.max_ngram))
        else:
            self.weights = weights

        # Download NLTK data if needed
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        # Smoothing function for handling zero counts
        self.smoothing_fn = SmoothingFunction().method1 if smoothing else None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        return nltk.word_tokenize(text.lower())

    def compute(
        self,
        prediction: str,
        reference: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute BLEU score between prediction and reference.

        Args:
            prediction: The model's predicted answer
            reference: The expected/reference answer
            query: Not used for BLEU
            contexts: Not used for BLEU
            **kwargs: Additional arguments (ignored)

        Returns:
            MetricResult with BLEU score
        """
        try:
            # Handle empty inputs
            if not prediction or not reference:
                return MetricResult(score=0.0, details={"reason": "Empty prediction or reference"})

            # Tokenize
            pred_tokens = self._tokenize(prediction)
            ref_tokens = self._tokenize(reference)

            if not pred_tokens or not ref_tokens:
                return MetricResult(score=0.0, details={"reason": "No tokens after tokenization"})

            # Calculate BLEU score
            # Reference should be a list of reference lists
            references = [ref_tokens]
            hypothesis = pred_tokens

            # Adjust weights if prediction is shorter than max_ngram
            effective_weights = self.weights
            if len(pred_tokens) < self.max_ngram:
                n = len(pred_tokens)
                effective_weights = tuple(1.0 / n for _ in range(n))

            score = sentence_bleu(
                references,
                hypothesis,
                weights=effective_weights,
                smoothing_function=self.smoothing_fn,
            )

            # Calculate individual n-gram scores for details
            ngram_scores = {}
            for n in range(1, min(self.max_ngram + 1, len(pred_tokens) + 1)):
                weights_n = tuple(1.0 if i == n - 1 else 0.0 for i in range(n))
                try:
                    ngram_score = sentence_bleu(
                        references,
                        hypothesis,
                        weights=weights_n,
                        smoothing_function=self.smoothing_fn,
                    )
                    ngram_scores[f"bleu_{n}"] = round(ngram_score, 4)
                except Exception:
                    ngram_scores[f"bleu_{n}"] = 0.0

            return MetricResult(
                score=round(score, 4),
                details={
                    "ngram_scores": ngram_scores,
                    "prediction_length": len(pred_tokens),
                    "reference_length": len(ref_tokens),
                },
            )

        except Exception as e:
            return MetricResult(score=0.0, error=f"BLEU computation failed: {str(e)}")

    def compute_batch(
        self,
        predictions: List[str],
        references: List[str],
        queries: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> List[MetricResult]:
        """
        Compute BLEU scores for a batch of examples.

        Uses the default batch implementation since BLEU
        doesn't benefit from batching.
        """
        return super().compute_batch(
            predictions=predictions,
            references=references,
            queries=queries,
            contexts=contexts,
            **kwargs,
        )
