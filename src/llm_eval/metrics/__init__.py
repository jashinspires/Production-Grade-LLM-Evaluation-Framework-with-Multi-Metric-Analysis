"""
Metrics module for LLM Evaluation Framework.

Provides metric implementations and factory for metric loading.
"""

from llm_eval.metrics.answer_relevancy import AnswerRelevancyMetric
from llm_eval.metrics.base import Metric, MetricFactory, MetricResult
from llm_eval.metrics.bertscore import BERTScoreMetric
from llm_eval.metrics.bleu import BLEUMetric
from llm_eval.metrics.context_relevancy import ContextRelevancyMetric
from llm_eval.metrics.faithfulness import FaithfulnessMetric
from llm_eval.metrics.rouge import ROUGELMetric

__all__ = [
    "Metric",
    "MetricResult",
    "MetricFactory",
    "BLEUMetric",
    "ROUGELMetric",
    "BERTScoreMetric",
    "FaithfulnessMetric",
    "ContextRelevancyMetric",
    "AnswerRelevancyMetric",
]

# Register all metrics with the factory
MetricFactory.register("bleu", BLEUMetric)
MetricFactory.register("rouge_l", ROUGELMetric)
MetricFactory.register("bertscore", BERTScoreMetric)
MetricFactory.register("faithfulness", FaithfulnessMetric)
MetricFactory.register("context_relevancy", ContextRelevancyMetric)
MetricFactory.register("answer_relevancy", AnswerRelevancyMetric)
