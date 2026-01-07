"""
LLM Evaluation Framework

A production-grade Python framework for systematically evaluating Large Language Model
applications including RAG pipelines, chatbots, and question-answering systems.
"""

__version__ = "1.0.0"
__author__ = "LLM Eval Team"

from llm_eval.config import EvaluationConfig
from llm_eval.dataset import DatasetLoader
from llm_eval.pipeline import EvaluationPipeline

__all__ = [
    "__version__",
    "EvaluationConfig",
    "EvaluationPipeline",
    "DatasetLoader",
]
