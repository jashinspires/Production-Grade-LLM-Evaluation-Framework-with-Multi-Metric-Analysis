"""
Pytest configuration and fixtures for LLM Evaluation Framework tests.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_benchmark_data() -> List[Dict]:
    """Sample benchmark data for testing."""
    return [
        {
            "query": "What is the capital of France?",
            "expected_answer": "The capital of France is Paris.",
            "retrieved_contexts": [
                "Paris is the capital and largest city of France.",
                "France is a country in Western Europe.",
            ],
        },
        {
            "query": "Who wrote Romeo and Juliet?",
            "expected_answer": "William Shakespeare wrote Romeo and Juliet.",
            "retrieved_contexts": ["Romeo and Juliet is a tragedy written by William Shakespeare."],
        },
        {
            "query": "What is photosynthesis?",
            "expected_answer": "Photosynthesis is the process by which plants convert sunlight into glucose.",
            "retrieved_contexts": [
                "Photosynthesis is a biological process used by plants to convert light energy."
            ],
        },
    ]


@pytest.fixture
def sample_model_outputs() -> List[Dict]:
    """Sample model outputs for testing."""
    return [
        {
            "query": "What is the capital of France?",
            "prediction": "Paris is the capital of France.",
        },
        {
            "query": "Who wrote Romeo and Juliet?",
            "prediction": "Shakespeare wrote Romeo and Juliet.",
        },
        {
            "query": "What is photosynthesis?",
            "prediction": "Photosynthesis is how plants make food from sunlight.",
        },
    ]


@pytest.fixture
def benchmark_file(temp_dir, sample_benchmark_data) -> Path:
    """Create a benchmark JSONL file."""
    filepath = temp_dir / "benchmark.jsonl"
    with open(filepath, "w") as f:
        for item in sample_benchmark_data:
            f.write(json.dumps(item) + "\n")
    return filepath


@pytest.fixture
def model_output_file(temp_dir, sample_model_outputs) -> Path:
    """Create a model output JSONL file."""
    filepath = temp_dir / "outputs.jsonl"
    with open(filepath, "w") as f:
        for item in sample_model_outputs:
            f.write(json.dumps(item) + "\n")
    return filepath


@pytest.fixture
def sample_config(temp_dir, benchmark_file, model_output_file) -> Dict:
    """Sample configuration for testing."""
    return {
        "dataset_path": str(benchmark_file),
        "output_dir": str(temp_dir / "results"),
        "models": [{"name": "test-model", "output_path": str(model_output_file)}],
        "metrics": {
            "bleu": True,
            "rouge_l": True,
            "bertscore": False,  # Disable for faster tests
            "faithfulness": False,
            "context_relevancy": False,
            "answer_relevancy": False,
            "llm_judge": False,
        },
        "judge": {"provider": "groq", "model": "llama-3.1-70b-versatile", "temperature": 0.0},
    }


@pytest.fixture
def config_file(temp_dir, sample_config) -> Path:
    """Create a config YAML file."""
    import yaml

    filepath = temp_dir / "config.yaml"
    with open(filepath, "w") as f:
        yaml.dump(sample_config, f)
    return filepath


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "coherence": {"score": 4, "reasoning": "Well structured"},
                        "relevance": {"score": 5, "reasoning": "Directly addresses query"},
                        "safety": {"score": 5, "reasoning": "No harmful content"},
                        "overall_assessment": "Good response",
                    }
                )
            )
        )
    ]
    return mock_response


@pytest.fixture
def mock_groq_response():
    """Mock Groq API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "coherence": {"score": 4, "reasoning": "Clear and logical"},
                        "relevance": {"score": 4, "reasoning": "Addresses the question"},
                        "safety": {"score": 5, "reasoning": "Appropriate content"},
                        "overall_assessment": "Solid response",
                    }
                )
            )
        )
    ]
    return mock_response


@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment for tests."""
    # Ensure no real API calls are made
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("GROQ_API_KEY", "test-key")
    yield
