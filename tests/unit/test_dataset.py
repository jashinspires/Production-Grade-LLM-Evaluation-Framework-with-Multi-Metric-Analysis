"""
Unit tests for dataset loader.
"""

import csv
import json
from pathlib import Path

import pytest

from llm_eval.dataset import (
    BenchmarkExample,
    DatasetLoader,
    ModelOutput,
    match_outputs_to_benchmark,
)


class TestBenchmarkExample:
    """Tests for BenchmarkExample model."""

    def test_valid_example(self):
        """Test creating valid benchmark example."""
        example = BenchmarkExample(
            query="What is AI?",
            expected_answer="AI is artificial intelligence.",
            retrieved_contexts=["AI stands for artificial intelligence."],
        )
        assert example.query == "What is AI?"
        assert len(example.retrieved_contexts) == 1

    def test_empty_contexts(self):
        """Test example with empty contexts."""
        example = BenchmarkExample(
            query="What is AI?",
            expected_answer="AI is artificial intelligence.",
            retrieved_contexts=[],
        )
        assert example.retrieved_contexts == []

    def test_contexts_from_string(self):
        """Test parsing contexts from JSON string."""
        example = BenchmarkExample(
            query="What is AI?",
            expected_answer="AI is artificial intelligence.",
            retrieved_contexts='["context 1", "context 2"]',
        )
        assert len(example.retrieved_contexts) == 2
        assert example.retrieved_contexts[0] == "context 1"

    def test_contexts_from_single_string(self):
        """Test parsing single context string."""
        example = BenchmarkExample(
            query="What is AI?",
            expected_answer="AI is artificial intelligence.",
            retrieved_contexts="This is a single context.",
        )
        assert len(example.retrieved_contexts) == 1


class TestModelOutput:
    """Tests for ModelOutput model."""

    def test_valid_output(self):
        """Test creating valid model output."""
        output = ModelOutput(query="What is AI?", prediction="AI is artificial intelligence.")
        assert output.query == "What is AI?"
        assert output.prediction == "AI is artificial intelligence."

    def test_none_prediction(self):
        """Test that None prediction becomes empty string."""
        output = ModelOutput(query="What is AI?", prediction=None)
        assert output.prediction == ""


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_jsonl_benchmark(self, benchmark_file):
        """Test loading JSONL benchmark."""
        loader = DatasetLoader(benchmark_file)
        examples = loader.load_benchmark()

        assert len(examples) == 3
        assert examples[0].query == "What is the capital of France?"

    def test_load_csv_benchmark(self, temp_dir, sample_benchmark_data):
        """Test loading CSV benchmark."""
        csv_file = temp_dir / "benchmark.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["query", "expected_answer", "retrieved_contexts"]
            )
            writer.writeheader()
            for item in sample_benchmark_data:
                writer.writerow(
                    {
                        "query": item["query"],
                        "expected_answer": item["expected_answer"],
                        "retrieved_contexts": json.dumps(item["retrieved_contexts"]),
                    }
                )

        loader = DatasetLoader(csv_file)
        examples = loader.load_benchmark()

        assert len(examples) == 3

    def test_load_model_outputs(self, model_output_file):
        """Test loading model outputs."""
        loader = DatasetLoader(model_output_file)
        outputs = loader.load_model_outputs()

        assert len(outputs) == 3
        assert outputs[0].query == "What is the capital of France?"

    def test_missing_file(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader("/nonexistent/file.jsonl")

    def test_unsupported_format(self, temp_dir):
        """Test that unsupported format raises error."""
        txt_file = temp_dir / "data.txt"
        txt_file.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            DatasetLoader(txt_file)

    def test_invalid_json(self, temp_dir):
        """Test that invalid JSON raises error."""
        invalid_file = temp_dir / "invalid.jsonl"
        with open(invalid_file, "w") as f:
            f.write("not valid json\n")

        loader = DatasetLoader(invalid_file)
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_benchmark()

    def test_missing_fields(self, temp_dir):
        """Test that missing fields raise error."""
        incomplete_file = temp_dir / "incomplete.jsonl"
        with open(incomplete_file, "w") as f:
            f.write(json.dumps({"query": "test"}) + "\n")  # Missing expected_answer

        loader = DatasetLoader(incomplete_file)
        with pytest.raises(ValueError, match="Missing required fields"):
            loader.load_benchmark()

    def test_len(self, benchmark_file):
        """Test getting dataset length."""
        loader = DatasetLoader(benchmark_file)
        assert len(loader) == 3

    def test_iter_benchmark(self, benchmark_file):
        """Test iterating over benchmark."""
        loader = DatasetLoader(benchmark_file)
        examples = list(loader.iter_benchmark())

        assert len(examples) == 3
        assert all(isinstance(e, BenchmarkExample) for e in examples)


class TestMatchOutputsToBenchmark:
    """Tests for matching outputs to benchmark."""

    def test_matching(self, sample_benchmark_data, sample_model_outputs):
        """Test matching outputs to benchmark examples."""
        benchmark = [BenchmarkExample(**d) for d in sample_benchmark_data]
        outputs = [ModelOutput(**d) for d in sample_model_outputs]

        matched = match_outputs_to_benchmark(benchmark, outputs)

        assert len(matched) == 3
        assert all(isinstance(e, BenchmarkExample) for e, _ in matched)
        assert all(isinstance(o, ModelOutput) for _, o in matched)

    def test_unmatched_examples(self, sample_benchmark_data):
        """Test handling unmatched examples."""
        benchmark = [BenchmarkExample(**d) for d in sample_benchmark_data]
        outputs = []  # No outputs

        matched = match_outputs_to_benchmark(benchmark, outputs)

        assert len(matched) == 3
        # All outputs should be empty predictions
        assert all(o.prediction == "" for _, o in matched)
