"""
Unit tests for reporting modules.
"""

import json
from pathlib import Path

import pytest

from llm_eval.reporting.json_reporter import JSONReporter
from llm_eval.reporting.markdown_reporter import MarkdownReporter
from llm_eval.reporting.visualizer import Visualizer


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_generate_report(self, temp_dir):
        """Test generating JSON report."""
        reporter = JSONReporter(temp_dir)

        results = {
            "model_name": "test-model",
            "examples": [
                {
                    "id": 0,
                    "query": "What is AI?",
                    "prediction": "AI is artificial intelligence.",
                    "reference": "Artificial intelligence.",
                    "metrics": {
                        "bleu": {"score": 0.75, "details": {}},
                        "rouge_l": {"score": 0.80, "details": {}},
                    },
                }
            ],
        }

        output_path = reporter.generate(results, "test-model")

        assert output_path.exists()
        assert output_path.suffix == ".json"

        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "summary" in data

    def test_generate_combined_report(self, temp_dir):
        """Test generating combined JSON report."""
        reporter = JSONReporter(temp_dir)

        all_results = {
            "model-a": {
                "model_name": "model-a",
                "examples": [{"metrics": {"bleu": {"score": 0.7}}}],
            },
            "model-b": {
                "model_name": "model-b",
                "examples": [{"metrics": {"bleu": {"score": 0.8}}}],
            },
        }

        output_path = reporter.generate_combined(all_results)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "models" in data


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_generate_report(self, temp_dir):
        """Test generating Markdown report."""
        reporter = MarkdownReporter(temp_dir)

        results = {
            "model_name": "test-model",
            "examples": [
                {
                    "id": 0,
                    "query": "What is AI?",
                    "prediction": "AI is artificial intelligence.",
                    "reference": "Artificial intelligence.",
                    "metrics": {
                        "bleu": {"score": 0.75},
                        "rouge_l": {"score": 0.80},
                    },
                }
            ],
        }

        output_path = reporter.generate(results, "test-model")

        assert output_path.exists()
        assert output_path.suffix == ".md"

        content = output_path.read_text()
        assert "# LLM Evaluation Report" in content
        assert "test-model" in content

    def test_generate_comparison_report(self, temp_dir):
        """Test generating comparison report."""
        reporter = MarkdownReporter(temp_dir)

        all_results = {
            "model-a": {
                "examples": [{"metrics": {"bleu": {"score": 0.7}}}],
            },
            "model-b": {
                "examples": [{"metrics": {"bleu": {"score": 0.8}}}],
            },
        }

        output_path = reporter.generate_comparison(all_results)

        assert output_path.exists()

        content = output_path.read_text()
        assert "Comparison" in content
        assert "model-a" in content
        assert "model-b" in content

    def test_format_score(self, temp_dir):
        """Test score formatting."""
        reporter = MarkdownReporter(temp_dir)

        # High score (green)
        high = reporter._format_score(0.85)
        assert "🟢" in high

        # Medium score (yellow)
        medium = reporter._format_score(0.65)
        assert "🟡" in medium

        # Low score (red)
        low = reporter._format_score(0.35)
        assert "🔴" in low

    def test_calculate_statistics(self, temp_dir):
        """Test statistics calculation."""
        reporter = MarkdownReporter(temp_dir)

        stats = reporter._calculate_statistics([0.5, 0.6, 0.7, 0.8, 0.9])

        assert stats["mean"] == 0.7
        assert stats["min"] == 0.5
        assert stats["max"] == 0.9

    def test_empty_statistics(self, temp_dir):
        """Test statistics with empty list."""
        reporter = MarkdownReporter(temp_dir)

        stats = reporter._calculate_statistics([])

        assert stats["mean"] == 0.0


class TestVisualizer:
    """Tests for Visualizer."""

    def test_generate_histogram(self, temp_dir):
        """Test generating histogram."""
        visualizer = Visualizer(temp_dir)

        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        output_path = visualizer.generate_histogram(scores, "bleu", "test-model")

        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_generate_radar_chart(self, temp_dir):
        """Test generating radar chart."""
        visualizer = Visualizer(temp_dir)

        model_scores = {
            "model-a": {"bleu": 0.7, "rouge_l": 0.8, "bertscore": 0.75},
            "model-b": {"bleu": 0.8, "rouge_l": 0.7, "bertscore": 0.85},
        }

        output_path = visualizer.generate_radar_chart(model_scores)

        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_generate_from_results(self, temp_dir):
        """Test generating all visualizations from results."""
        visualizer = Visualizer(temp_dir)

        all_results = {
            "model-a": {
                "examples": [{"metrics": {"bleu": {"score": 0.7}, "rouge_l": {"score": 0.8}}}],
            },
        }

        visualizer.generate_from_results(all_results)

        # Should have created histogram files
        png_files = list(temp_dir.glob("*.png"))
        assert len(png_files) >= 1
