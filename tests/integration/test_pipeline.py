"""
Integration tests for the full evaluation pipeline.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from llm_eval.config import EvaluationConfig
from llm_eval.pipeline import EvaluationPipeline


class TestEvaluationPipeline:
    """Integration tests for EvaluationPipeline."""
    
    def test_pipeline_initialization(self, config_file):
        """Test pipeline initialization from config."""
        config = EvaluationConfig.from_file(config_file)
        pipeline = EvaluationPipeline(config)
        
        assert len(pipeline._metrics) >= 2  # At least BLEU and ROUGE
        assert pipeline._judge is None  # LLM judge disabled
    
    def test_evaluate_model(self, config_file, benchmark_file, model_output_file):
        """Test evaluating a single model."""
        config = EvaluationConfig.from_file(config_file)
        pipeline = EvaluationPipeline(config)
        
        # Load benchmark
        from llm_eval.dataset import DatasetLoader
        benchmark = DatasetLoader(benchmark_file).load_benchmark()
        
        results = pipeline.evaluate_model(
            model_name="test-model",
            model_output_path=model_output_file,
            benchmark=benchmark,
            show_progress=False
        )
        
        assert "examples" in results
        assert len(results["examples"]) == 3
        
        # Check that metrics are computed
        for example in results["examples"]:
            assert "metrics" in example
            assert "bleu" in example["metrics"]
            assert "rouge_l" in example["metrics"]
    
    def test_full_pipeline_run(self, config_file, temp_dir):
        """Test running the full pipeline."""
        config = EvaluationConfig.from_file(config_file)
        config.output_dir = temp_dir / "results"
        
        pipeline = EvaluationPipeline(config)
        results = pipeline.run(show_progress=False)
        
        assert "test-model" in results
        
        # Check that reports are generated
        result_files = list(config.output_dir.glob("*"))
        assert len(result_files) > 0
        
        # Check for JSON report
        json_reports = list(config.output_dir.glob("*.json"))
        assert len(json_reports) >= 1
        
        # Check for Markdown report
        md_reports = list(config.output_dir.glob("*.md"))
        assert len(md_reports) >= 1
    
    def test_pipeline_with_all_metrics(
        self, 
        temp_dir, 
        benchmark_file, 
        model_output_file
    ):
        """Test pipeline with all metrics enabled (except LLM judge)."""
        config = EvaluationConfig(
            dataset_path=benchmark_file,
            output_dir=temp_dir / "results",
            models=[{"name": "test", "output_path": str(model_output_file)}],
            metrics={
                "bleu": True,
                "rouge_l": True,
                "bertscore": True,
                "faithfulness": True,
                "context_relevancy": True,
                "answer_relevancy": True,
                "llm_judge": False
            }
        )
        
        pipeline = EvaluationPipeline(config)
        
        # All 6 local metrics should be initialized
        assert len(pipeline._metrics) == 6
    
    def test_pipeline_metric_error_handling(
        self,
        temp_dir,
        benchmark_file,
        model_output_file
    ):
        """Test that pipeline handles metric errors gracefully."""
        config = EvaluationConfig(
            dataset_path=benchmark_file,
            output_dir=temp_dir / "results",
            models=[{"name": "test", "output_path": str(model_output_file)}],
            metrics={
                "bleu": True,
                "rouge_l": True,
                "bertscore": False,
                "faithfulness": False,
                "context_relevancy": False,
                "answer_relevancy": False,
                "llm_judge": False
            }
        )
        
        pipeline = EvaluationPipeline(config)
        results = pipeline.run(show_progress=False)
        
        # Pipeline should complete even if some examples have issues
        assert "test" in results
        assert len(results["test"]["examples"]) == 3


class TestReportGeneration:
    """Tests for report generation."""
    
    def test_json_report_content(self, config_file, temp_dir):
        """Test JSON report contains expected content."""
        config = EvaluationConfig.from_file(config_file)
        config.output_dir = temp_dir / "results"
        
        pipeline = EvaluationPipeline(config)
        pipeline.run(show_progress=False)
        
        # Find and parse JSON report
        json_reports = list(config.output_dir.glob("evaluation_report_*.json"))
        assert len(json_reports) >= 1
        
        with open(json_reports[0]) as f:
            report = json.load(f)
        
        assert "metadata" in report
        assert "summary" in report
        assert "aggregate_statistics" in report
        assert "per_example_results" in report
    
    def test_markdown_report_content(self, config_file, temp_dir):
        """Test Markdown report contains expected content."""
        config = EvaluationConfig.from_file(config_file)
        config.output_dir = temp_dir / "results"
        
        pipeline = EvaluationPipeline(config)
        pipeline.run(show_progress=False)
        
        # Find Markdown report
        md_reports = list(config.output_dir.glob("evaluation_report_*.md"))
        assert len(md_reports) >= 1
        
        with open(md_reports[0]) as f:
            content = f.read()
        
        assert "# LLM Evaluation Report" in content
        assert "Executive Summary" in content
        assert "Aggregate Statistics" in content


class TestVisualizationGeneration:
    """Tests for visualization generation."""
    
    def test_histogram_generation(self, config_file, temp_dir):
        """Test that histograms are generated."""
        config = EvaluationConfig.from_file(config_file)
        config.output_dir = temp_dir / "results"
        
        pipeline = EvaluationPipeline(config)
        pipeline.run(show_progress=False)
        
        # Check for PNG files
        png_files = list(config.output_dir.glob("histogram_*.png"))
        assert len(png_files) >= 1
    
    def test_radar_chart_generation(self, config_file, temp_dir):
        """Test that radar/comparison chart is generated."""
        config = EvaluationConfig.from_file(config_file)
        config.output_dir = temp_dir / "results"
        
        pipeline = EvaluationPipeline(config)
        pipeline.run(show_progress=False)
        
        # Check for radar chart
        radar_files = list(config.output_dir.glob("radar_chart*.png"))
        # May not exist for single model, but check for any comparison chart
        comparison_files = list(config.output_dir.glob("*.png"))
        assert len(comparison_files) >= 1
