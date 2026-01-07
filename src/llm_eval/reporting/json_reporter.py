"""
JSON report generator for evaluation results.

Provides machine-readable output with comprehensive statistics
and per-example breakdowns.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class JSONReporter:
    """
    Generate JSON reports from evaluation results.
    
    Produces machine-readable reports with aggregate statistics
    and detailed per-example results.
    """
    
    def __init__(self, output_dir: Path) -> None:
        """
        Initialize JSON reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate aggregate statistics for a list of scores.
        
        Args:
            scores: List of metric scores
            
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not scores:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        
        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
        
        if not valid_scores:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        
        return {
            "mean": round(float(np.mean(valid_scores)), 4),
            "median": round(float(np.median(valid_scores)), 4),
            "std": round(float(np.std(valid_scores)), 4),
            "min": round(float(np.min(valid_scores)), 4),
            "max": round(float(np.max(valid_scores)), 4),
            "count": len(valid_scores),
        }
    
    def generate(
        self,
        results: Dict[str, Any],
        model_name: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate JSON report from evaluation results.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the evaluated model
            filename: Optional custom filename
            
        Returns:
            Path to the generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{model_name}_{timestamp}.json"
        
        # Build report structure
        report = {
            "metadata": {
                "model_name": model_name,
                "generated_at": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "total_examples": len(results.get("examples", [])),
            },
            "summary": {},
            "aggregate_statistics": {},
            "per_example_results": [],
            "errors": [],
        }
        
        # Collect all metric scores
        metric_scores: Dict[str, List[float]] = {}
        examples = results.get("examples", [])
        
        for example in examples:
            metrics = example.get("metrics", {})
            
            for metric_name, metric_result in metrics.items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                
                if isinstance(metric_result, dict):
                    score = metric_result.get("score", 0.0)
                    error = metric_result.get("error")
                    if error:
                        report["errors"].append({
                            "example_id": example.get("id", "unknown"),
                            "metric": metric_name,
                            "error": error,
                        })
                else:
                    score = float(metric_result) if metric_result else 0.0
                
                metric_scores[metric_name].append(score)
        
        # Calculate aggregate statistics
        for metric_name, scores in metric_scores.items():
            report["aggregate_statistics"][metric_name] = self._calculate_statistics(scores)
        
        # Build summary
        if metric_scores:
            summary_scores = {
                name: stats["mean"]
                for name, stats in report["aggregate_statistics"].items()
            }
            report["summary"] = {
                "overall_average": round(
                    sum(summary_scores.values()) / len(summary_scores), 4
                ) if summary_scores else 0.0,
                "metric_averages": summary_scores,
                "total_errors": len(report["errors"]),
            }
        
        # Add per-example results
        for i, example in enumerate(examples):
            example_result = {
                "id": example.get("id", i),
                "query": example.get("query", ""),
                "prediction": example.get("prediction", ""),
                "reference": example.get("reference", ""),
                "metrics": example.get("metrics", {}),
            }
            report["per_example_results"].append(example_result)
        
        # Write report
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to: {output_path}")
        return output_path
    
    def generate_combined(
        self,
        all_results: Dict[str, Dict[str, Any]],
        filename: str = "combined_evaluation_report.json"
    ) -> Path:
        """
        Generate combined report for multiple models.
        
        Args:
            all_results: Dictionary mapping model names to results
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "models_evaluated": list(all_results.keys()),
            },
            "model_summaries": {},
            "comparison": {},
        }
        
        # Process each model's results
        for model_name, results in all_results.items():
            examples = results.get("examples", [])
            metric_scores: Dict[str, List[float]] = {}
            
            for example in examples:
                metrics = example.get("metrics", {})
                for metric_name, metric_result in metrics.items():
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    
                    if isinstance(metric_result, dict):
                        score = metric_result.get("score", 0.0)
                    else:
                        score = float(metric_result) if metric_result else 0.0
                    
                    metric_scores[metric_name].append(score)
            
            model_summary = {
                "total_examples": len(examples),
                "metrics": {},
            }
            
            for metric_name, scores in metric_scores.items():
                model_summary["metrics"][metric_name] = self._calculate_statistics(scores)
            
            report["model_summaries"][model_name] = model_summary
        
        # Build comparison table
        all_metrics = set()
        for model_summary in report["model_summaries"].values():
            all_metrics.update(model_summary["metrics"].keys())
        
        comparison = {}
        for metric in sorted(all_metrics):
            comparison[metric] = {}
            for model_name, model_summary in report["model_summaries"].items():
                if metric in model_summary["metrics"]:
                    comparison[metric][model_name] = model_summary["metrics"][metric]["mean"]
                else:
                    comparison[metric][model_name] = None
        
        report["comparison"] = comparison
        
        # Write report
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Combined JSON report saved to: {output_path}")
        return output_path
