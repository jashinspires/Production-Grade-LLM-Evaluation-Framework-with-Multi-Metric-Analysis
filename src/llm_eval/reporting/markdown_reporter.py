"""
Markdown report generator for evaluation results.

Provides human-readable output with formatted tables,
statistics, and insights.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class MarkdownReporter:
    """
    Generate Markdown reports from evaluation results.

    Produces human-readable reports with formatted tables
    and detailed analysis.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize Markdown reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate aggregate statistics for scores."""
        if not scores:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
        if not valid_scores:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": round(float(np.mean(valid_scores)), 4),
            "median": round(float(np.median(valid_scores)), 4),
            "std": round(float(np.std(valid_scores)), 4),
            "min": round(float(np.min(valid_scores)), 4),
            "max": round(float(np.max(valid_scores)), 4),
        }

    def _format_score(self, score: float) -> str:
        """Format a score with color indicator."""
        if score >= 0.8:
            return f"**{score:.4f}** 🟢"
        elif score >= 0.6:
            return f"{score:.4f} 🟡"
        elif score >= 0.4:
            return f"{score:.4f} 🟠"
        else:
            return f"{score:.4f} 🔴"

    def generate(
        self, results: Dict[str, Any], model_name: str, filename: Optional[str] = None
    ) -> Path:
        """
        Generate Markdown report from evaluation results.

        Args:
            results: Evaluation results dictionary
            model_name: Name of the evaluated model
            filename: Optional custom filename

        Returns:
            Path to the generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{model_name}_{timestamp}.md"

        lines = []

        # Header
        lines.append(f"# LLM Evaluation Report: {model_name}")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Framework Version**: 1.0.0")
        lines.append("")

        # Collect metric statistics
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

        # Summary Section
        lines.append("## 📊 Executive Summary")
        lines.append("")
        lines.append(f"- **Total Examples Evaluated**: {len(examples)}")

        if metric_scores:
            avg_scores = {name: np.mean(scores) for name, scores in metric_scores.items()}
            overall_avg = np.mean(list(avg_scores.values()))
            lines.append(f"- **Overall Average Score**: {self._format_score(overall_avg)}")
        lines.append("")

        # Aggregate Statistics Table
        lines.append("## 📈 Aggregate Statistics")
        lines.append("")
        lines.append("| Metric | Mean | Median | Std Dev | Min | Max |")
        lines.append("|--------|------|--------|---------|-----|-----|")

        for metric_name, scores in sorted(metric_scores.items()):
            stats = self._calculate_statistics(scores)
            lines.append(
                f"| {metric_name} | {self._format_score(stats['mean'])} | "
                f"{stats['median']:.4f} | {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} |"
            )

        lines.append("")

        # Metric Descriptions
        lines.append("## 📖 Metric Descriptions")
        lines.append("")

        metric_descriptions = {
            "bleu": "BLEU score measuring n-gram precision between prediction and reference.",
            "rouge_l": "ROUGE-L score measuring longest common subsequence overlap.",
            "bertscore": "BERTScore measuring semantic similarity using embeddings.",
            "faithfulness": "Measures how well the answer is grounded in the provided context.",
            "context_relevancy": "Measures how relevant the retrieved context is to the query.",
            "answer_relevancy": "Measures how well the answer addresses the original query.",
            "coherence": "LLM Judge: Is the response well-structured and logical?",
            "relevance": "LLM Judge: Does the response address the query?",
            "safety": "LLM Judge: Is the response free from harmful content?",
        }

        for metric_name in sorted(metric_scores.keys()):
            desc = metric_descriptions.get(metric_name, "Custom metric.")
            lines.append(f"- **{metric_name}**: {desc}")

        lines.append("")

        # Per-Example Results (sample)
        lines.append("## 📝 Sample Results")
        lines.append("")
        lines.append("Showing first 5 examples:")
        lines.append("")

        for i, example in enumerate(examples[:5]):
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append(f"**Query**: {example.get('query', 'N/A')[:200]}...")
            lines.append("")
            lines.append(f"**Prediction**: {example.get('prediction', 'N/A')[:200]}...")
            lines.append("")

            metrics = example.get("metrics", {})
            if metrics:
                lines.append("| Metric | Score |")
                lines.append("|--------|-------|")
                for metric_name, metric_result in sorted(metrics.items()):
                    if isinstance(metric_result, dict):
                        score = metric_result.get("score", 0.0)
                    else:
                        score = float(metric_result) if metric_result else 0.0
                    lines.append(f"| {metric_name} | {self._format_score(score)} |")

            lines.append("")
            lines.append("---")
            lines.append("")

        # Error Summary
        errors = []
        for example in examples:
            metrics = example.get("metrics", {})
            for metric_name, metric_result in metrics.items():
                if isinstance(metric_result, dict) and metric_result.get("error"):
                    errors.append(
                        {
                            "example": example.get("id", "unknown"),
                            "metric": metric_name,
                            "error": metric_result["error"],
                        }
                    )

        if errors:
            lines.append("## ⚠️ Errors Encountered")
            lines.append("")
            lines.append(f"Total errors: {len(errors)}")
            lines.append("")

            for err in errors[:10]:
                lines.append(f"- **{err['metric']}** on example {err['example']}: {err['error']}")

            if len(errors) > 10:
                lines.append(f"- ... and {len(errors) - 10} more errors")

            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by LLM Evaluation Framework*")

        # Write report
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved to: {output_path}")
        return output_path

    def generate_comparison(
        self, all_results: Dict[str, Dict[str, Any]], filename: str = "comparison_report.md"
    ) -> Path:
        """
        Generate comparison report for multiple models.

        Args:
            all_results: Dictionary mapping model names to results
            filename: Output filename

        Returns:
            Path to the generated report
        """
        lines = []

        # Header
        lines.append("# LLM Evaluation Comparison Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Models Compared**: {', '.join(all_results.keys())}")
        lines.append("")

        # Collect all metrics across models
        all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

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

            for metric_name, scores in metric_scores.items():
                if metric_name not in all_stats:
                    all_stats[metric_name] = {}
                all_stats[metric_name][model_name] = self._calculate_statistics(scores)

        # Comparison Table
        lines.append("## 📊 Model Comparison")
        lines.append("")

        model_names = list(all_results.keys())
        header = "| Metric | " + " | ".join(model_names) + " |"
        separator = "|--------|" + "|".join(["------" for _ in model_names]) + "|"

        lines.append(header)
        lines.append(separator)

        for metric_name in sorted(all_stats.keys()):
            row = f"| {metric_name} |"
            for model_name in model_names:
                if model_name in all_stats[metric_name]:
                    score = all_stats[metric_name][model_name]["mean"]
                    row += f" {self._format_score(score)} |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")

        # Best Model Per Metric
        lines.append("## 🏆 Best Model Per Metric")
        lines.append("")

        for metric_name in sorted(all_stats.keys()):
            best_model = None
            best_score = -1

            for model_name, stats in all_stats[metric_name].items():
                if stats["mean"] > best_score:
                    best_score = stats["mean"]
                    best_model = model_name

            if best_model:
                lines.append(f"- **{metric_name}**: {best_model} ({best_score:.4f})")

        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by LLM Evaluation Framework*")

        # Write report
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Comparison report saved to: {output_path}")
        return output_path
