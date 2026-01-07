"""
Visualization generator for evaluation results.

Creates histograms, radar charts, and other visualizations
for metric analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150


class Visualizer:
    """
    Generate visualizations from evaluation results.

    Creates histograms for score distributions and radar charts
    for multi-metric comparison.
    """

    # Color palette for models
    COLORS = [
        "#2ecc71",  # Green
        "#3498db",  # Blue
        "#e74c3c",  # Red
        "#9b59b6",  # Purple
        "#f39c12",  # Orange
        "#1abc9c",  # Teal
    ]

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_histogram(
        self, scores: List[float], metric_name: str, model_name: str, filename: Optional[str] = None
    ) -> Path:
        """
        Generate histogram for a single metric.

        Args:
            scores: List of scores for the metric
            metric_name: Name of the metric
            model_name: Name of the model
            filename: Optional custom filename

        Returns:
            Path to the saved visualization
        """
        if filename is None:
            filename = f"histogram_{metric_name}_{model_name}.png"

        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter valid scores
        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]

        if not valid_scores:
            ax.text(
                0.5,
                0.5,
                "No valid scores to display",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            # Create histogram
            ax.hist(valid_scores, bins=20, color=self.COLORS[0], edgecolor="white", alpha=0.8)

            # Add statistics
            mean_score = np.mean(valid_scores)
            median_score = np.median(valid_scores)

            ax.axvline(
                mean_score,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_score:.3f}",
            )
            ax.axvline(
                median_score,
                color="blue",
                linestyle=":",
                linewidth=2,
                label=f"Median: {median_score:.3f}",
            )

            ax.legend(loc="upper right")

        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"{metric_name.upper()} Score Distribution - {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(0, 1)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Histogram saved to: {output_path}")
        return output_path

    def generate_all_histograms(self, results: Dict[str, Any], model_name: str) -> List[Path]:
        """
        Generate histograms for all metrics in results.

        Args:
            results: Evaluation results dictionary
            model_name: Name of the model

        Returns:
            List of paths to saved visualizations
        """
        paths = []

        # Collect metric scores
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

        # Generate histogram for each metric
        for metric_name, scores in metric_scores.items():
            path = self.generate_histogram(scores, metric_name, model_name)
            paths.append(path)

        return paths

    def generate_radar_chart(
        self, model_scores: Dict[str, Dict[str, float]], filename: str = "radar_chart.png"
    ) -> Path:
        """
        Generate radar chart comparing models across metrics.

        Args:
            model_scores: Dict mapping model names to metric scores
            filename: Output filename

        Returns:
            Path to the saved visualization
        """
        if not model_scores:
            logger.warning("No scores provided for radar chart")
            return self.output_dir / filename

        # Get all metrics
        all_metrics = set()
        for scores in model_scores.values():
            all_metrics.update(scores.keys())
        metrics = sorted(list(all_metrics))

        if len(metrics) < 3:
            logger.warning("Radar chart requires at least 3 metrics")
            return self._generate_bar_chart_fallback(model_scores, filename)

        # Number of variables
        n_metrics = len(metrics)

        # Compute angle for each metric
        angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
        angles += angles[:1]  # Complete the circle

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each model
        for i, (model_name, scores) in enumerate(model_scores.items()):
            values = [scores.get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle

            color = self.COLORS[i % len(self.COLORS)]

            ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)

        # Set y-axis
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

        ax.set_title("Model Comparison Across Metrics", size=14, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Radar chart saved to: {output_path}")
        return output_path

    def _generate_bar_chart_fallback(
        self, model_scores: Dict[str, Dict[str, float]], filename: str
    ) -> Path:
        """Generate bar chart when radar chart is not possible."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get all metrics
        all_metrics = set()
        for scores in model_scores.values():
            all_metrics.update(scores.keys())
        metrics = sorted(list(all_metrics))

        # Bar positions
        x = np.arange(len(metrics))
        width = 0.8 / len(model_scores)

        # Plot bars for each model
        for i, (model_name, scores) in enumerate(model_scores.items()):
            values = [scores.get(m, 0) for m in metrics]
            color = self.COLORS[i % len(self.COLORS)]

            ax.bar(x + i * width, values, width, label=model_name, color=color, alpha=0.8)

        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison Across Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(model_scores) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return output_path

    def generate_from_results(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
        """
        Generate all visualizations from evaluation results.

        Args:
            all_results: Dict mapping model names to results

        Returns:
            Dict mapping visualization names to paths
        """
        output_paths = {}

        # Generate histograms for each model
        for model_name, results in all_results.items():
            histogram_paths = self.generate_all_histograms(results, model_name)
            for path in histogram_paths:
                output_paths[path.stem] = path

        # Collect aggregate scores for radar chart
        model_scores: Dict[str, Dict[str, float]] = {}

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

            model_scores[model_name] = {
                name: np.mean(scores) for name, scores in metric_scores.items()
            }

        # Generate radar chart
        if model_scores:
            radar_path = self.generate_radar_chart(model_scores)
            output_paths["radar_chart"] = radar_path

        return output_paths
