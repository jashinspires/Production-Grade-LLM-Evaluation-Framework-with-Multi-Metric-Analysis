"""
Evaluation pipeline for LLM outputs.

Handles the coordination between metrics, judges, and report generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from llm_eval.config import EvaluationConfig, get_settings
from llm_eval.dataset import (
    BenchmarkExample,
    DatasetLoader,
    ModelOutput,
    match_outputs_to_benchmark,
)
from llm_eval.judges import Judge, create_judge
from llm_eval.metrics import Metric, MetricFactory
from llm_eval.reporting import JSONReporter, MarkdownReporter, Visualizer
from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """Main pipeline that runs all evaluations and generates reports."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Set up pipeline with given config."""
        self.config = config
        self._metrics: List[Metric] = []
        self._judge: Optional[Judge] = None
        self._initialize_metrics()
        self._initialize_judge()

    def _initialize_metrics(self) -> None:
        """Load metrics based on what's enabled in config."""
        metrics_config = self.config.metrics

        if metrics_config.bleu:
            self._metrics.append(MetricFactory.create("bleu"))

        if metrics_config.rouge_l:
            self._metrics.append(MetricFactory.create("rouge_l"))

        if metrics_config.bertscore:
            self._metrics.append(MetricFactory.create("bertscore"))

        if metrics_config.faithfulness:
            self._metrics.append(MetricFactory.create("faithfulness"))

        if metrics_config.context_relevancy:
            self._metrics.append(MetricFactory.create("context_relevancy"))

        if metrics_config.answer_relevancy:
            self._metrics.append(MetricFactory.create("answer_relevancy"))

        logger.info(f"Initialized {len(self._metrics)} metrics: {[m.name for m in self._metrics]}")

    def _initialize_judge(self) -> None:
        """Set up LLM judge if it's enabled and API key exists."""
        if not self.config.metrics.llm_judge:
            logger.info("LLM judge disabled in configuration")
            return

        try:
            settings = get_settings()
            api_key = settings.get_api_key(self.config.judge.provider)

            if not api_key:
                logger.warning(
                    f"No API key found for {self.config.judge.provider}. "
                    "LLM judge will be skipped."
                )
                return

            self._judge = create_judge(
                provider=self.config.judge.provider,
                model=self.config.judge.model,
                temperature=self.config.judge.temperature,
                max_retries=self.config.judge.max_retries,
                dimensions=self.config.judge.rubric_dimensions,
            )

            logger.info(
                f"Initialized {self.config.judge.provider} judge with model {self.config.judge.model}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize LLM judge: {e}. Judge will be skipped.")
            self._judge = None

    def _compute_metrics(
        self, example: BenchmarkExample, output: ModelOutput
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute all metrics for a single example.

        Args:
            example: Benchmark example
            output: Model output

        Returns:
            Dictionary mapping metric names to results
        """
        results = {}

        for metric in self._metrics:
            try:
                result = metric.compute(
                    prediction=output.prediction,
                    reference=example.expected_answer,
                    query=example.query,
                    contexts=example.retrieved_contexts,
                )
                results[metric.name] = result.to_dict()
            except Exception as e:
                logger.error(f"Error computing {metric.name}: {e}")
                results[metric.name] = {
                    "score": 0.0,
                    "error": str(e),
                }

        return results

    def _run_judge(
        self, example: BenchmarkExample, output: ModelOutput
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run LLM judge evaluation for a single example.

        Args:
            example: Benchmark example
            output: Model output

        Returns:
            Dictionary with judge results
        """
        if not self._judge:
            return {}

        try:
            result = self._judge.evaluate(
                query=example.query,
                answer=output.prediction,
                contexts=example.retrieved_contexts,
                reference=example.expected_answer,
            )

            if result.is_valid:
                # Add individual dimension scores as separate metrics
                judge_results = {}
                for dim, score in result.scores.items():
                    judge_results[dim] = {
                        "score": score,
                        "raw_score": result.raw_scores.get(dim, 0),
                        "reasoning": result.reasoning.get(dim, ""),
                    }
                return judge_results
            else:
                logger.warning(f"Judge evaluation failed: {result.error}")
                return {}

        except Exception as e:
            logger.error(f"Error running judge: {e}")
            return {}

    def evaluate_model(
        self,
        model_name: str,
        model_output_path: Path,
        benchmark: List[BenchmarkExample],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model against the benchmark.

        Args:
            model_name: Name of the model
            model_output_path: Path to model outputs
            benchmark: List of benchmark examples
            show_progress: Whether to show progress bar

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating model: {model_name}")

        # Load model outputs
        output_loader = DatasetLoader(model_output_path)
        outputs = output_loader.load_model_outputs()

        # Match outputs to benchmark
        matched = match_outputs_to_benchmark(benchmark, outputs)

        results = {
            "model_name": model_name,
            "examples": [],
        }

        # Evaluate each example
        progress_context = (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            if show_progress
            else None
        )

        if progress_context:
            with progress_context:
                task = progress_context.add_task(f"Evaluating {model_name}...", total=len(matched))

                for i, (example, output) in enumerate(matched):
                    example_result = self._evaluate_single(example, output, i)
                    results["examples"].append(example_result)
                    progress_context.update(task, advance=1)
        else:
            for i, (example, output) in enumerate(matched):
                example_result = self._evaluate_single(example, output, i)
                results["examples"].append(example_result)

        logger.info(f"Completed evaluation for {model_name}: {len(results['examples'])} examples")

        return results

    def _evaluate_single(
        self, example: BenchmarkExample, output: ModelOutput, index: int
    ) -> Dict[str, Any]:
        """Evaluate a single example."""
        # Compute metrics
        metric_results = self._compute_metrics(example, output)

        # Run judge if available
        judge_results = self._run_judge(example, output)
        metric_results.update(judge_results)

        return {
            "id": index,
            "query": example.query,
            "prediction": output.prediction,
            "reference": example.expected_answer,
            "contexts": example.retrieved_contexts,
            "metrics": metric_results,
        }

    def run(self, show_progress: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Run the full evaluation pipeline.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping model names to results
        """
        logger.info("Starting evaluation pipeline")

        # Load benchmark dataset
        benchmark_loader = DatasetLoader(self.config.dataset_path)
        benchmark = benchmark_loader.load_benchmark()
        logger.info(f"Loaded {len(benchmark)} benchmark examples")

        # Evaluate each model
        all_results = {}

        for model_config in self.config.models:
            results = self.evaluate_model(
                model_name=model_config.name,
                model_output_path=model_config.output_path,
                benchmark=benchmark,
                show_progress=show_progress,
            )
            all_results[model_config.name] = results

        # Generate reports
        self._generate_reports(all_results)

        logger.info("Evaluation pipeline completed successfully")

        return all_results

    def _generate_reports(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Generate all reports and visualizations."""
        output_dir = self.config.output_dir

        # Initialize reporters
        json_reporter = JSONReporter(output_dir)
        md_reporter = MarkdownReporter(output_dir)
        visualizer = Visualizer(output_dir)

        # Generate individual reports for each model
        for model_name, results in all_results.items():
            json_reporter.generate(results, model_name)
            md_reporter.generate(results, model_name)

        # Generate combined reports if multiple models
        if len(all_results) > 1:
            json_reporter.generate_combined(all_results)
            md_reporter.generate_comparison(all_results)

        # Generate visualizations
        visualizer.generate_from_results(all_results)

        logger.info(f"Reports and visualizations saved to: {output_dir}")
