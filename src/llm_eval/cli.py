"""
CLI for running LLM evaluations.

Built with Typer - supports run, validate, and list-metrics commands.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_eval import __version__
from llm_eval.config import EvaluationConfig
from llm_eval.pipeline import EvaluationPipeline
from llm_eval.utils.logging import setup_logging, get_logger

# Create Typer app
app = typer.Typer(
    name="llm-eval",
    help="Production-grade LLM evaluation framework for RAG/QA systems",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]LLM Evaluation Framework[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    LLM Evaluation Framework - Production-grade evaluation for RAG/QA systems.
    
    Use 'llm-eval run' to execute evaluations.
    """
    pass


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override output directory from config",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose/debug logging",
    ),
    models: Optional[List[str]] = typer.Option(
        None,
        "--models",
        "-m",
        help="Override models to evaluate (comma-separated names)",
    ),
    metrics: Optional[List[str]] = typer.Option(
        None,
        "--metrics",
        help="Override metrics to compute (comma-separated names)",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress bar (for CI/CD)",
    ),
) -> None:
    """
    Run LLM evaluation pipeline.
    
    Loads configuration, evaluates models, and generates reports.
    
    Example:
        llm-eval run --config examples/config.yaml --output-dir results/
    """
    # Setup logging
    setup_logging(verbose=verbose)
    
    console.print(Panel.fit(
        "[bold blue]LLM Evaluation Framework[/bold blue]\n"
        f"Version: {__version__}",
        title="Starting Evaluation"
    ))
    
    try:
        # Load configuration
        console.print(f"\n[bold]Loading configuration from:[/bold] {config}")
        eval_config = EvaluationConfig.from_file(config)
        
        # Override output directory if specified
        if output_dir:
            eval_config.output_dir = output_dir
            console.print(f"[bold]Output directory:[/bold] {output_dir}")
        
        # Override verbose setting
        if verbose:
            eval_config.verbose = True
        
        # Filter models if specified
        if models:
            model_names = [m.strip() for m in models]
            eval_config.models = [
                m for m in eval_config.models
                if m.name in model_names
            ]
            if not eval_config.models:
                console.print(f"[red]Error:[/red] No matching models found: {model_names}")
                raise typer.Exit(code=1)
            console.print(f"[bold]Evaluating models:[/bold] {model_names}")
        
        # Override metrics if specified
        if metrics:
            metric_names = [m.strip().lower() for m in metrics]
            valid_metrics = {"bleu", "rouge_l", "bertscore", "faithfulness", 
                           "context_relevancy", "answer_relevancy", "llm_judge"}
            
            # Disable all metrics first
            eval_config.metrics.bleu = False
            eval_config.metrics.rouge_l = False
            eval_config.metrics.bertscore = False
            eval_config.metrics.faithfulness = False
            eval_config.metrics.context_relevancy = False
            eval_config.metrics.answer_relevancy = False
            eval_config.metrics.llm_judge = False
            
            # Enable selected metrics
            for m in metric_names:
                if m == "bleu":
                    eval_config.metrics.bleu = True
                elif m == "rouge_l":
                    eval_config.metrics.rouge_l = True
                elif m == "bertscore":
                    eval_config.metrics.bertscore = True
                elif m == "faithfulness":
                    eval_config.metrics.faithfulness = True
                elif m == "context_relevancy":
                    eval_config.metrics.context_relevancy = True
                elif m == "answer_relevancy":
                    eval_config.metrics.answer_relevancy = True
                elif m == "llm_judge":
                    eval_config.metrics.llm_judge = True
                else:
                    console.print(f"[yellow]Warning:[/yellow] Unknown metric: {m}")
            
            console.print(f"[bold]Computing metrics:[/bold] {metric_names}")
        
        # Display configuration summary
        _display_config_summary(eval_config)
        
        # Create and run pipeline
        pipeline = EvaluationPipeline(eval_config)
        results = pipeline.run(show_progress=not no_progress)
        
        # Display summary
        _display_results_summary(results, eval_config.output_dir)
        
        console.print("\n[bold green]✓ Evaluation completed successfully![/bold green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] File not found: {e}")
        raise typer.Exit(code=1)
    
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=1)
    
    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def validate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file to validate",
        exists=True,
    ),
) -> None:
    """
    Validate a configuration file without running evaluation.
    
    Checks for syntax errors and validates all paths and settings.
    """
    console.print(f"[bold]Validating configuration:[/bold] {config}")
    
    try:
        eval_config = EvaluationConfig.from_file(config)
        
        console.print("[green]✓ Configuration is valid![/green]")
        _display_config_summary(eval_config)
        
    except Exception as e:
        console.print(f"[red]✗ Configuration error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def list_metrics() -> None:
    """List all available metrics."""
    from llm_eval.metrics import MetricFactory
    
    table = Table(title="Available Metrics")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description")
    
    metric_info = {
        "bleu": ("Reference-based", "BLEU score measuring n-gram precision"),
        "rouge_l": ("Reference-based", "ROUGE-L score measuring LCS overlap"),
        "bertscore": ("Reference-based", "BERTScore using semantic embeddings"),
        "faithfulness": ("RAG-specific", "Measures answer grounding in context"),
        "context_relevancy": ("RAG-specific", "Measures context-query relevance"),
        "answer_relevancy": ("RAG-specific", "Measures answer-query alignment"),
    }
    
    for name in MetricFactory.list_metrics():
        info = metric_info.get(name, ("Custom", "Custom metric"))
        table.add_row(name, info[0], info[1])
    
    # Add judge info
    table.add_row("llm_judge", "LLM-as-a-Judge", "Multi-dimensional evaluation (coherence, relevance, safety)")
    
    console.print(table)


def _display_config_summary(config: EvaluationConfig) -> None:
    """Display configuration summary."""
    table = Table(title="Configuration Summary", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("Dataset", str(config.dataset_path))
    table.add_row("Output Directory", str(config.output_dir))
    table.add_row("Models", ", ".join(m.name for m in config.models))
    
    enabled_metrics = []
    if config.metrics.bleu:
        enabled_metrics.append("bleu")
    if config.metrics.rouge_l:
        enabled_metrics.append("rouge_l")
    if config.metrics.bertscore:
        enabled_metrics.append("bertscore")
    if config.metrics.faithfulness:
        enabled_metrics.append("faithfulness")
    if config.metrics.context_relevancy:
        enabled_metrics.append("context_relevancy")
    if config.metrics.answer_relevancy:
        enabled_metrics.append("answer_relevancy")
    if config.metrics.llm_judge:
        enabled_metrics.append("llm_judge")
    
    table.add_row("Metrics", ", ".join(enabled_metrics))
    
    if config.metrics.llm_judge:
        table.add_row("Judge Provider", config.judge.provider)
        table.add_row("Judge Model", config.judge.model)
    
    console.print(table)


def _display_results_summary(
    results: dict,
    output_dir: Path
) -> None:
    """Display evaluation results summary."""
    console.print("\n[bold]Results Summary:[/bold]")
    
    for model_name, model_results in results.items():
        examples = model_results.get("examples", [])
        
        # Collect metric averages
        metric_scores = {}
        for example in examples:
            for metric_name, metric_result in example.get("metrics", {}).items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                
                if isinstance(metric_result, dict):
                    score = metric_result.get("score", 0.0)
                else:
                    score = float(metric_result) if metric_result else 0.0
                
                metric_scores[metric_name].append(score)
        
        table = Table(title=f"Model: {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean Score", justify="right")
        table.add_column("Status")
        
        for metric_name, scores in sorted(metric_scores.items()):
            mean_score = sum(scores) / len(scores) if scores else 0.0
            
            if mean_score >= 0.8:
                status = "[green]●[/green]"
            elif mean_score >= 0.6:
                status = "[yellow]●[/yellow]"
            else:
                status = "[red]●[/red]"
            
            table.add_row(metric_name, f"{mean_score:.4f}", status)
        
        console.print(table)
    
    console.print(f"\n[bold]Reports saved to:[/bold] {output_dir}")


if __name__ == "__main__":
    app()
