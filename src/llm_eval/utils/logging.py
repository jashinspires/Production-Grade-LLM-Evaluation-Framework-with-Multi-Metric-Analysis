"""
Logging configuration for LLM Evaluation Framework.

Provides configurable logging with rich formatting for console output.
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Default format for file logging
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Console for rich output
console = Console()


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, verbose: bool = False
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        verbose: If True, set level to DEBUG
    """
    if verbose:
        level = "DEBUG"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("llm_eval")
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
    )
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Name of the module (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"llm_eval.{name}")
