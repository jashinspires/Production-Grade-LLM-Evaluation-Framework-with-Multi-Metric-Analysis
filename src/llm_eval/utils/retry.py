"""
Retry utilities with exponential backoff for API calls.

Provides decorators and utilities for handling transient failures
in API-based metrics and LLM judge calls.
"""

import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from llm_eval.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Common retryable exceptions
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)

# Try to import API-specific exceptions
try:
    from openai import RateLimitError as OpenAIRateLimitError
    from openai import APIError as OpenAIAPIError
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (OpenAIRateLimitError, OpenAIAPIError)
except ImportError:
    pass

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
    from anthropic import APIError as AnthropicAPIError
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (AnthropicRateLimitError, AnthropicAPIError)
except ImportError:
    pass

try:
    from groq import RateLimitError as GroqRateLimitError
    from groq import APIError as GroqAPIError
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (GroqRateLimitError, GroqAPIError)
except ImportError:
    pass


def retry_with_backoff(
    max_retries: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    if exceptions is None:
        exceptions = RETRYABLE_EXCEPTIONS
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Calculate wait time with exponential backoff and jitter
                        wait_time = min(
                            max_wait,
                            min_wait * (2 ** attempt) + random.uniform(0, 1)
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed. Last error: {e}"
                        )
            
            # If we've exhausted all retries, raise the last exception
            if last_exception:
                raise last_exception
            
            raise RuntimeError("Unexpected error in retry logic")
        
        return wrapper
    
    return decorator


def create_retry_decorator(
    max_retries: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0
) -> Callable:
    """
    Create a tenacity retry decorator with the specified parameters.
    
    Args:
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries
        max_wait: Maximum wait time between retries
        
    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
