"""
Anthropic Claude Judge implementation.

Provides LLM-as-a-Judge evaluation using Anthropic's Claude model.
"""

from typing import Any, List, Optional

from anthropic import Anthropic

from llm_eval.config import get_settings
from llm_eval.judges.base import Judge, JudgeResult
from llm_eval.utils.logging import get_logger
from llm_eval.utils.retry import retry_with_backoff

logger = get_logger(__name__)


class AnthropicJudge(Judge):
    """
    LLM-as-a-Judge implementation using Anthropic Claude.

    Provides multi-dimensional evaluation of model responses
    using Claude's reasoning capabilities.
    """

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        custom_rubric: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Anthropic Judge.

        Args:
            model: Anthropic model identifier
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            api_key: Anthropic API key (or from env)
            dimensions: Custom evaluation dimensions
            custom_rubric: Custom evaluation rubric
            **kwargs: Additional options
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            dimensions=dimensions,
            custom_rubric=custom_rubric,
            **kwargs,
        )

        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            settings = get_settings()
            self.api_key = settings.get_api_key("anthropic")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize client
        self.client = Anthropic(api_key=self.api_key)

        logger.info(f"Initialized Anthropic Judge with model: {model}")

    @retry_with_backoff(max_retries=3, min_wait=1.0, max_wait=30.0)
    def _call_llm(self, prompt: str) -> str:
        """
        Make an API call to Anthropic.

        Args:
            prompt: The evaluation prompt

        Returns:
            Response content from Claude
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            system="You are an expert evaluator. Always respond with valid JSON only, no additional text.",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
