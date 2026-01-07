"""
Groq Judge implementation.

Provides LLM-as-a-Judge evaluation using Groq's fast inference
with models like Llama 3.1 and Mixtral.
"""

from typing import Any, List, Optional

from groq import Groq

from llm_eval.config import get_settings
from llm_eval.judges.base import Judge, JudgeResult
from llm_eval.utils.logging import get_logger
from llm_eval.utils.retry import retry_with_backoff

logger = get_logger(__name__)


class GroqJudge(Judge):
    """
    LLM-as-a-Judge implementation using Groq.

    Provides multi-dimensional evaluation of model responses
    using Groq's fast inference with open-source models.
    """

    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        custom_rubric: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Groq Judge.

        Args:
            model: Groq model identifier (default: llama-3.1-70b-versatile)
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            api_key: Groq API key (or from env)
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
            self.api_key = settings.get_api_key("groq")

        if not self.api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize client
        self.client = Groq(api_key=self.api_key)

        logger.info(f"Initialized Groq Judge with model: {model}")

    @retry_with_backoff(max_retries=3, min_wait=1.0, max_wait=30.0)
    def _call_llm(self, prompt: str) -> str:
        """
        Make an API call to Groq.

        Args:
            prompt: The evaluation prompt

        Returns:
            Response content from the model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Always respond with valid JSON only, no additional text or explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content
