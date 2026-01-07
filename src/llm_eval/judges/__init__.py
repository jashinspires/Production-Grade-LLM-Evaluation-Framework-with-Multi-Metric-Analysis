"""
LLM-as-a-Judge module for LLM Evaluation Framework.

Provides judge implementations for multi-dimensional evaluation
using various LLM providers.
"""

from llm_eval.judges.base import Judge, JudgeResult
from llm_eval.judges.openai_judge import OpenAIJudge
from llm_eval.judges.anthropic_judge import AnthropicJudge
from llm_eval.judges.groq_judge import GroqJudge

__all__ = [
    "Judge",
    "JudgeResult",
    "OpenAIJudge",
    "AnthropicJudge",
    "GroqJudge",
]


def create_judge(provider: str, **kwargs) -> Judge:
    """
    Factory function to create a judge based on provider.
    
    Args:
        provider: LLM provider name ('openai', 'anthropic', 'groq')
        **kwargs: Configuration options for the judge
        
    Returns:
        Configured judge instance
        
    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        "openai": OpenAIJudge,
        "anthropic": AnthropicJudge,
        "groq": GroqJudge,
    }
    
    provider_lower = provider.lower()
    if provider_lower not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: {list(providers.keys())}"
        )
    
    return providers[provider_lower](**kwargs)
