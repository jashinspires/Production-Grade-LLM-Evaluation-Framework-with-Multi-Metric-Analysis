"""
Config models for the evaluation framework.

Uses Pydantic for validation, supports YAML and JSON files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Config for a single model to evaluate."""

    name: str = Field(..., description="Unique identifier for the model")
    output_path: Path = Field(..., description="Path to model outputs (JSONL/CSV)")
    description: Optional[str] = Field(None, description="Model description")

    @field_validator("output_path", mode="before")
    @classmethod
    def validate_output_path(cls, v: Any) -> Path:
        """Validate and convert output path."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model output file not found: {path}")
        return path


class MetricsConfig(BaseModel):
    """Which metrics to run."""

    # Reference-based metrics
    bleu: bool = Field(True, description="Compute BLEU score")
    rouge_l: bool = Field(True, description="Compute ROUGE-L score")
    bertscore: bool = Field(True, description="Compute BERTScore")

    # RAG-specific metrics
    faithfulness: bool = Field(True, description="Compute faithfulness score")
    context_relevancy: bool = Field(True, description="Compute context relevancy")
    answer_relevancy: bool = Field(True, description="Compute answer relevancy")

    # LLM-as-a-Judge
    llm_judge: bool = Field(True, description="Use LLM-as-a-Judge evaluation")


class JudgeConfig(BaseModel):
    """Settings for LLM-as-a-Judge."""

    provider: str = Field("groq", description="LLM provider: 'openai', 'anthropic', or 'groq'")
    model: str = Field("llama-3.3-70b-versatile", description="Model identifier for the judge")
    temperature: float = Field(
        0.0, ge=0.0, le=2.0, description="Sampling temperature for judge responses"
    )
    max_retries: int = Field(3, ge=1, le=10, description="Maximum retry attempts for API calls")
    rubric_dimensions: List[str] = Field(
        default=["coherence", "relevance", "safety"],
        description="Evaluation dimensions for the judge",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = {"openai", "anthropic", "groq"}
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of: {valid_providers}")
        return v.lower()


class EvaluationConfig(BaseModel):
    """Main config model for running evaluations."""

    # Dataset configuration
    dataset_path: Path = Field(..., description="Path to benchmark dataset (JSONL/CSV)")
    output_dir: Path = Field(Path("results"), description="Directory for output files")

    # Models to evaluate
    models: List[ModelConfig] = Field(..., min_length=1, description="Models to evaluate")

    # Metrics configuration
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    # Judge configuration
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    # Execution options
    verbose: bool = Field(False, description="Enable verbose logging")
    batch_size: int = Field(32, ge=1, description="Batch size for processing")
    num_workers: int = Field(1, ge=1, description="Number of parallel workers")

    @field_validator("dataset_path", mode="before")
    @classmethod
    def validate_dataset_path(cls, v: Any) -> Path:
        """Validate and convert dataset path."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Dataset file not found: {path}")
        return path

    @field_validator("output_dir", mode="before")
    @classmethod
    def create_output_dir(cls, v: Any) -> Path:
        """Create output directory if it doesn't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "EvaluationConfig":
        """
        Load configuration from a YAML or JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            EvaluationConfig instance

        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                import json

                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {path.suffix}. " "Use .yaml, .yml, or .json"
                )

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(mode="json")


class EnvironmentSettings(BaseSettings):
    """Loads API keys and settings from environment/.env file."""

    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = Field(None, alias="GROQ_API_KEY")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for the specified provider.

        Args:
            provider: LLM provider name

        Returns:
            API key if available, None otherwise
        """
        key_mapping = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
        }
        return key_mapping.get(provider.lower())

    def validate_api_key(self, provider: str) -> None:
        """
        Validate that an API key is available for the provider.

        Args:
            provider: LLM provider name

        Raises:
            ValueError: If API key is not configured
        """
        key = self.get_api_key(provider)
        if not key:
            env_var = f"{provider.upper()}_API_KEY"
            raise ValueError(
                f"API key for {provider} not found. " f"Please set {env_var} environment variable."
            )


# Global settings instance
_settings: Optional[EnvironmentSettings] = None


def get_settings() -> EnvironmentSettings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = EnvironmentSettings()
    return _settings
