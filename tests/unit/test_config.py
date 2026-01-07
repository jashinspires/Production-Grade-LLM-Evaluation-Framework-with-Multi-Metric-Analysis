"""
Unit tests for configuration system.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from llm_eval.config import (
    EvaluationConfig,
    ModelConfig,
    MetricsConfig,
    JudgeConfig,
    EnvironmentSettings,
)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_valid_model_config(self, model_output_file):
        """Test valid model configuration."""
        config = ModelConfig(
            name="test-model",
            output_path=model_output_file
        )
        assert config.name == "test-model"
        assert config.output_path == model_output_file
    
    def test_invalid_output_path(self):
        """Test that invalid output path raises error."""
        with pytest.raises(ValueError, match="not found"):
            ModelConfig(
                name="test-model",
                output_path=Path("/nonexistent/path.jsonl")
            )


class TestMetricsConfig:
    """Tests for MetricsConfig."""
    
    def test_default_metrics(self):
        """Test default metrics are enabled."""
        config = MetricsConfig()
        assert config.bleu is True
        assert config.rouge_l is True
        assert config.bertscore is True
        assert config.faithfulness is True
        assert config.llm_judge is True
    
    def test_custom_metrics(self):
        """Test custom metrics configuration."""
        config = MetricsConfig(
            bleu=True,
            rouge_l=False,
            bertscore=False,
            faithfulness=False
        )
        assert config.bleu is True
        assert config.rouge_l is False


class TestJudgeConfig:
    """Tests for JudgeConfig."""
    
    def test_default_judge_config(self):
        """Test default judge configuration."""
        config = JudgeConfig()
        assert config.provider == "groq"
        assert config.temperature == 0.0
        assert "coherence" in config.rubric_dimensions
    
    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Invalid provider"):
            JudgeConfig(provider="invalid_provider")
    
    def test_valid_providers(self):
        """Test all valid providers."""
        for provider in ["openai", "anthropic", "groq"]:
            config = JudgeConfig(provider=provider)
            assert config.provider == provider


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""
    
    def test_from_yaml_file(self, config_file):
        """Test loading from YAML file."""
        config = EvaluationConfig.from_file(config_file)
        assert config.models[0].name == "test-model"
        assert config.metrics.bleu is True
    
    def test_from_json_file(self, temp_dir, sample_config):
        """Test loading from JSON file."""
        json_file = temp_dir / "config.json"
        with open(json_file, "w") as f:
            json.dump(sample_config, f)
        
        config = EvaluationConfig.from_file(json_file)
        assert config.models[0].name == "test-model"
    
    def test_invalid_file_format(self, temp_dir):
        """Test that invalid file format raises error."""
        txt_file = temp_dir / "config.txt"
        txt_file.touch()
        
        with pytest.raises(ValueError, match="Unsupported config format"):
            EvaluationConfig.from_file(txt_file)
    
    def test_missing_file(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            EvaluationConfig.from_file("/nonexistent/config.yaml")
    
    def test_output_dir_creation(self, temp_dir, benchmark_file, model_output_file):
        """Test that output directory is created."""
        output_dir = temp_dir / "new_output_dir"
        
        config = EvaluationConfig(
            dataset_path=benchmark_file,
            output_dir=output_dir,
            models=[ModelConfig(name="test", output_path=model_output_file)]
        )
        
        assert output_dir.exists()
    
    def test_to_dict(self, config_file):
        """Test converting config to dictionary."""
        config = EvaluationConfig.from_file(config_file)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "dataset_path" in config_dict
        assert "models" in config_dict


class TestEnvironmentSettings:
    """Tests for EnvironmentSettings."""
    
    def test_get_api_key(self):
        """Test getting API key."""
        import os
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        
        settings = EnvironmentSettings()
        key = settings.get_api_key("openai")
        
        assert key == "test-openai-key"
    
    def test_get_api_key_missing(self):
        """Test getting missing API key."""
        import os
        if "NONEXISTENT_KEY" in os.environ:
            del os.environ["NONEXISTENT_KEY"]
        
        settings = EnvironmentSettings()
        key = settings.get_api_key("nonexistent")
        
        assert key is None
    
    def test_validate_api_key_missing(self):
        """Test validation raises error for missing key."""
        import os
        # Temporarily remove key
        old_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        settings = EnvironmentSettings()
        
        with pytest.raises(ValueError, match="API key.*not found"):
            settings.validate_api_key("openai")
        
        # Restore key
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key
