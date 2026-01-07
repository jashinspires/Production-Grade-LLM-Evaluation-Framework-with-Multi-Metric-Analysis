"""
Unit tests for CLI interface.
"""

from typer.testing import CliRunner

from llm_eval.cli import app

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "LLM Evaluation Framework" in result.stdout

    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "llm-eval" in result.stdout.lower() or "evaluation" in result.stdout.lower()

    def test_run_help(self):
        """Test run --help."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()  # Handle ANSI color codes

    def test_validate_help(self):
        """Test validate --help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()  # Handle ANSI color codes

    def test_list_metrics(self):
        """Test list-metrics command."""
        result = runner.invoke(app, ["list-metrics"])
        assert result.exit_code == 0
        assert "bleu" in result.stdout.lower()
        assert "rouge" in result.stdout.lower()

    def test_validate_valid_config(self, config_file):
        """Test validating a valid config file."""
        result = runner.invoke(app, ["validate", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_missing_config(self):
        """Test validating missing config file."""
        result = runner.invoke(app, ["validate", "--config", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_run_missing_config(self):
        """Test run with missing config file."""
        result = runner.invoke(app, ["run", "--config", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_run_with_valid_config(self, config_file, temp_dir):
        """Test run with valid config."""
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(config_file),
                "--output-dir",
                str(temp_dir / "cli_results"),
                "--no-progress",
            ],
        )
        assert result.exit_code == 0
        assert "complete" in result.stdout.lower() or "success" in result.stdout.lower()

    def test_run_with_metrics_override(self, config_file, temp_dir):
        """Test run with metrics override."""
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(config_file),
                "--output-dir",
                str(temp_dir / "cli_results"),
                "--metrics",
                "bleu",
                "--no-progress",
            ],
        )
        assert result.exit_code == 0

    def test_run_with_verbose(self, config_file, temp_dir):
        """Test run with verbose flag."""
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(config_file),
                "--output-dir",
                str(temp_dir / "cli_results"),
                "--verbose",
                "--no-progress",
            ],
        )
        assert result.exit_code == 0


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_display_config_summary(self, config_file):
        """Test config summary display."""
        from llm_eval.cli import _display_config_summary
        from llm_eval.config import EvaluationConfig

        config = EvaluationConfig.from_file(config_file)
        # Should not raise
        _display_config_summary(config)

    def test_display_results_summary(self, temp_dir):
        """Test results summary display."""
        from llm_eval.cli import _display_results_summary

        results = {
            "test-model": {
                "examples": [
                    {
                        "metrics": {
                            "bleu": {"score": 0.75},
                            "rouge_l": {"score": 0.80},
                        }
                    }
                ]
            }
        }
        # Should not raise
        _display_results_summary(results, temp_dir)
