"""
Dataset loading and validation for LLM Evaluation Framework.

This module provides utilities for loading benchmark datasets and model outputs
in JSONL and CSV formats with proper validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class BenchmarkExample(BaseModel):
    """A single example from the benchmark dataset."""

    query: str = Field(..., min_length=1, description="The input query/question")
    expected_answer: str = Field(..., description="The expected/reference answer")
    retrieved_contexts: List[str] = Field(
        default_factory=list, description="List of retrieved context passages"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the example"
    )

    @field_validator("retrieved_contexts", mode="before")
    @classmethod
    def parse_contexts(cls, v: Any) -> List[str]:
        """Parse contexts from various formats."""
        if v is None:
            return []
        if isinstance(v, str):
            # Try to parse as JSON list
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(c) for c in parsed]
            except json.JSONDecodeError:
                # Treat as single context
                return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(c) for c in v]
        return []


class ModelOutput(BaseModel):
    """A model's output for a single example."""

    query: str = Field(..., description="The input query (for matching)")
    prediction: str = Field(..., description="The model's predicted answer")
    model_name: Optional[str] = Field(None, description="Name of the model")

    @field_validator("prediction", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        """Ensure prediction is a string."""
        if v is None:
            return ""
        return str(v)


class DatasetLoader:
    """
    Loader for benchmark datasets and model outputs.

    Supports JSONL and CSV formats with automatic format detection
    and validation of required fields.
    """

    REQUIRED_BENCHMARK_FIELDS = {"query", "expected_answer"}
    REQUIRED_OUTPUT_FIELDS = {"query", "prediction"}

    def __init__(self, path: Union[str, Path]):
        """
        Initialize the dataset loader.

        Args:
            path: Path to the dataset file (JSONL or CSV)
        """
        self.path = Path(path)
        self._validate_path()
        self._format = self._detect_format()

    def _validate_path(self) -> None:
        """Validate that the dataset file exists."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")

    def _detect_format(self) -> str:
        """Detect the format of the dataset file."""
        suffix = self.path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return "jsonl"
        elif suffix == ".csv":
            return "csv"
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. " "Supported formats: .jsonl, .json, .csv"
            )

    def load_benchmark(self) -> List[BenchmarkExample]:
        """
        Load and validate the benchmark dataset.

        Returns:
            List of validated BenchmarkExample instances

        Raises:
            ValueError: If required fields are missing
        """
        if self._format == "jsonl":
            return self._load_jsonl_benchmark()
        else:
            return self._load_csv_benchmark()

    def _load_jsonl_benchmark(self) -> List[BenchmarkExample]:
        """Load benchmark from JSONL format."""
        examples = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._validate_fields(data, self.REQUIRED_BENCHMARK_FIELDS, line_num)
                    examples.append(BenchmarkExample(**data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        return examples

    def _load_csv_benchmark(self) -> List[BenchmarkExample]:
        """Load benchmark from CSV format."""
        df = pd.read_csv(self.path)

        # Validate required columns
        missing = self.REQUIRED_BENCHMARK_FIELDS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in CSV: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        examples = []
        for idx, row in df.iterrows():
            data = row.to_dict()
            examples.append(BenchmarkExample(**data))

        return examples

    def load_model_outputs(self) -> List[ModelOutput]:
        """
        Load and validate model outputs.

        Returns:
            List of validated ModelOutput instances
        """
        if self._format == "jsonl":
            return self._load_jsonl_outputs()
        else:
            return self._load_csv_outputs()

    def _load_jsonl_outputs(self) -> List[ModelOutput]:
        """Load model outputs from JSONL format."""
        outputs = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._validate_fields(data, self.REQUIRED_OUTPUT_FIELDS, line_num)
                    outputs.append(ModelOutput(**data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        return outputs

    def _load_csv_outputs(self) -> List[ModelOutput]:
        """Load model outputs from CSV format."""
        df = pd.read_csv(self.path)

        # Validate required columns
        missing = self.REQUIRED_OUTPUT_FIELDS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in CSV: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        outputs = []
        for idx, row in df.iterrows():
            data = row.to_dict()
            outputs.append(ModelOutput(**data))

        return outputs

    def _validate_fields(self, data: Dict[str, Any], required: set, line_num: int) -> None:
        """Validate that required fields are present."""
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields at line {line_num}: {missing}")

    def iter_benchmark(self) -> Iterator[BenchmarkExample]:
        """
        Iterate over benchmark examples (memory-efficient for large files).

        Yields:
            BenchmarkExample instances one at a time
        """
        if self._format == "jsonl":
            with open(self.path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    yield BenchmarkExample(**data)
        else:
            # For CSV, load all at once (pandas doesn't support efficient iteration)
            for example in self.load_benchmark():
                yield example

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self._format == "jsonl":
            with open(self.path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        else:
            df = pd.read_csv(self.path)
            return len(df)


def match_outputs_to_benchmark(
    benchmark: List[BenchmarkExample], outputs: List[ModelOutput]
) -> List[tuple]:
    """
    Match model outputs to benchmark examples by query.

    Args:
        benchmark: List of benchmark examples
        outputs: List of model outputs

    Returns:
        List of (example, output) tuples
    """
    # Create lookup by query
    output_lookup = {o.query: o for o in outputs}

    matched = []
    for example in benchmark:
        output = output_lookup.get(example.query)
        if output:
            matched.append((example, output))
        else:
            # Create empty output for unmatched examples
            matched.append((example, ModelOutput(query=example.query, prediction="")))

    return matched
