# LLM Evaluation Framework - Architecture

This document provides a detailed overview of the system architecture, design decisions, and component interactions.

## Table of Contents

- [System Overview](#system-overview)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Extensibility](#extensibility)

## System Overview

The LLM Evaluation Framework is designed as a modular, extensible system for evaluating Large Language Model outputs. It follows clean architecture principles with clear separation of concerns.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Typer CLI (cli.py)                                                     │  │
│  │  - Command parsing       - Progress display      - Exit codes          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────┤
│                           Orchestration Layer                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Pipeline (pipeline.py)                                                 │  │
│  │  - Coordinates evaluation    - Manages metrics    - Generates reports  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────┤
│                             Core Layer                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Config     │  │   Dataset    │  │   Metrics    │  │    Judges        │  │
│  │  (Pydantic)  │  │   Loader     │  │   (6 types)  │  │  (3 providers)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────┤
│                           Output Layer                                        │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐  │
│  │   JSON Reporter      │  │   Markdown Reporter  │  │    Visualizer      │  │
│  │   (machine-readable) │  │   (human-readable)   │  │   (PNG charts)     │  │
│  └──────────────────────┘  └──────────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/llm_eval/
├── __init__.py              # Package initialization
├── cli.py                   # Typer CLI commands
├── config.py                # Pydantic configuration models
├── dataset.py               # Dataset loading and validation
├── pipeline.py              # Evaluation orchestration
│
├── metrics/                 # Metric implementations
│   ├── __init__.py          # Exports and factory registration
│   ├── base.py              # Abstract Metric class
│   ├── bleu.py              # BLEU score
│   ├── rouge.py             # ROUGE-L score
│   ├── bertscore.py         # BERTScore
│   ├── faithfulness.py      # Faithfulness metric
│   ├── context_relevancy.py # Context relevancy
│   └── answer_relevancy.py  # Answer relevancy
│
├── judges/                  # LLM-as-a-Judge implementations
│   ├── __init__.py          # Exports and factory
│   ├── base.py              # Abstract Judge class
│   ├── openai_judge.py      # GPT-4 integration
│   ├── anthropic_judge.py   # Claude integration
│   └── groq_judge.py        # Groq/Llama integration
│
├── reporting/               # Report generators
│   ├── __init__.py          # Exports
│   ├── json_reporter.py     # JSON output
│   ├── markdown_reporter.py # Markdown output
│   └── visualizer.py        # Chart generation
│
└── utils/                   # Utilities
    ├── __init__.py          # Exports
    ├── logging.py           # Logging configuration
    └── retry.py             # Retry with backoff
```

## Core Components

### 1. Configuration System (`config.py`)

The configuration system uses Pydantic for robust validation:

```python
class EvaluationConfig(BaseModel):
    dataset_path: Path           # Benchmark dataset
    output_dir: Path             # Results directory
    models: List[ModelConfig]    # Models to evaluate
    metrics: MetricsConfig       # Which metrics to compute
    judge: JudgeConfig           # LLM judge settings
```

**Key Features:**
- YAML and JSON support
- Environment variable integration for API keys
- Early validation with clear error messages
- Type hints for all fields

### 2. Dataset Loader (`dataset.py`)

Handles loading and validation of benchmark data:

```python
class DatasetLoader:
    def load_benchmark(self) -> List[BenchmarkExample]
    def load_model_outputs(self) -> List[ModelOutput]
    def iter_benchmark(self) -> Iterator[BenchmarkExample]
```

**Formats Supported:**
- JSONL (JSON Lines)
- CSV with proper escaping

**Required Fields:**
- `query`: The input question
- `expected_answer`: Reference answer
- `retrieved_contexts`: List of context passages

### 3. Metrics (`metrics/`)

All metrics inherit from a common base class:

```python
class Metric(ABC):
    name: str
    description: str
    
    @abstractmethod
    def compute(
        self,
        prediction: str,
        reference: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult
```

**Metric Types:**

| Category | Metrics | Implementation |
|----------|---------|----------------|
| Reference-based | BLEU, ROUGE-L, BERTScore | NLTK, rouge-score, sentence-transformers |
| RAG-specific | Faithfulness, Context Relevancy, Answer Relevancy | Embedding similarity |
| LLM-based | Coherence, Relevance, Safety | API calls to LLMs |

### 4. Judges (`judges/`)

LLM-as-a-Judge implementations for nuanced evaluation:

```python
class Judge(ABC):
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: Optional[List[str]] = None,
        reference: Optional[str] = None
    ) -> JudgeResult
```

**Supported Providers:**
- OpenAI (GPT-4)
- Anthropic (Claude)
- Groq (Llama 3.1, Mixtral)

**Features:**
- Structured JSON output
- Multi-dimensional rubric
- Retry with exponential backoff
- Response parsing with fallbacks

### 5. Pipeline (`pipeline.py`)

Orchestrates the entire evaluation process:

```python
class EvaluationPipeline:
    def __init__(self, config: EvaluationConfig)
    def run(self) -> Dict[str, Dict[str, Any]]
    def evaluate_model(self, model_name, output_path, benchmark)
```

**Responsibilities:**
1. Load benchmark dataset
2. Load model outputs
3. Match outputs to examples
4. Compute all enabled metrics
5. Run LLM judge (if enabled)
6. Generate reports and visualizations

### 6. Reporting (`reporting/`)

Generates output in multiple formats:

**JSON Reporter:**
- Machine-readable format
- Aggregate statistics (mean, median, std, min, max)
- Per-example breakdowns
- Error tracking

**Markdown Reporter:**
- Human-readable format
- Formatted tables
- Color-coded scores
- Sample results

**Visualizer:**
- Histogram per metric
- Radar chart for multi-model comparison
- PNG output at high resolution

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Config     │────▶│  Pipeline   │────▶│  Dataset Loader │
│  (YAML/JSON)│     │             │     │                 │
└─────────────┘     └──────┬──────┘     └────────┬────────┘
                          │                      │
                          ▼                      ▼
                   ┌─────────────┐     ┌─────────────────┐
                   │   Metrics   │◀────│  Benchmark +    │
                   │   Factory   │     │  Model Outputs  │
                   └──────┬──────┘     └─────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   BLEU      │  │  BERTScore  │  │   LLM       │
│   ROUGE-L   │  │  RAG Metrics│  │   Judge     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
               ┌─────────────────┐
               │   Aggregation   │
               │   & Statistics  │
               └────────┬────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    JSON     │  │  Markdown   │  │   Charts    │
│   Report    │  │   Report    │  │   (PNG)     │
└─────────────┘  └─────────────┘  └─────────────┘
```

## Design Patterns

### Factory Pattern (Metric Registration)

```python
class MetricFactory:
    _registry: Dict[str, Type[Metric]] = {}
    
    @classmethod
    def register(cls, name: str, metric_class: Type[Metric]):
        cls._registry[name] = metric_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Metric:
        return cls._registry[name](**kwargs)
```

### Strategy Pattern (Interchangeable Metrics)

Each metric implements the same interface, allowing them to be used interchangeably:

```python
for metric in self._metrics:
    result = metric.compute(prediction, reference, query, contexts)
```

### Template Method (Judge Base Class)

The Judge base class defines the evaluation template:

```python
class Judge(ABC):
    def evaluate(self, query, answer, contexts, reference):
        prompt = self._build_prompt(...)  # Template step
        response = self._call_llm(prompt)  # Abstract step
        return self._parse_response(response)  # Template step
```

### Retry Pattern (API Resilience)

```python
@retry_with_backoff(max_retries=3, min_wait=1.0, max_wait=30.0)
def _call_llm(self, prompt: str) -> str:
    return self.client.chat.completions.create(...)
```

## Extensibility

### Adding a New Metric

1. Create a new file in `src/llm_eval/metrics/`:

```python
# my_metric.py
from llm_eval.metrics.base import Metric, MetricResult

class MyMetric(Metric):
    name = "my_metric"
    
    def compute(self, prediction, reference, **kwargs):
        score = your_calculation(prediction, reference)
        return MetricResult(score=score)
```

2. Register in `metrics/__init__.py`:

```python
from llm_eval.metrics.my_metric import MyMetric
MetricFactory.register("my_metric", MyMetric)
```

### Adding a New Judge Provider

1. Create a new file in `src/llm_eval/judges/`:

```python
# custom_judge.py
from llm_eval.judges.base import Judge

class CustomJudge(Judge):
    def _call_llm(self, prompt: str) -> str:
        # Implement your API call
        return response
```

2. Register in `judges/__init__.py`:

```python
from llm_eval.judges.custom_judge import CustomJudge

def create_judge(provider, **kwargs):
    providers = {
        "custom": CustomJudge,
        # ...
    }
```

### Adding a New Report Format

1. Create a new reporter in `src/llm_eval/reporting/`:

```python
class HTMLReporter:
    def generate(self, results, model_name):
        # Generate HTML report
        pass
```

2. Integrate into the pipeline's `_generate_reports` method.

## Performance Considerations

### Batch Processing

BERTScore and embedding-based metrics batch encode texts:

```python
def compute_batch(self, predictions, references):
    embeddings = self._model.encode(all_texts, batch_size=32)
    # Process in batch...
```

### Model Caching

Sentence transformer models are cached at class level:

```python
class BERTScoreMetric(Metric):
    _model: Optional[SentenceTransformer] = None
    
    def _ensure_model_loaded(self):
        if BERTScoreMetric._model is None:
            BERTScoreMetric._model = SentenceTransformer(self.model_name)
```

### Rate Limiting

LLM judge calls use exponential backoff to handle rate limits:

```python
@retry_with_backoff(max_retries=3, min_wait=1.0, max_wait=30.0)
def _call_llm(self, prompt):
    # API call with automatic retry on rate limit
```

## Security Considerations

1. **API Keys**: Stored in environment variables, never in code
2. **Input Validation**: Pydantic models validate all inputs
3. **Container Security**: Non-root user in Docker
4. **CI/CD**: Trivy security scanning in GitHub Actions
