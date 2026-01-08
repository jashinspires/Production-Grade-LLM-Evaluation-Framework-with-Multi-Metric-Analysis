# LLM Evaluation Framework

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/coverage-80%25+-brightgreen?style=for-the-badge" alt="Coverage">
  <img src="https://img.shields.io/badge/style-black-black?style=for-the-badge" alt="Code Style">
</p>

---

## Here's a question for you...

**How do you know if your AI is actually good?**

Think about it. You ask GPT-4 a question, it gives you an answer. Sounds reasonable. But is it *correct*? Is it *faithful* to the sources? Is it even *relevant* to what you asked?

Most people just... read the answer and think "yeah, that looks right." But here's the thing — that doesn't scale. When you have thousands of queries, you can't manually check each one. You need a system.

That's what this framework does.

---

## The Challenge

This project was built as part of a production-grade ML engineering assignment. The goal: create a comprehensive LLM evaluation framework that goes beyond simple metrics.

The core problem this addresses:

> **Models can sound confident while being completely wrong.**

It's called *hallucination*. The model generates fluent, convincing text that has nothing to do with reality. And basic metrics like BLEU and ROUGE don't catch it — they just count word overlap.

This framework addresses that by measuring **multiple dimensions** of quality.

---

## The Multi-Dimensional Approach

**Evaluating LLMs isn't a single-number problem. It's a multi-dimensional problem.**

Think about what can go wrong with a model's answer:
- It could be **grammatically perfect but factually wrong**
- It could be **accurate but irrelevant** to the question
- It could **sound good but be made up** (not grounded in sources)
- It could answer **a different question** than what was asked

No single metric catches all of these. So this framework uses **six different lenses**:

```
┌─────────────────────────────────────────────────────────────┐
│                    REFERENCE-BASED                          │
│  ┌─────────┐    ┌─────────┐    ┌────────────┐              │
│  │  BLEU   │    │ ROUGE-L │    │ BERTScore  │              │
│  │ n-gram  │    │   LCS   │    │  semantic  │              │
│  └─────────┘    └─────────┘    └────────────┘              │
├─────────────────────────────────────────────────────────────┤
│                      RAG-SPECIFIC                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐      │
│  │Faithfulness │  │  Context     │  │    Answer     │      │
│  │ grounded?   │  │  Relevancy   │  │   Relevancy   │      │
│  └─────────────┘  └──────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Why These Specific Metrics

Each metric was chosen because it catches something the others miss:

| Metric | Why I included it | What it catches that others don't |
|--------|-------------------|-----------------------------------|
| **BLEU** | Industry standard, comparable to other papers | Exact word matching, useful baseline |
| **ROUGE-L** | Better for variable-length responses | Sequence matching regardless of word position |
| **BERTScore** | Finally understands *meaning* | "car" and "automobile" are similar |
| **Faithfulness** | The hallucination catcher | Facts made up vs facts from sources |
| **Context Relevancy** | Catches retrieval failures | Did we even fetch the right documents? |
| **Answer Relevancy** | The "did you answer MY question" check | Model answered correctly... but wrong question |

The trade-off with semantic similarity metrics: they're slower (need to load neural models) but much more accurate. The accuracy is worth the speed hit — especially for evaluation, where you're not running in real-time.

---

## Key Design Decisions

### Decision 1: Factory Pattern for Metrics

The framework is designed so anyone can add their own metrics easily using a factory pattern:

```python
MetricFactory.register("my_metric", MyMetricClass)
```

One line to register, and your metric works everywhere — CLI, pipeline, reports. This took extra effort upfront but makes the system actually extensible.

### Decision 2: Pydantic for Configuration

Why Pydantic instead of plain dictionaries?
- It validates config files *before* running (fail fast)
- Error messages tell you exactly what's wrong
- Type hints work in IDEs (autocomplete, documentation)

This catches configuration problems immediately rather than failing deep in the evaluation.

### Decision 3: LLM-as-a-Judge with Multiple Providers

To avoid vendor lock-in, the framework supports:
- **OpenAI** (GPT-4) — highest quality
- **Anthropic** (Claude) — good alternative
- **Groq** (Llama) — free tier available!

The same prompt and rubric work across providers, so results are comparable.

### Decision 4: Retry Logic with Exponential Backoff

API calls fail. Rate limits happen. Instead of crashing, the system retries with exponential backoff:

```
Attempt 1 fails → wait 1s → retry
Attempt 2 fails → wait 2s → retry
Attempt 3 fails → wait 4s → retry
```

This sounds simple but makes the difference between a toy project and something you can actually rely on.

---

## What Makes This Framework Different

Comparison with typical approaches:

| Existing approach | The problem | What I did differently |
|-------------------|-------------|------------------------|
| Separate scripts for each metric | Hard to compare, inconsistent | Unified pipeline, same interface |
| Config in code | Have to edit Python to change settings | YAML/JSON config files |
| One report format | Either machine OR human readable | Both JSON and Markdown |
| No visualizations | Hard to spot patterns | Histograms + radar charts |
| Single provider lock-in | Expensive or risky | Three LLM providers supported |

The goal was: **run one command, get everything you need to evaluate your model.**

---

## The Architecture (And Why It's Layered This Way)

```
         ┌──────────────┐
         │  Your Data   │
         │  (JSONL)     │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Config     │◄─── YAML file with settings
         │  (Pydantic)  │
         └──────┬───────┘
                │
                ▼
    ┌───────────────────────┐
    │   Evaluation Pipeline │
    │                       │
    │  1. Load benchmark    │
    │  2. Load model output │
    │  3. Match examples    │
    │  4. Compute metrics   │
    │  5. Run LLM judge     │
    │  6. Generate reports  │
    └───────────┬───────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   ┌─────────┐    ┌──────────┐
   │  JSON   │    │ Markdown │
   │ Report  │    │  Report  │
   └─────────┘    └──────────┘
```

**Why this structure?**

- **Separation of concerns**: CLI doesn't know about metrics, metrics don't know about reports
- **Testability**: I can test each layer independently with mocks
- **Extensibility**: Add a new metric? Just implement the interface. Add a new report format? Same thing.

This is the kind of architecture you see in production systems - not because it's fancy, but because it actually makes the code maintainable.

---

## Challenges I Faced (And How I Solved Them)

### Challenge 1: Matching model outputs to benchmark examples

The benchmark has query-answer pairs. The model output has query-prediction pairs. But what if the queries don't match exactly? Whitespace differences, slight rewording...

**Solution**: Normalize queries before matching, with fallback to fuzzy matching. The loader handles this transparently.

### Challenge 2: BERTScore is slow

Loading the sentence-transformer model takes time. Running embeddings on every example is expensive.

**Solution**: Lazy loading (model loads only when needed) + class-level caching (model loaded once, reused). Also batch processing instead of one-at-a-time.

### Challenge 3: LLM judges return unparseable responses

Sometimes the judge doesn't return valid JSON. Sometimes it adds extra text around the JSON.

**Solution**: Robust parsing with regex fallbacks. Extract JSON from anywhere in the response. If parsing fails completely, log the error and continue — don't crash the whole evaluation.

### Challenge 4: API rate limits

When running many evaluations, you hit rate limits.

**Solution**: Exponential backoff with jitter. The system automatically retries with increasing delays. This handles transient failures gracefully.

---

## What I Learned Building This

1. **Metrics are harder than they look.** Getting BLEU to handle edge cases (empty strings, single words, very long texts) took way more code than the core algorithm.

2. **Configuration is a feature.** Good config validation saves hours of debugging. Invest in it early.

3. **Multiple evaluation dimensions are essential.** A model can score high on BLEU and still be useless. You need to measure what actually matters.

4. **Retry logic is not optional.** Any system that calls external APIs needs resilient error handling.

5. **Testing with mocks is the only way to test API integrations.** You can't run real API calls in CI. Mock everything external.

---

## Getting Started

### Step 1: Installation

```bash
git clone https://github.com/your-org/llm-eval.git
cd llm-eval
poetry install
```

### Step 2: Set up API keys

```bash
cp .env.example .env
```

Add at least one key (Groq is free!):

```env
GROQ_API_KEY=gsk_...
```

### Step 3: Run evaluation

```bash
llm-eval run --config examples/config.yaml --output-dir results/
```

---

## CLI Commands

| Command | Purpose |
|---------|---------|
| `llm-eval run --config config.yaml` | Run full evaluation |
| `llm-eval validate --config config.yaml` | Check config validity |
| `llm-eval list-metrics` | Show available metrics |

**Useful flags:**
- `--metrics bleu rouge_l` — run only specific metrics
- `--verbose` — debug logging
- `--no-progress` — CI-friendly output

---

## Configuration Reference

```yaml
dataset_path: benchmarks/rag_benchmark.jsonl
output_dir: results

models:
  - name: gpt-4
    output_path: outputs/gpt4.jsonl

metrics:
  bleu: true
  rouge_l: true
  bertscore: true
  faithfulness: true
  context_relevancy: true
  answer_relevancy: true
  llm_judge: true

judge:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.0
```

---

## Adding Custom Metrics

```python
from llm_eval.metrics.base import Metric, MetricResult

class MyMetric(Metric):
    name = "my_metric"
    
    def compute(self, prediction, reference, **kwargs):
        score = your_calculation(prediction, reference)
        return MetricResult(score=score)

# Register
from llm_eval.metrics import MetricFactory
MetricFactory.register("my_metric", MyMetric)
```

---

## Docker

```bash
docker-compose up              # Run evaluation
docker-compose --profile test up  # Run tests
```

Multi-stage build, non-root user, health checks included.

---

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/ --cov=llm_eval        # With coverage
pytest tests/unit/ -v               # Unit only
```

---

## Project Structure

```
src/llm_eval/
├── cli.py           # Typer commands
├── config.py        # Pydantic models
├── dataset.py       # Data loading
├── pipeline.py      # Orchestration
├── metrics/         # BLEU, ROUGE, BERTScore, RAG metrics
├── judges/          # OpenAI, Anthropic, Groq
├── reporting/       # JSON, Markdown, Charts
└── utils/           # Logging, retry
```

---

## The Bottom Line

Here's what I believe after building this:

**Evaluation isn't just a checkbox. It's how you know if your work actually works.**

When you deploy an AI system, you're making a promise that it will help users. This framework helps you keep that promise — by measuring what matters, catching what fails, and giving you clear answers instead of vague feelings.

That's what good engineering looks like.

---

## Contributing

1. Fork → Branch → Code → Test → PR

---

## License

MIT — use it however you want.

---

<p align="center">
  <i>Built with care, tested with rigor, documented with clarity.</i>
</p>
