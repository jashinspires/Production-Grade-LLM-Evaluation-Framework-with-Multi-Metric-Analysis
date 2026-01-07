# LLM Evaluation Framework - Multi-stage Dockerfile
# Build stage for dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Create non-root user
RUN groupadd -r llmeval && useradd -r -g llmeval llmeval

# Set work directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pyproject.toml ./

# Install the package
RUN pip install -e . --no-deps

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Create results directory
RUN mkdir -p /app/results && chown -R llmeval:llmeval /app

# Switch to non-root user
USER llmeval

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from llm_eval import __version__; print(__version__)" || exit 1

# Default command
CMD ["llm-eval", "--help"]

# Development stage
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
RUN pip install pytest pytest-cov pytest-mock black isort mypy flake8

# Switch back to non-root user
USER llmeval

# Default command for development
CMD ["pytest", "tests/", "-v"]
