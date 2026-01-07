"""
Base classes for metrics.

All metric implementations should extend from Metric class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union


@dataclass
class MetricResult:
    """Holds the result of a metric computation."""
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if the result is valid (no error)."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "score": self.score,
            "details": self.details,
            "error": self.error,
        }


class Metric(ABC):
    """Base class - extend this and implement compute() method."""
    
    # Metric name (should be overridden by subclasses)
    name: str = "base_metric"
    
    # Description of what the metric measures
    description: str = "Base metric class"
    
    # Score range (for documentation)
    score_range: tuple = (0.0, 1.0)
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the metric with optional configuration.
        
        Args:
            **kwargs: Metric-specific configuration options
        """
        self.config = kwargs
    
    @abstractmethod
    def compute(
        self,
        prediction: str,
        reference: str,
        query: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute the metric score.
        
        Args:
            prediction: The model's predicted answer
            reference: The expected/reference answer
            query: Optional query that generated the prediction
            contexts: Optional list of retrieved context passages
            **kwargs: Additional metric-specific arguments
            
        Returns:
            MetricResult containing the score and any additional details
        """
        pass
    
    def compute_batch(
        self,
        predictions: List[str],
        references: List[str],
        queries: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any
    ) -> List[MetricResult]:
        """
        Compute the metric for a batch of examples.
        
        Default implementation calls compute() for each example.
        Subclasses may override for more efficient batch processing.
        
        Args:
            predictions: List of model predictions
            references: List of reference answers
            queries: Optional list of queries
            contexts: Optional list of context lists
            **kwargs: Additional metric-specific arguments
            
        Returns:
            List of MetricResult instances
        """
        results = []
        
        # Handle None cases
        if queries is None:
            queries = [None] * len(predictions)
        if contexts is None:
            contexts = [None] * len(predictions)
        
        for pred, ref, query, ctx in zip(predictions, references, queries, contexts):
            try:
                result = self.compute(
                    prediction=pred,
                    reference=ref,
                    query=query,
                    contexts=ctx,
                    **kwargs
                )
            except Exception as e:
                result = MetricResult(
                    score=0.0,
                    error=str(e)
                )
            results.append(result)
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class MetricFactory:
    """Registry for creating metric instances by name."""
    
    _registry: Dict[str, Type[Metric]] = {}
    
    @classmethod
    def register(cls, name: str, metric_class: Type[Metric]) -> None:
        """
        Register a metric class with the factory.
        
        Args:
            name: Unique identifier for the metric
            metric_class: The metric class to register
        """
        cls._registry[name.lower()] = metric_class
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Metric:
        """
        Create a metric instance by name.
        
        Args:
            name: Name of the metric to create
            **kwargs: Configuration options for the metric
            
        Returns:
            Configured metric instance
            
        Raises:
            ValueError: If the metric name is not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown metric: {name}. Available metrics: {available}"
            )
        
        metric_class = cls._registry[name_lower]
        return metric_class(**kwargs)
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """
        List all registered metric names.
        
        Returns:
            List of registered metric names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_metric_class(cls, name: str) -> Optional[Type[Metric]]:
        """
        Get the metric class by name without instantiating.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric class or None if not found
        """
        return cls._registry.get(name.lower())
