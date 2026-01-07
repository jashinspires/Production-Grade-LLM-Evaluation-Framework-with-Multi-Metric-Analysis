"""
Unit tests for LLM-as-a-Judge implementations.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_eval.judges.base import Judge, JudgeResult, DEFAULT_RUBRIC
from llm_eval.judges.openai_judge import OpenAIJudge
from llm_eval.judges.anthropic_judge import AnthropicJudge
from llm_eval.judges.groq_judge import GroqJudge
from llm_eval.judges import create_judge


class TestJudgeResult:
    """Tests for JudgeResult."""
    
    def test_valid_result(self):
        """Test valid judge result."""
        result = JudgeResult(
            scores={"coherence": 0.75, "relevance": 0.80},
            raw_scores={"coherence": 4, "relevance": 5},
            reasoning={"coherence": "Clear", "relevance": "Addresses query"}
        )
        
        assert result.is_valid
        assert result.average_score == 0.775
    
    def test_invalid_result(self):
        """Test invalid judge result with error."""
        result = JudgeResult(error="API call failed")
        
        assert not result.is_valid
        assert result.average_score == 0.0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result = JudgeResult(
            scores={"coherence": 0.75},
            overall_assessment="Good response"
        )
        
        d = result.to_dict()
        
        assert "scores" in d
        assert "average_score" in d
        assert d["overall_assessment"] == "Good response"


class TestJudgeBase:
    """Tests for base Judge class."""
    
    def test_build_prompt(self):
        """Test prompt building."""
        class TestJudge(Judge):
            def _call_llm(self, prompt):
                return ""
        
        judge = TestJudge(model="test-model")
        
        prompt = judge._build_prompt(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI refers to machine intelligence."],
            reference="AI stands for artificial intelligence."
        )
        
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence." in prompt
        assert "AI refers to machine intelligence." in prompt
    
    def test_parse_response_valid(self):
        """Test parsing valid JSON response."""
        class TestJudge(Judge):
            def _call_llm(self, prompt):
                return ""
        
        judge = TestJudge(model="test-model")
        
        response = json.dumps({
            "coherence": {"score": 4, "reasoning": "Clear"},
            "relevance": {"score": 5, "reasoning": "On topic"},
            "safety": {"score": 5, "reasoning": "Safe"},
            "overall_assessment": "Good"
        })
        
        result = judge._parse_response(response)
        
        assert result.is_valid
        assert result.scores["coherence"] == 0.75  # (4-1)/4
        assert result.scores["relevance"] == 1.0  # (5-1)/4
    
    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        class TestJudge(Judge):
            def _call_llm(self, prompt):
                return ""
        
        judge = TestJudge(model="test-model")
        result = judge._parse_response("not valid json")
        
        assert not result.is_valid
        assert "No JSON found" in result.error


class TestOpenAIJudge:
    """Tests for OpenAIJudge."""
    
    @patch("llm_eval.judges.openai_judge.OpenAI")
    def test_evaluate(self, mock_openai_class, mock_openai_response):
        """Test evaluation with mocked API."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client
        
        judge = OpenAIJudge(model="gpt-4", api_key="test-key")
        result = judge.evaluate(
            query="What is AI?",
            answer="AI is artificial intelligence."
        )
        
        assert result.is_valid
        assert "coherence" in result.scores
    
    @patch("llm_eval.judges.openai_judge.OpenAI")
    def test_missing_api_key(self, mock_openai_class):
        """Test that missing API key raises error."""
        import os
        old_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises(ValueError, match="API key not found"):
            OpenAIJudge(model="gpt-4")
        
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


class TestGroqJudge:
    """Tests for GroqJudge."""
    
    @patch("llm_eval.judges.groq_judge.Groq")
    def test_evaluate(self, mock_groq_class, mock_groq_response):
        """Test evaluation with mocked API."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response
        mock_groq_class.return_value = mock_client
        
        judge = GroqJudge(
            model="llama-3.1-70b-versatile",
            api_key="test-key"
        )
        result = judge.evaluate(
            query="What is AI?",
            answer="AI is artificial intelligence."
        )
        
        assert result.is_valid
        assert "coherence" in result.scores


class TestCreateJudge:
    """Tests for create_judge factory function."""
    
    @patch("llm_eval.judges.openai_judge.OpenAI")
    def test_create_openai_judge(self, mock_openai):
        """Test creating OpenAI judge."""
        judge = create_judge("openai", model="gpt-4", api_key="test")
        assert isinstance(judge, OpenAIJudge)
    
    @patch("llm_eval.judges.groq_judge.Groq")
    def test_create_groq_judge(self, mock_groq):
        """Test creating Groq judge."""
        judge = create_judge("groq", model="llama-3.1-70b-versatile", api_key="test")
        assert isinstance(judge, GroqJudge)
    
    def test_unknown_provider(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_judge("unknown_provider")
