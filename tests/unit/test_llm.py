"""Tests for the LLM layer components."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from doc_agent.config.models import LLMConfig, LLMProviderType
from doc_agent.context.metadata_extractor import MetadataExtractor
from doc_agent.llm.claude import ClaudeProvider
from doc_agent.llm.factory import create_llm_provider, get_available_providers
from doc_agent.llm.protocol import LLMMessage, LLMProvider, LLMResponse, LLMUsage
from doc_agent.llm.prompts.article_eval import (
    build_article_evaluation_prompt,
    build_quick_scan_prompt,
    ARTICLE_EVALUATION_SYSTEM_PROMPT,
)
from doc_agent.llm.response_parser import (
    extract_json_from_response,
    parse_evaluation_response,
    parse_quick_scan_response,
)
from doc_agent.models.enums import Dimension, Severity


class TestLLMMessage:
    """Tests for LLMMessage."""

    def test_user_message(self):
        """Test creating a user message."""
        msg = LLMMessage.user("Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_assistant_message(self):
        """Test creating an assistant message."""
        msg = LLMMessage.assistant("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_system_message(self):
        """Test creating a system message."""
        msg = LLMMessage.system("You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_basic_response(self):
        """Test creating a basic response."""
        response = LLMResponse(
            content="Hello!",
            model="claude-sonnet-4-20250514",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        assert response.content == "Hello!"
        assert response.usage.total_tokens == 15

    def test_truncation_detection(self):
        """Test detecting truncated responses."""
        truncated = LLMResponse(
            content="...",
            model="test",
            stop_reason="max_tokens",
        )
        assert truncated.was_truncated is True

        normal = LLMResponse(
            content="...",
            model="test",
            stop_reason="end_turn",
        )
        assert normal.was_truncated is False


class TestLLMProviderProtocol:
    """Tests for the LLMProvider protocol."""

    def test_claude_implements_protocol(self):
        """Test that ClaudeProvider implements LLMProvider protocol."""
        provider = ClaudeProvider()
        assert isinstance(provider, LLMProvider)

    def test_model_name_property(self):
        """Test model name property."""
        provider = ClaudeProvider(model="claude-opus-4-20250514")
        assert provider.model_name == "claude-opus-4-20250514"


class TestProviderFactory:
    """Tests for the provider factory."""

    def test_create_claude_provider(self):
        """Test creating a Claude provider."""
        config = LLMConfig(provider=LLMProviderType.CLAUDE)
        provider = create_llm_provider(config)
        assert isinstance(provider, ClaudeProvider)

    def test_create_with_model_override(self):
        """Test creating provider with model override."""
        config = LLMConfig(model="claude-sonnet-4-20250514")
        provider = create_llm_provider(config, model="claude-opus-4-20250514")
        assert provider.model_name == "claude-opus-4-20250514"

    def test_create_with_defaults(self):
        """Test creating provider with all defaults."""
        provider = create_llm_provider()
        assert isinstance(provider, ClaudeProvider)

    def test_available_providers(self):
        """Test getting available providers."""
        providers = get_available_providers()
        assert "claude" in providers


class TestPrompts:
    """Tests for prompt generation."""

    @pytest.fixture
    def sample_article(self, fixtures_dir: Path):
        """Create a sample article for testing."""
        extractor = MetadataExtractor()
        article_path = fixtures_dir / "sample_article.md"
        return extractor.create_article(article_path, fixtures_dir)

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert len(ARTICLE_EVALUATION_SYSTEM_PROMPT) > 100
        assert "developer" in ARTICLE_EVALUATION_SYSTEM_PROMPT.lower()

    def test_build_evaluation_prompt(self, sample_article):
        """Test building evaluation prompt."""
        prompt = build_article_evaluation_prompt(sample_article)

        # Check key elements are present
        assert sample_article.relative_path in prompt
        assert sample_article.metadata.title in prompt
        assert "pattern_compliance" in prompt
        assert "dev_focus" in prompt
        assert "code_quality" in prompt
        assert "JSON" in prompt

    def test_build_quick_scan_prompt(self, sample_article):
        """Test building quick scan prompt."""
        prompt = build_quick_scan_prompt(sample_article)

        # Check quick scan specific elements
        assert "quick" in prompt.lower()
        assert sample_article.relative_path in prompt
        assert "needs_deep_evaluation" in prompt


class TestResponseParser:
    """Tests for response parsing."""

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        content = '''Here is the evaluation:

```json
{"score": 5, "issues": []}
```

That's my assessment.'''

        result = extract_json_from_response(content)
        assert result == '{"score": 5, "issues": []}'

    def test_extract_json_direct(self):
        """Test extracting JSON without code block."""
        content = '{"score": 5, "issues": []}'
        result = extract_json_from_response(content)
        assert result == '{"score": 5, "issues": []}'

    def test_extract_json_with_extra_content(self):
        """Test extracting JSON with trailing content."""
        content = '{"score": 5, "issues": []} Some extra text'
        result = extract_json_from_response(content)
        assert result == '{"score": 5, "issues": []}'

    def test_parse_evaluation_response(self):
        """Test parsing a full evaluation response."""
        response = json.dumps({
            "dimension_scores": [
                {
                    "dimension": "pattern_compliance",
                    "score": 5,
                    "confidence": 0.9,
                    "rationale": "Good pattern adherence",
                },
                {
                    "dimension": "dev_focus",
                    "score": 4,
                    "confidence": 0.8,
                    "rationale": "Code could come earlier",
                },
            ],
            "issues": [
                {
                    "dimension": "dev_focus",
                    "severity": "medium",
                    "title": "Code position",
                    "description": "First code block appears late",
                    "location": "line 50",
                    "suggestion": "Move code example earlier",
                },
            ],
            "summary": "Overall a good article with room for improvement.",
        })

        result = parse_evaluation_response(
            response,
            article_path="test.md",
            article_title="Test Article",
            model_used="test-model",
        )

        assert result.article_path == "test.md"
        assert len(result.dimension_scores) == 2
        assert len(result.issues) == 1
        assert result.issues[0].severity == Severity.MEDIUM
        assert result.issues[0].dimension == Dimension.DEV_FOCUS

    def test_parse_quick_scan_response(self):
        """Test parsing a quick scan response."""
        response = json.dumps({
            "quick_score": 4.5,
            "needs_deep_evaluation": False,
            "flags": ["Minor heading hierarchy issue"],
            "pattern_match": True,
            "structure_valid": True,
            "has_code": True,
            "is_recent": True,
            "has_prerequisites": True,
        })

        result = parse_quick_scan_response(response)

        assert result.quick_score == 4.5
        assert result.needs_deep_evaluation is False
        assert len(result.flags) == 1
        assert result.has_code is True

    def test_parse_handles_invalid_json(self):
        """Test that parser raises on invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_evaluation_response(
                "not valid json {{}",
                article_path="test.md",
                article_title="Test",
                model_used="test",
            )

    def test_parse_handles_unknown_dimension(self):
        """Test that parser handles unknown dimensions gracefully."""
        response = json.dumps({
            "dimension_scores": [
                {
                    "dimension": "unknown_dimension",
                    "score": 5,
                    "confidence": 0.9,
                    "rationale": "Test",
                },
                {
                    "dimension": "structure",
                    "score": 4,
                    "confidence": 0.8,
                    "rationale": "Test",
                },
            ],
            "issues": [],
            "summary": "Test",
        })

        result = parse_evaluation_response(
            response,
            article_path="test.md",
            article_title="Test",
            model_used="test",
        )

        # Unknown dimension should be skipped
        assert len(result.dimension_scores) == 1
        assert result.dimension_scores[0].dimension == Dimension.STRUCTURE
