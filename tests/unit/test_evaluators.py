"""Tests for the evaluators module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_eval.context.metadata_extractor import MetadataExtractor
from foundry_eval.evaluators.article import ArticleEvaluator, TieredArticleEvaluator
from foundry_eval.evaluators.base import BaseEvaluator
from foundry_eval.llm.protocol import LLMMessage, LLMResponse, LLMUsage
from foundry_eval.models.article import Article
from foundry_eval.models.enums import Dimension, Severity
from foundry_eval.models.evaluation import EvaluationResult


class TestBaseEvaluator:
    """Tests for BaseEvaluator."""

    def test_is_abstract(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator(MagicMock())

    def test_model_name_property(self):
        """Test model name property delegates to provider."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"

        # Create a concrete subclass for testing
        class ConcreteEvaluator(BaseEvaluator):
            async def evaluate(self, article):
                pass

            def get_prompt(self, article):
                return ""

            def get_system_prompt(self):
                return ""

        evaluator = ConcreteEvaluator(mock_provider)
        assert evaluator.model_name == "test-model"


class TestArticleEvaluator:
    """Tests for ArticleEvaluator."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.model_name = "test-model"
        return provider

    @pytest.fixture
    def sample_article(self, fixtures_dir: Path):
        """Load the sample article fixture."""
        extractor = MetadataExtractor()
        article_path = fixtures_dir / "sample_article.md"
        return extractor.create_article(article_path, fixtures_dir)

    @pytest.fixture
    def mock_evaluation_response(self):
        """Create a mock evaluation response."""
        return json.dumps({
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
                    "confidence": 0.85,
                    "rationale": "Code examples present",
                },
                {
                    "dimension": "code_quality",
                    "score": 5,
                    "confidence": 0.9,
                    "rationale": "Well-written samples",
                },
                {
                    "dimension": "code_completeness",
                    "score": 4,
                    "confidence": 0.8,
                    "rationale": "Most steps have code",
                },
                {
                    "dimension": "structure",
                    "score": 5,
                    "confidence": 0.95,
                    "rationale": "Clear organization",
                },
                {
                    "dimension": "accuracy",
                    "score": 5,
                    "confidence": 0.85,
                    "rationale": "Content appears current",
                },
                {
                    "dimension": "jtbd_alignment",
                    "score": 4,
                    "confidence": 0.75,
                    "rationale": "Supports stated intent",
                },
            ],
            "issues": [
                {
                    "dimension": "dev_focus",
                    "severity": "medium",
                    "title": "Code could be earlier",
                    "description": "First code block appears after significant prose",
                    "location": "line 30",
                    "suggestion": "Move example to after prerequisites",
                },
            ],
            "summary": "Overall a well-structured article with good code examples.",
        })

    @pytest.fixture
    def mock_quick_scan_response(self):
        """Create a mock quick scan response."""
        return json.dumps({
            "quick_score": 4.5,
            "needs_deep_evaluation": False,
            "flags": [],
            "pattern_match": True,
            "structure_valid": True,
            "has_code": True,
            "is_recent": True,
            "has_prerequisites": True,
        })

    def test_evaluator_creation(self, mock_llm_provider):
        """Test creating an article evaluator."""
        evaluator = ArticleEvaluator(mock_llm_provider)
        assert evaluator.model_name == "test-model"
        assert evaluator.is_quick_scan is False

    def test_quick_scan_mode(self, mock_llm_provider):
        """Test quick scan mode setting."""
        evaluator = ArticleEvaluator(mock_llm_provider, quick_scan_mode=True)
        assert evaluator.is_quick_scan is True

    def test_get_prompt_full(self, mock_llm_provider, sample_article):
        """Test generating full evaluation prompt."""
        evaluator = ArticleEvaluator(mock_llm_provider)
        prompt = evaluator.get_prompt(sample_article)

        assert sample_article.relative_path in prompt
        assert sample_article.metadata.title in prompt
        assert "pattern_compliance" in prompt
        assert "dev_focus" in prompt
        assert "JSON" in prompt

    def test_get_prompt_quick_scan(self, mock_llm_provider, sample_article):
        """Test generating quick scan prompt."""
        evaluator = ArticleEvaluator(mock_llm_provider, quick_scan_mode=True)
        prompt = evaluator.get_prompt(sample_article)

        assert "quick" in prompt.lower()
        assert "needs_deep_evaluation" in prompt

    def test_get_system_prompt(self, mock_llm_provider):
        """Test getting system prompt."""
        evaluator = ArticleEvaluator(mock_llm_provider)
        system = evaluator.get_system_prompt()

        assert len(system) > 100
        assert "developer" in system.lower() or "documentation" in system.lower()

    @pytest.mark.asyncio
    async def test_evaluate_full(
        self,
        mock_llm_provider,
        sample_article,
        mock_evaluation_response,
    ):
        """Test full article evaluation."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content=mock_evaluation_response,
            model="test-model",
            usage=LLMUsage(input_tokens=1000, output_tokens=500),
        )

        evaluator = ArticleEvaluator(mock_llm_provider)
        result = await evaluator.evaluate(sample_article)

        assert isinstance(result, EvaluationResult)
        assert result.article_path == sample_article.relative_path
        assert len(result.dimension_scores) == 7
        assert len(result.issues) == 1
        assert result.llm_model_used == "test-model"
        assert result.token_usage["input_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_evaluate_quick_scan(
        self,
        mock_llm_provider,
        sample_article,
        mock_quick_scan_response,
    ):
        """Test quick scan evaluation."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content=mock_quick_scan_response,
            model="test-model",
            usage=LLMUsage(input_tokens=500, output_tokens=200),
        )

        evaluator = ArticleEvaluator(mock_llm_provider, quick_scan_mode=True)
        result = await evaluator.evaluate(sample_article)

        assert isinstance(result, EvaluationResult)
        assert result.is_quick_scan is True
        assert result.needs_deep_evaluation is False

    @pytest.mark.asyncio
    async def test_evaluate_batch(
        self,
        mock_llm_provider,
        sample_article,
        mock_evaluation_response,
    ):
        """Test batch evaluation."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content=mock_evaluation_response,
            model="test-model",
            usage=LLMUsage(input_tokens=1000, output_tokens=500),
        )

        evaluator = ArticleEvaluator(mock_llm_provider, max_concurrency=2)

        # Create multiple copies of the article
        articles = [sample_article, sample_article]
        results = await evaluator.evaluate_batch(articles)

        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)


class TestTieredArticleEvaluator:
    """Tests for TieredArticleEvaluator."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.model_name = "test-model"
        return provider

    def test_evaluator_creation(self, mock_llm_provider):
        """Test creating a tiered evaluator."""
        evaluator = TieredArticleEvaluator(
            mock_llm_provider,
            quick_scan_threshold=3.5,
        )
        assert evaluator._threshold == 3.5

    @pytest.mark.asyncio
    async def test_tiered_evaluation_passes_quick_scan(
        self,
        mock_llm_provider,
        fixtures_dir: Path,
    ):
        """Test tiered evaluation when article passes quick scan."""
        # Mock responses - passing quick scan
        quick_scan_response = json.dumps({
            "quick_score": 4.5,
            "needs_deep_evaluation": False,
            "flags": [],
            "pattern_match": True,
            "structure_valid": True,
            "has_code": True,
            "is_recent": True,
            "has_prerequisites": True,
        })

        mock_llm_provider.complete.return_value = LLMResponse(
            content=quick_scan_response,
            model="test-model",
            usage=LLMUsage(input_tokens=500, output_tokens=200),
        )

        extractor = MetadataExtractor()
        article = extractor.create_article(
            fixtures_dir / "sample_article.md",
            fixtures_dir,
        )

        evaluator = TieredArticleEvaluator(
            mock_llm_provider,
            quick_scan_threshold=4.0,
        )
        results = await evaluator.evaluate_batch([article])

        assert len(results) == 1
        assert results[0].is_quick_scan is True
        # Should only have called LLM once (quick scan only)
        assert mock_llm_provider.complete.call_count == 1
