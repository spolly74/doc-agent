"""Article evaluator implementing 7-dimension rubric scoring."""

import logging
import time
from typing import Optional

from foundry_eval.evaluators.base import BaseEvaluator
from foundry_eval.llm.protocol import LLMMessage, LLMProvider
from foundry_eval.llm.prompts.article_eval import (
    ARTICLE_EVALUATION_SYSTEM_PROMPT,
    build_article_evaluation_prompt,
    build_quick_scan_prompt,
)
from foundry_eval.llm.response_parser import (
    parse_evaluation_response,
    parse_quick_scan_response,
    quick_scan_to_evaluation_result,
)
from foundry_eval.models.article import Article
from foundry_eval.models.enums import Dimension
from foundry_eval.models.evaluation import EvaluationResult

logger = logging.getLogger("foundry_eval.evaluators.article")


class ArticleEvaluator(BaseEvaluator[EvaluationResult]):
    """Evaluates articles against the 7-dimension rubric.

    Dimensions:
    1. Pattern Compliance - Following correct content pattern
    2. Developer Focus - Time to first success optimization
    3. Code Quality - Well-written, maintainable code samples
    4. Code Completeness - All necessary samples present
    5. Structure - Organization and scannability
    6. Accuracy - Technical correctness and currency
    7. JTBD Alignment - Supporting customer jobs (when applicable)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_concurrency: int = 5,
        quick_scan_mode: bool = False,
        dimensions: Optional[list[Dimension]] = None,
    ):
        """Initialize the article evaluator.

        Args:
            llm_provider: LLM provider for evaluation calls.
            max_concurrency: Maximum concurrent evaluations.
            quick_scan_mode: If True, perform lightweight structural scan only.
            dimensions: Optional list of dimensions to evaluate (defaults to all).
        """
        super().__init__(llm_provider, max_concurrency)
        self._quick_scan = quick_scan_mode
        self._dimensions = dimensions or list(Dimension)

    @property
    def is_quick_scan(self) -> bool:
        """Whether this evaluator is in quick scan mode."""
        return self._quick_scan

    async def evaluate(self, article: Article) -> EvaluationResult:
        """Evaluate a single article against the rubric.

        Args:
            article: Article to evaluate.

        Returns:
            EvaluationResult with scores and issues.
        """
        start_time = time.time()

        if self._quick_scan:
            result = await self._evaluate_quick_scan(article)
        else:
            result = await self._evaluate_full(article)

        # Record evaluation duration
        result.evaluation_duration_seconds = time.time() - start_time

        return result

    async def _evaluate_full(self, article: Article) -> EvaluationResult:
        """Perform full 7-dimension evaluation.

        Args:
            article: Article to evaluate.

        Returns:
            Complete EvaluationResult.
        """
        prompt = self.get_prompt(article)
        system = self.get_system_prompt()

        logger.debug(f"Evaluating {article.relative_path} with full rubric")

        response = await self._llm.complete(
            messages=[LLMMessage.user(prompt)],
            system=system,
            max_tokens=4096,
            temperature=0.0,
        )

        # Parse the response
        result = parse_evaluation_response(
            content=response.content,
            article_path=article.relative_path,
            article_title=article.metadata.title,
            model_used=self._llm.model_name,
        )

        # Add token usage
        result.token_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return result

    async def _evaluate_quick_scan(self, article: Article) -> EvaluationResult:
        """Perform lightweight structural scan.

        Used in tiered mode to quickly identify articles needing deep evaluation.

        Args:
            article: Article to evaluate.

        Returns:
            Minimal EvaluationResult with quick scan data.
        """
        prompt = build_quick_scan_prompt(article)
        system = "You are a technical documentation analyst. Perform quick structural assessments."

        logger.debug(f"Quick scanning {article.relative_path}")

        response = await self._llm.complete(
            messages=[LLMMessage.user(prompt)],
            system=system,
            max_tokens=1024,
            temperature=0.0,
        )

        # Parse quick scan response
        scan_result = parse_quick_scan_response(response.content)

        # Convert to EvaluationResult
        result = quick_scan_to_evaluation_result(
            scan=scan_result,
            article_path=article.relative_path,
            article_title=article.metadata.title,
            model_used=self._llm.model_name,
        )

        result.token_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return result

    def get_prompt(self, article: Article) -> str:
        """Generate the evaluation prompt for an article.

        Args:
            article: Article to evaluate.

        Returns:
            Formatted prompt string.
        """
        if self._quick_scan:
            return build_quick_scan_prompt(article)
        return build_article_evaluation_prompt(article)

    def get_system_prompt(self) -> str:
        """Get the system prompt for article evaluation.

        Returns:
            System prompt string.
        """
        return ARTICLE_EVALUATION_SYSTEM_PROMPT

    async def evaluate_with_deep_followup(
        self,
        article: Article,
        quick_scan_threshold: float = 4.0,
    ) -> EvaluationResult:
        """Evaluate with tiered approach: quick scan, then deep if needed.

        Args:
            article: Article to evaluate.
            quick_scan_threshold: Score threshold for triggering deep evaluation.

        Returns:
            EvaluationResult (from quick or deep evaluation).
        """
        # First, do quick scan
        quick_evaluator = ArticleEvaluator(
            llm_provider=self._llm,
            quick_scan_mode=True,
        )
        quick_result = await quick_evaluator.evaluate(article)

        # Check if deep evaluation is needed
        if (
            quick_result.needs_deep_evaluation
            or quick_result.overall_score < quick_scan_threshold
        ):
            logger.info(
                f"Article {article.relative_path} needs deep evaluation "
                f"(quick score: {quick_result.overall_score:.1f})"
            )
            # Perform full evaluation
            full_result = await self._evaluate_full(article)
            return full_result

        return quick_result


class TieredArticleEvaluator:
    """Evaluator that performs tiered evaluation on article batches.

    First pass: Quick scan all articles
    Second pass: Full evaluation on articles below threshold
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_concurrency: int = 5,
        quick_scan_threshold: float = 4.0,
    ):
        """Initialize the tiered evaluator.

        Args:
            llm_provider: LLM provider for evaluation calls.
            max_concurrency: Maximum concurrent evaluations.
            quick_scan_threshold: Score below which triggers deep evaluation.
        """
        self._llm = llm_provider
        self._max_concurrency = max_concurrency
        self._threshold = quick_scan_threshold

    async def evaluate_batch(
        self,
        articles: list[Article],
        on_progress: callable = None,
    ) -> list[EvaluationResult]:
        """Evaluate articles with tiered approach.

        Args:
            articles: Articles to evaluate.
            on_progress: Optional progress callback.

        Returns:
            List of evaluation results.
        """
        total = len(articles)

        # Phase 1: Quick scan all articles
        logger.info(f"Phase 1: Quick scanning {total} articles")
        quick_evaluator = ArticleEvaluator(
            llm_provider=self._llm,
            max_concurrency=self._max_concurrency,
            quick_scan_mode=True,
        )

        quick_results = await quick_evaluator.evaluate_batch(articles)

        # Identify articles needing deep evaluation
        needs_deep = []
        final_results: list[EvaluationResult | None] = [None] * total

        for i, (article, result) in enumerate(zip(articles, quick_results)):
            if isinstance(result, Exception):
                # Keep failed evaluations as-is (will be handled later)
                final_results[i] = None
                needs_deep.append((i, article))
            elif result.needs_deep_evaluation or result.overall_score < self._threshold:
                needs_deep.append((i, article))
            else:
                final_results[i] = result

        # Phase 2: Deep evaluation on flagged articles
        if needs_deep:
            logger.info(
                f"Phase 2: Deep evaluating {len(needs_deep)} articles "
                f"(below threshold or flagged)"
            )

            full_evaluator = ArticleEvaluator(
                llm_provider=self._llm,
                max_concurrency=self._max_concurrency,
                quick_scan_mode=False,
            )

            deep_articles = [article for _, article in needs_deep]
            deep_results = await full_evaluator.evaluate_batch(deep_articles)

            # Merge deep results back
            for (original_idx, _), deep_result in zip(needs_deep, deep_results):
                if isinstance(deep_result, Exception):
                    # Create a minimal error result
                    final_results[original_idx] = EvaluationResult(
                        article_path=articles[original_idx].relative_path,
                        article_title=articles[original_idx].metadata.title,
                        dimension_scores=[],
                        issues=[],
                        llm_model_used=self._llm.model_name,
                    )
                else:
                    final_results[original_idx] = deep_result

        # Report progress
        if on_progress:
            for i, result in enumerate(final_results):
                if result:
                    on_progress(i + 1, total, result)

        return [r for r in final_results if r is not None]
