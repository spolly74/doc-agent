"""Base evaluator abstract class."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from doc_agent.llm.protocol import LLMProvider
from doc_agent.models.article import Article

T = TypeVar("T")  # Result type

logger = logging.getLogger("doc_agent.evaluators.base")


class BaseEvaluator(ABC, Generic[T]):
    """Abstract base class for all evaluators.

    Evaluators are responsible for assessing some aspect of documentation
    quality. They use an LLM provider to perform the evaluation and return
    typed results.

    Type Parameters:
        T: The type of result this evaluator produces.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_concurrency: int = 5,
    ):
        """Initialize the evaluator.

        Args:
            llm_provider: LLM provider for evaluation calls.
            max_concurrency: Maximum concurrent evaluations.
        """
        self._llm = llm_provider
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def model_name(self) -> str:
        """Get the name of the LLM model being used."""
        return self._llm.model_name

    @abstractmethod
    async def evaluate(self, article: Article) -> T:
        """Evaluate a single article.

        Args:
            article: Article to evaluate.

        Returns:
            Evaluation result of type T.
        """
        ...

    @abstractmethod
    def get_prompt(self, article: Article) -> str:
        """Generate the evaluation prompt for an article.

        Args:
            article: Article to evaluate.

        Returns:
            Formatted prompt string.
        """
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this evaluator.

        Returns:
            System prompt string.
        """
        ...

    async def evaluate_batch(
        self,
        articles: list[Article],
        on_progress: callable = None,
    ) -> list[T | Exception]:
        """Evaluate multiple articles with controlled concurrency.

        Args:
            articles: List of articles to evaluate.
            on_progress: Optional callback called after each evaluation.
                         Signature: (completed: int, total: int, result: T | Exception) -> None

        Returns:
            List of results (or Exceptions for failed evaluations).
        """
        total = len(articles)

        async def evaluate_with_semaphore(
            index: int,
            article: Article,
        ) -> tuple[int, T | Exception]:
            """Evaluate with semaphore and error handling."""
            async with self._semaphore:
                try:
                    result = await self.evaluate(article)
                    return index, result
                except Exception as e:
                    logger.error(
                        f"Evaluation failed for {article.relative_path}: {type(e).__name__}: {e}"
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Full exception for {article.relative_path}", exc_info=True)
                    return index, e

        # Create tasks for all articles
        tasks = [
            evaluate_with_semaphore(i, article)
            for i, article in enumerate(articles)
        ]

        # Process results as they complete
        results: list[T | Exception | None] = [None] * total
        completed = 0

        for coro in asyncio.as_completed(tasks):
            index, result = await coro
            results[index] = result
            completed += 1

            if on_progress:
                try:
                    on_progress(completed, total, result)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

        return results  # type: ignore

    async def evaluate_with_timeout(
        self,
        article: Article,
        timeout_seconds: float = 60.0,
    ) -> T:
        """Evaluate an article with a timeout.

        Args:
            article: Article to evaluate.
            timeout_seconds: Maximum time to wait for evaluation.

        Returns:
            Evaluation result.

        Raises:
            asyncio.TimeoutError: If evaluation takes too long.
        """
        return await asyncio.wait_for(
            self.evaluate(article),
            timeout=timeout_seconds,
        )
