"""Batch processing for article evaluation with state management."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from foundry_eval.context.toc_parser import TOCParser
from foundry_eval.evaluators.article import ArticleEvaluator, TieredArticleEvaluator
from foundry_eval.llm.protocol import LLMProvider
from foundry_eval.models.article import Article
from foundry_eval.models.enums import RunMode
from foundry_eval.models.evaluation import BatchEvaluationResult, EvaluationResult
from foundry_eval.orchestration.progress import ProgressTracker
from foundry_eval.orchestration.state import StateManager

logger = logging.getLogger("foundry_eval.orchestration.batch")


class BatchProcessor:
    """Processes batches of articles for evaluation with state management."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        state_manager: Optional[StateManager] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        max_concurrency: int = 5,
    ):
        """Initialize the batch processor.

        Args:
            llm_provider: LLM provider for evaluations.
            state_manager: Optional state manager for resume capability.
            progress_tracker: Optional progress tracker for UI updates.
            max_concurrency: Maximum concurrent evaluations.
        """
        self._llm = llm_provider
        self._state = state_manager or StateManager()
        self._progress = progress_tracker or ProgressTracker()
        self._max_concurrency = max_concurrency

    async def process_full(
        self,
        target_path: Path,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> BatchEvaluationResult:
        """Process all articles in full evaluation mode.

        Args:
            target_path: Path to documentation directory.
            run_id: Optional run ID (generated if not provided).
            force: If True, ignore cached results.

        Returns:
            BatchEvaluationResult with all results.
        """
        run_id = run_id or self._generate_run_id()

        # Initialize state
        await self._state.start_run(
            run_id=run_id,
            run_mode=RunMode.FULL.value,
            target_path=str(target_path),
        )

        batch_result = BatchEvaluationResult(
            run_id=run_id,
            run_mode=RunMode.FULL.value,
            started_at=datetime.now(timezone.utc),
            target_path=str(target_path),
        )

        try:
            # Load articles
            toc_parser = TOCParser(target_path)
            articles = await toc_parser.load_all_articles()
            batch_result.total_articles = len(articles)

            logger.info(f"Found {len(articles)} articles to evaluate")
            self._progress.set_total(len(articles))

            # Get already evaluated paths (for resume)
            evaluated_paths = set()
            if not force:
                evaluated_paths = await self._state.get_evaluated_paths(run_id)
                if evaluated_paths:
                    logger.info(f"Resuming: {len(evaluated_paths)} already evaluated")

            # Filter out already evaluated
            articles_to_evaluate = [
                a for a in articles
                if a.relative_path not in evaluated_paths
            ]

            # Create evaluator
            evaluator = ArticleEvaluator(
                llm_provider=self._llm,
                max_concurrency=self._max_concurrency,
            )

            # Process articles
            results = await self._process_articles(
                evaluator=evaluator,
                articles=articles_to_evaluate,
                run_id=run_id,
                batch_result=batch_result,
            )

            # Add cached results
            if evaluated_paths:
                cached_results = await self._state.get_run_results(run_id)
                results.extend(cached_results)

            batch_result.results = results
            batch_result.evaluated_articles = len(results)
            batch_result.completed_at = datetime.now(timezone.utc)

            await self._state.complete_run(run_id)

        except Exception as e:
            await self._state.fail_run(run_id, str(e))
            raise

        return batch_result

    async def process_tiered(
        self,
        target_path: Path,
        threshold: float = 4.0,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> BatchEvaluationResult:
        """Process articles in tiered mode (quick scan + deep where needed).

        Args:
            target_path: Path to documentation directory.
            threshold: Score threshold for triggering deep evaluation.
            run_id: Optional run ID.
            force: If True, ignore cached results.

        Returns:
            BatchEvaluationResult with all results.
        """
        run_id = run_id or self._generate_run_id()

        await self._state.start_run(
            run_id=run_id,
            run_mode=RunMode.TIERED.value,
            target_path=str(target_path),
            config={"threshold": threshold},
        )

        batch_result = BatchEvaluationResult(
            run_id=run_id,
            run_mode=RunMode.TIERED.value,
            started_at=datetime.now(timezone.utc),
            target_path=str(target_path),
        )

        try:
            # Load articles
            toc_parser = TOCParser(target_path)
            articles = await toc_parser.load_all_articles()
            batch_result.total_articles = len(articles)

            logger.info(f"Found {len(articles)} articles for tiered evaluation")
            self._progress.set_total(len(articles))

            # Create tiered evaluator
            evaluator = TieredArticleEvaluator(
                llm_provider=self._llm,
                max_concurrency=self._max_concurrency,
                quick_scan_threshold=threshold,
            )

            # Process with tiered approach
            def on_progress(completed: int, total: int, result: EvaluationResult):
                self._progress.update(completed, result.article_path)

            results = await evaluator.evaluate_batch(
                articles=articles,
                on_progress=on_progress,
            )

            # Cache results
            for article, result in zip(articles, results):
                await self._state.cache_result(
                    run_id=run_id,
                    article_path=article.relative_path,
                    article_content=article.content,
                    result=result,
                )

            batch_result.results = results
            batch_result.evaluated_articles = len(results)
            batch_result.completed_at = datetime.now(timezone.utc)

            await self._state.complete_run(run_id)

        except Exception as e:
            await self._state.fail_run(run_id, str(e))
            raise

        return batch_result

    async def _process_articles(
        self,
        evaluator: ArticleEvaluator,
        articles: list[Article],
        run_id: str,
        batch_result: BatchEvaluationResult,
    ) -> list[EvaluationResult]:
        """Process a list of articles with progress tracking.

        Args:
            evaluator: Article evaluator to use.
            articles: Articles to evaluate.
            run_id: Run identifier.
            batch_result: Batch result to update.

        Returns:
            List of evaluation results.
        """
        results = []

        def on_progress(completed: int, total: int, result):
            if isinstance(result, Exception):
                batch_result.failed_articles += 1
                self._progress.update(
                    completed,
                    f"Failed: {result}",
                    is_error=True,
                )
            else:
                self._progress.update(completed, result.article_path)

        raw_results = await evaluator.evaluate_batch(
            articles=articles,
            on_progress=on_progress,
        )

        # Process results and cache
        for article, result in zip(articles, raw_results):
            if isinstance(result, Exception):
                batch_result.failed_articles += 1
                logger.error(f"Failed to evaluate {article.relative_path}: {result}")
            else:
                results.append(result)
                batch_result.total_input_tokens += (
                    result.token_usage.get("input_tokens", 0)
                    if result.token_usage
                    else 0
                )
                batch_result.total_output_tokens += (
                    result.token_usage.get("output_tokens", 0)
                    if result.token_usage
                    else 0
                )

                # Cache result
                await self._state.cache_result(
                    run_id=run_id,
                    article_path=article.relative_path,
                    article_content=article.content,
                    result=result,
                )

        return results

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return f"run-{uuid.uuid4().hex[:8]}"

    async def resume_run(self, run_id: str) -> Optional[BatchEvaluationResult]:
        """Resume an incomplete run.

        Args:
            run_id: Run ID to resume.

        Returns:
            BatchEvaluationResult if run can be resumed, None otherwise.
        """
        # Get existing results
        results = await self._state.get_run_results(run_id)

        if not results:
            return None

        # Get run info
        # TODO: Load run metadata from state

        return BatchEvaluationResult(
            run_id=run_id,
            run_mode="resumed",
            started_at=datetime.now(timezone.utc),
            target_path="",
            results=results,
            evaluated_articles=len(results),
        )
