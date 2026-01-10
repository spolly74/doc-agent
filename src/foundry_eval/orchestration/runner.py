"""Main orchestration runner for foundry-eval."""

import logging
from pathlib import Path
from typing import Optional

from foundry_eval.config.models import FoundryEvalConfig
from foundry_eval.llm.factory import create_llm_provider
from foundry_eval.models.enums import RunMode
from foundry_eval.models.evaluation import BatchEvaluationResult
from foundry_eval.orchestration.batch_processor import BatchProcessor
from foundry_eval.orchestration.progress import ProgressTracker
from foundry_eval.orchestration.state import StateManager
from foundry_eval.output.csv_writer import CSVWriter
from foundry_eval.output.markdown_writer import MarkdownWriter

logger = logging.getLogger("foundry_eval.orchestration.runner")


class EvaluationRunner:
    """Main orchestration class for running evaluations."""

    def __init__(
        self,
        config: FoundryEvalConfig,
        output_path: Optional[Path] = None,
        state_db_path: Optional[str] = None,
    ):
        """Initialize the evaluation runner.

        Args:
            config: Evaluation configuration.
            output_path: Directory for output files (overrides config).
            state_db_path: Path to SQLite state database.
        """
        self._config = config
        self._output_path = output_path or config.output_path or Path("./eval-output")
        self._state_db_path = state_db_path or config.state_db_path or ".foundry-eval-state.db"

        # Will be initialized lazily
        self._llm_provider = None
        self._state_manager = None
        self._progress_tracker = None
        self._batch_processor = None

    async def _initialize(self) -> None:
        """Initialize components lazily."""
        if self._llm_provider is None:
            self._llm_provider = create_llm_provider(self._config.llm)
            logger.info(f"Using LLM provider: {self._llm_provider.model_name}")

        if self._state_manager is None:
            self._state_manager = StateManager(self._state_db_path)
            await self._state_manager.initialize()

        if self._progress_tracker is None:
            self._progress_tracker = ProgressTracker()

        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(
                llm_provider=self._llm_provider,
                state_manager=self._state_manager,
                progress_tracker=self._progress_tracker,
                max_concurrency=self._config.llm.max_concurrent_requests,
            )

    async def run_full(
        self,
        target_path: Path,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> BatchEvaluationResult:
        """Run full evaluation on all articles.

        Args:
            target_path: Path to documentation directory.
            run_id: Optional run ID (generated if not provided).
            force: If True, ignore cached results.

        Returns:
            BatchEvaluationResult with all results.
        """
        await self._initialize()

        logger.info(f"Starting full evaluation of {target_path}")

        batch_result = await self._batch_processor.process_full(
            target_path=target_path,
            run_id=run_id,
            force=force,
        )

        # Write outputs
        await self._write_outputs(batch_result)

        return batch_result

    async def run_tiered(
        self,
        target_path: Path,
        threshold: Optional[float] = None,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> BatchEvaluationResult:
        """Run tiered evaluation (quick scan + deep where needed).

        Args:
            target_path: Path to documentation directory.
            threshold: Score threshold for triggering deep evaluation.
            run_id: Optional run ID.
            force: If True, ignore cached results.

        Returns:
            BatchEvaluationResult with all results.
        """
        await self._initialize()

        threshold = threshold or self._config.evaluation.quick_scan_threshold

        logger.info(f"Starting tiered evaluation of {target_path} (threshold={threshold})")

        batch_result = await self._batch_processor.process_tiered(
            target_path=target_path,
            threshold=threshold,
            run_id=run_id,
            force=force,
        )

        # Write outputs
        await self._write_outputs(batch_result, threshold=threshold)

        return batch_result

    async def run_jtbd(
        self,
        target_path: Path,
        jtbd_file: Path,
        jtbd_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> BatchEvaluationResult:
        """Run JTBD-scoped evaluation.

        Args:
            target_path: Path to documentation directory.
            jtbd_file: Path to JTBD definitions file.
            jtbd_id: Optional specific JTBD ID to evaluate.
            run_id: Optional run ID.

        Returns:
            BatchEvaluationResult with results.
        """
        await self._initialize()

        logger.info(f"Starting JTBD evaluation (file={jtbd_file}, id={jtbd_id})")

        # TODO: Implement JTBD-scoped evaluation in Phase 6
        raise NotImplementedError("JTBD evaluation will be implemented in Phase 6")

    async def run_samples(
        self,
        articles_repo: Path,
        samples_repo: Path,
        run_id: Optional[str] = None,
    ) -> BatchEvaluationResult:
        """Run code samples validation.

        Args:
            articles_repo: Path to documentation articles.
            samples_repo: Path to code samples repository.
            run_id: Optional run ID.

        Returns:
            BatchEvaluationResult with results.
        """
        await self._initialize()

        logger.info(f"Starting samples validation (articles={articles_repo}, samples={samples_repo})")

        # TODO: Implement samples validation in Phase 6
        raise NotImplementedError("Samples validation will be implemented in Phase 6")

    async def resume_run(self, run_id: str) -> Optional[BatchEvaluationResult]:
        """Resume an incomplete run.

        Args:
            run_id: Run ID to resume.

        Returns:
            BatchEvaluationResult if run can be resumed, None otherwise.
        """
        await self._initialize()

        logger.info(f"Attempting to resume run {run_id}")

        batch_result = await self._batch_processor.resume_run(run_id)

        if batch_result and batch_result.results:
            await self._write_outputs(batch_result)

        return batch_result

    async def _write_outputs(
        self,
        batch_result: BatchEvaluationResult,
        threshold: Optional[float] = None,
    ) -> tuple[Path, Path]:
        """Write evaluation outputs (CSV and Markdown report).

        Args:
            batch_result: Batch evaluation result.
            threshold: Score threshold for report highlighting.

        Returns:
            Tuple of (csv_path, report_path).
        """
        threshold = threshold or self._config.evaluation.threshold

        # Ensure output directory exists
        self._output_path.mkdir(parents=True, exist_ok=True)

        # Write CSV
        csv_writer = CSVWriter(
            output_path=self._output_path,
            filename=self._config.output.csv_filename,
        )
        csv_path = csv_writer.write(batch_result)
        logger.info(f"Wrote CSV output to {csv_path}")

        # Write Markdown report
        md_writer = MarkdownWriter(
            output_path=self._output_path,
            filename=self._config.output.report_filename,
        )
        report_path = md_writer.write(batch_result, threshold=threshold)
        logger.info(f"Wrote report to {report_path}")

        return csv_path, report_path

    def get_output_paths(self) -> dict[str, Path]:
        """Get paths to output files.

        Returns:
            Dictionary with 'csv' and 'report' keys.
        """
        return {
            "csv": self._output_path / self._config.output.csv_filename,
            "report": self._output_path / self._config.output.report_filename,
        }


async def run_evaluation(
    config: FoundryEvalConfig,
    run_mode: RunMode,
    target_path: Path,
    output_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    jtbd_file: Optional[Path] = None,
    jtbd_id: Optional[str] = None,
    samples_repo: Optional[Path] = None,
    run_id: Optional[str] = None,
    force: bool = False,
) -> BatchEvaluationResult:
    """Convenience function to run an evaluation.

    Args:
        config: Evaluation configuration.
        run_mode: Type of evaluation to run.
        target_path: Path to documentation directory.
        output_path: Optional output directory override.
        threshold: Optional score threshold override.
        jtbd_file: Path to JTBD file (for jtbd mode).
        jtbd_id: Specific JTBD ID (for jtbd mode).
        samples_repo: Path to samples repo (for samples mode).
        run_id: Optional run ID.
        force: If True, ignore cached results.

    Returns:
        BatchEvaluationResult with all results.
    """
    runner = EvaluationRunner(
        config=config,
        output_path=output_path,
    )

    if run_mode == RunMode.FULL:
        return await runner.run_full(
            target_path=target_path,
            run_id=run_id,
            force=force,
        )
    elif run_mode == RunMode.TIERED:
        return await runner.run_tiered(
            target_path=target_path,
            threshold=threshold,
            run_id=run_id,
            force=force,
        )
    elif run_mode == RunMode.JTBD:
        if not jtbd_file:
            raise ValueError("JTBD file path is required for JTBD mode")
        return await runner.run_jtbd(
            target_path=target_path,
            jtbd_file=jtbd_file,
            jtbd_id=jtbd_id,
            run_id=run_id,
        )
    elif run_mode == RunMode.SAMPLES:
        if not samples_repo:
            raise ValueError("Samples repo path is required for samples mode")
        return await runner.run_samples(
            articles_repo=target_path,
            samples_repo=samples_repo,
            run_id=run_id,
        )
    else:
        raise ValueError(f"Unknown run mode: {run_mode}")
