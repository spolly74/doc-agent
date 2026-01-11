"""Background worker for processing evaluation jobs."""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from doc_agent.config import load_config
from doc_agent.jobs.manager import JobManager
from doc_agent.jobs.models import JobStatus
from doc_agent.orchestration.runner import EvaluationRunner

logger = logging.getLogger("doc_agent.jobs.worker")

# PID file location
PID_FILE = Path.home() / ".doc-agent" / "worker.pid"

# Worker settings
POLL_INTERVAL = 2  # seconds between checking for jobs
IDLE_TIMEOUT = 300  # 5 minutes of idle before auto-exit


class Worker:
    """Background worker that processes evaluation jobs."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the worker.

        Args:
            db_path: Optional path to jobs database.
        """
        self._manager = JobManager(db_path)
        self._running = True
        self._current_job_id: str | None = None
        self._last_job_time = time.time()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def _run_job(self, job_id: str) -> None:
        """Run a single evaluation job.

        Args:
            job_id: The job ID to process.
        """
        job = self._manager.get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found")
            return

        self._current_job_id = job_id
        config = job.config

        try:
            # Mark as running
            self._manager.update_job_status(job_id, JobStatus.RUNNING)

            logger.info(f"Starting job {job_id}: {config.target_path}")

            # Load configuration
            cfg = load_config(
                target=Path(config.target_path),
                output=Path(config.output_path),
                provider=config.provider,
                model=config.model,
                concurrency=config.concurrency,
                threshold=config.threshold,
            )

            # Create runner with progress callback
            runner = EvaluationRunner(
                config=cfg,
                output_path=Path(config.output_path),
            )

            # Count articles for progress
            target_path = Path(config.target_path)
            article_count = sum(1 for f in target_path.rglob("*.md")
                               if f.name.lower() not in ("readme.md", "changelog.md", "contributing.md"))
            self._manager.update_job_progress(job_id, 0, article_count)

            # Run evaluation
            if config.mode == "tiered":
                batch_result = await runner.run_tiered(
                    target_path=target_path,
                    threshold=config.threshold,
                )
            else:
                batch_result = await runner.run_full(
                    target_path=target_path,
                )

            # Store result summary
            result_summary = {
                "total_articles": batch_result.total_articles,
                "evaluated_articles": batch_result.evaluated_articles,
                "failed_articles": batch_result.failed_articles,
                "average_score": batch_result.average_score,
                "below_threshold_count": len(batch_result.articles_below_threshold),
                "duration_seconds": batch_result.duration_seconds,
                "output_path": config.output_path,
            }
            self._manager.set_job_result(job_id, result_summary)
            self._manager.update_job_progress(job_id, article_count, article_count)

            # Mark complete
            self._manager.update_job_status(job_id, JobStatus.COMPLETE)
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            self._manager.update_job_status(job_id, JobStatus.FAILED, str(e))

        finally:
            self._current_job_id = None
            self._last_job_time = time.time()

    async def run(self, once: bool = False) -> None:
        """Run the worker loop.

        Args:
            once: If True, process one job and exit (useful for testing).
        """
        logger.info("Worker started")

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        while self._running:
            # Check for pending job
            job = self._manager.get_pending_job()

            if job:
                await self._run_job(job.job_id)
                if once:
                    break
            else:
                # Check idle timeout
                idle_time = time.time() - self._last_job_time
                if idle_time > IDLE_TIMEOUT:
                    logger.info(f"No jobs for {IDLE_TIMEOUT}s, shutting down")
                    break

                # Wait before checking again
                await asyncio.sleep(POLL_INTERVAL)

        logger.info("Worker stopped")


def is_worker_running() -> bool:
    """Check if a worker process is already running.

    Returns:
        True if a worker is running.
    """
    if not PID_FILE.exists():
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        # Invalid PID or process doesn't exist
        PID_FILE.unlink(missing_ok=True)
        return False


def write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def remove_pid_file() -> None:
    """Remove the PID file."""
    PID_FILE.unlink(missing_ok=True)


def start_worker(db_path: Path | str | None = None, once: bool = False) -> None:
    """Start the worker process.

    Args:
        db_path: Optional path to jobs database.
        once: If True, process one job and exit.
    """
    if is_worker_running():
        logger.warning("Worker is already running")
        return

    try:
        write_pid_file()
        worker = Worker(db_path)
        asyncio.run(worker.run(once=once))
    finally:
        remove_pid_file()


def main():
    """Entry point for doc-agent-worker command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Doc Agent background worker")
    parser.add_argument("--once", action="store_true", help="Process one job and exit")
    parser.add_argument("--db", type=str, help="Path to jobs database")
    args = parser.parse_args()

    start_worker(db_path=args.db, once=args.once)


if __name__ == "__main__":
    main()
