"""Job manager for creating, querying, and updating jobs."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from doc_agent.jobs.models import Job, JobConfig, JobStatus

logger = logging.getLogger("doc_agent.jobs.manager")

# Default database path
DEFAULT_DB_PATH = Path.home() / ".doc-agent" / "jobs.db"


class JobManager:
    """Manages job queue operations using SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the job manager.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.doc-agent/jobs.db
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    config_json TEXT NOT NULL,
                    progress_current INTEGER DEFAULT 0,
                    progress_total INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    result_summary_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
            """)
            conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert a database row to a Job object."""
        config = JobConfig(**json.loads(row["config_json"]))
        result_summary = None
        if row["result_summary_json"]:
            result_summary = json.loads(row["result_summary_json"])

        return Job(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            config=config,
            progress_current=row["progress_current"],
            progress_total=row["progress_total"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            error_message=row["error_message"],
            result_summary=result_summary,
        )

    def create_job(self, config: JobConfig) -> Job:
        """Create a new job in the queue.

        Args:
            config: Job configuration.

        Returns:
            The created Job object.
        """
        job_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow()

        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            config=config,
            created_at=now,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, config_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, job.status.value, config.model_dump_json(), now.isoformat()),
            )
            conn.commit()

        logger.info(f"Created job {job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID.

        Args:
            job_id: The job ID.

        Returns:
            Job object or None if not found.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        if row is None:
            return None
        return self._row_to_job(row)

    def get_pending_job(self) -> Optional[Job]:
        """Get the oldest pending job.

        Returns:
            Job object or None if no pending jobs.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (JobStatus.PENDING.value,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_job(row)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs, optionally filtered by status.

        Args:
            status: Optional status filter.
            limit: Maximum number of jobs to return.

        Returns:
            List of Job objects.
        """
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [self._row_to_job(row) for row in rows]

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status.

        Args:
            job_id: The job ID.
            status: New status.
            error_message: Optional error message for failed jobs.
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            if status == JobStatus.RUNNING:
                conn.execute(
                    "UPDATE jobs SET status = ?, started_at = ? WHERE job_id = ?",
                    (status.value, now, job_id),
                )
            elif status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, completed_at = ?, error_message = ?
                    WHERE job_id = ?
                    """,
                    (status.value, now, error_message, job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ? WHERE job_id = ?",
                    (status.value, job_id),
                )
            conn.commit()

        logger.info(f"Job {job_id} status -> {status.value}")

    def update_job_progress(
        self,
        job_id: str,
        current: int,
        total: int,
    ) -> None:
        """Update job progress.

        Args:
            job_id: The job ID.
            current: Current progress count.
            total: Total count.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET progress_current = ?, progress_total = ?
                WHERE job_id = ?
                """,
                (current, total, job_id),
            )
            conn.commit()

    def set_job_result(
        self,
        job_id: str,
        result_summary: dict,
    ) -> None:
        """Set the job result summary.

        Args:
            job_id: The job ID.
            result_summary: Summary dict with evaluation results.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET result_summary_json = ? WHERE job_id = ?",
                (json.dumps(result_summary), job_id),
            )
            conn.commit()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job.

        Args:
            job_id: The job ID.

        Returns:
            True if job was cancelled, False if not found or already terminal.
        """
        job = self.get_job(job_id)
        if job is None or job.is_terminal:
            return False

        self.update_job_status(job_id, JobStatus.CANCELLED)
        return True

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """Remove jobs older than specified days.

        Args:
            days: Number of days to keep.

        Returns:
            Number of jobs deleted.
        """
        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM jobs
                WHERE created_at < ? AND status IN (?, ?, ?)
                """,
                (cutoff, JobStatus.COMPLETE.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value),
            )
            conn.commit()
            return cursor.rowcount

    def has_active_jobs(self) -> bool:
        """Check if there are any pending or running jobs.

        Returns:
            True if there are active jobs.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count FROM jobs
                WHERE status IN (?, ?)
                """,
                (JobStatus.PENDING.value, JobStatus.RUNNING.value),
            ).fetchone()

        return row["count"] > 0
