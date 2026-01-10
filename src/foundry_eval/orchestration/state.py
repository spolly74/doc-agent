"""SQLite-based state management for evaluation runs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import hashlib

import aiosqlite

from foundry_eval.models.evaluation import EvaluationResult

logger = logging.getLogger("foundry_eval.orchestration.state")


class StateManager:
    """Manages evaluation state in SQLite for resume capability.

    State is stored in a local SQLite database, allowing evaluation runs
    to be resumed after interruption.
    """

    def __init__(self, db_path: str = ".foundry-eval-state.db"):
        """Initialize the state manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        async with aiosqlite.connect(self._db_path) as db:
            # Create tables
            await db.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    run_mode TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    config_json TEXT,
                    status TEXT DEFAULT 'running'
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    article_path TEXT NOT NULL,
                    article_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id),
                    UNIQUE (run_id, article_path)
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_run_id
                ON evaluation_results(run_id)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_article_path
                ON evaluation_results(article_path)
            """)

            await db.commit()

        self._initialized = True
        logger.debug(f"State database initialized at {self._db_path}")

    async def start_run(
        self,
        run_id: str,
        run_mode: str,
        target_path: str,
        config: Optional[dict] = None,
    ) -> None:
        """Record the start of an evaluation run.

        Args:
            run_id: Unique identifier for this run.
            run_mode: Run mode (full, tiered, jtbd, samples).
            target_path: Target documentation path.
            config: Optional configuration dict.
        """
        await self.initialize()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs
                (run_id, run_mode, target_path, started_at, config_json, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    run_mode,
                    target_path,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(config) if config else None,
                ),
            )
            await db.commit()

        logger.info(f"Started evaluation run {run_id}")

    async def complete_run(self, run_id: str) -> None:
        """Mark an evaluation run as complete.

        Args:
            run_id: Run identifier.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE evaluation_runs
                SET completed_at = ?, status = 'completed'
                WHERE run_id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
            await db.commit()

        logger.info(f"Completed evaluation run {run_id}")

    async def fail_run(self, run_id: str, error: str) -> None:
        """Mark an evaluation run as failed.

        Args:
            run_id: Run identifier.
            error: Error message.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE evaluation_runs
                SET completed_at = ?, status = 'failed'
                WHERE run_id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
            await db.commit()

        logger.error(f"Evaluation run {run_id} failed: {error}")

    async def cache_result(
        self,
        run_id: str,
        article_path: str,
        article_content: str,
        result: EvaluationResult,
    ) -> None:
        """Cache an evaluation result.

        Args:
            run_id: Run identifier.
            article_path: Path to the article.
            article_content: Content of the article (for hash).
            result: Evaluation result to cache.
        """
        await self.initialize()

        # Compute content hash
        content_hash = hashlib.sha256(article_content.encode()).hexdigest()[:16]

        # Serialize result
        result_json = result.model_dump_json()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO evaluation_results
                (run_id, article_path, article_hash, result_json, evaluated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    article_path,
                    content_hash,
                    result_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

    async def get_cached_result(
        self,
        run_id: str,
        article_path: str,
        article_content: str,
    ) -> Optional[EvaluationResult]:
        """Get a cached evaluation result if available and valid.

        Args:
            run_id: Run identifier.
            article_path: Path to the article.
            article_content: Current content of the article.

        Returns:
            Cached result if valid, None otherwise.
        """
        await self.initialize()

        # Compute current content hash
        current_hash = hashlib.sha256(article_content.encode()).hexdigest()[:16]

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT result_json, article_hash FROM evaluation_results
                WHERE run_id = ? AND article_path = ?
                """,
                (run_id, article_path),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        result_json, stored_hash = row

        # Verify content hasn't changed
        if stored_hash != current_hash:
            logger.debug(
                f"Cache miss for {article_path}: content changed"
            )
            return None

        # Parse and return result
        try:
            return EvaluationResult.model_validate_json(result_json)
        except Exception as e:
            logger.warning(f"Failed to parse cached result: {e}")
            return None

    async def get_evaluated_paths(self, run_id: str) -> set[str]:
        """Get all article paths that have been evaluated in a run.

        Args:
            run_id: Run identifier.

        Returns:
            Set of evaluated article paths.
        """
        await self.initialize()

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT article_path FROM evaluation_results WHERE run_id = ?",
                (run_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        return {row[0] for row in rows}

    async def get_run_results(self, run_id: str) -> list[EvaluationResult]:
        """Get all results for a run.

        Args:
            run_id: Run identifier.

        Returns:
            List of evaluation results.
        """
        await self.initialize()

        results = []
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT result_json FROM evaluation_results WHERE run_id = ?",
                (run_id,),
            ) as cursor:
                async for row in cursor:
                    try:
                        result = EvaluationResult.model_validate_json(row[0])
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to parse result: {e}")

        return results

    async def get_incomplete_run(
        self,
        target_path: str,
        run_mode: str,
    ) -> Optional[str]:
        """Find an incomplete run for the given target.

        Args:
            target_path: Target documentation path.
            run_mode: Run mode.

        Returns:
            Run ID if found, None otherwise.
        """
        await self.initialize()

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT run_id FROM evaluation_runs
                WHERE target_path = ? AND run_mode = ? AND status = 'running'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (target_path, run_mode),
            ) as cursor:
                row = await cursor.fetchone()

        return row[0] if row else None

    async def cleanup_old_runs(self, days: int = 30) -> int:
        """Remove old completed runs.

        Args:
            days: Remove runs older than this many days.

        Returns:
            Number of runs removed.
        """
        await self.initialize()

        cutoff = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            # Get old run IDs
            async with db.execute(
                """
                SELECT run_id FROM evaluation_runs
                WHERE completed_at < datetime(?, '-' || ? || ' days')
                AND status IN ('completed', 'failed')
                """,
                (cutoff, days),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return 0

            run_ids = [row[0] for row in rows]

            # Delete results first
            await db.execute(
                f"""
                DELETE FROM evaluation_results
                WHERE run_id IN ({','.join('?' * len(run_ids))})
                """,
                run_ids,
            )

            # Delete runs
            await db.execute(
                f"""
                DELETE FROM evaluation_runs
                WHERE run_id IN ({','.join('?' * len(run_ids))})
                """,
                run_ids,
            )

            await db.commit()

        return len(run_ids)
