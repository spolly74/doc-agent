"""Tests for the orchestration module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_agent.models.enums import Dimension
from doc_agent.models.evaluation import DimensionScore, EvaluationResult
from doc_agent.orchestration.progress import ProgressTracker
from doc_agent.orchestration.state import StateManager


class TestStateManager:
    """Tests for StateManager."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def state_manager(self, temp_db):
        """Create a state manager with temporary database."""
        return StateManager(db_path=temp_db)

    @pytest.fixture
    def sample_result(self):
        """Create a sample evaluation result."""
        return EvaluationResult(
            article_path="test/article.md",
            article_title="Test Article",
            dimension_scores=[
                DimensionScore(
                    dimension=Dimension.STRUCTURE,
                    score=5.0,
                    confidence=0.9,
                    rationale="Good structure",
                ),
            ],
            issues=[],
            llm_model_used="test-model",
        )

    @pytest.mark.asyncio
    async def test_initialize(self, state_manager):
        """Test database initialization."""
        await state_manager.initialize()
        assert state_manager._initialized is True

        # Calling again should be idempotent
        await state_manager.initialize()
        assert state_manager._initialized is True

    @pytest.mark.asyncio
    async def test_start_run(self, state_manager):
        """Test starting a run."""
        await state_manager.start_run(
            run_id="test-run-1",
            run_mode="full",
            target_path="/path/to/docs",
        )

        # Should be able to find incomplete run
        run_id = await state_manager.get_incomplete_run(
            target_path="/path/to/docs",
            run_mode="full",
        )
        assert run_id == "test-run-1"

    @pytest.mark.asyncio
    async def test_complete_run(self, state_manager):
        """Test completing a run."""
        await state_manager.start_run(
            run_id="test-run-2",
            run_mode="full",
            target_path="/path/to/docs",
        )
        await state_manager.complete_run("test-run-2")

        # Should no longer find as incomplete
        run_id = await state_manager.get_incomplete_run(
            target_path="/path/to/docs",
            run_mode="full",
        )
        assert run_id is None

    @pytest.mark.asyncio
    async def test_cache_and_retrieve_result(self, state_manager, sample_result):
        """Test caching and retrieving results."""
        run_id = "test-run-cache"
        article_content = "# Test Article\n\nContent here."

        await state_manager.start_run(
            run_id=run_id,
            run_mode="full",
            target_path="/docs",
        )

        # Cache result
        await state_manager.cache_result(
            run_id=run_id,
            article_path="test/article.md",
            article_content=article_content,
            result=sample_result,
        )

        # Retrieve result
        cached = await state_manager.get_cached_result(
            run_id=run_id,
            article_path="test/article.md",
            article_content=article_content,
        )

        assert cached is not None
        assert cached.article_path == sample_result.article_path
        assert len(cached.dimension_scores) == 1

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_content_change(
        self,
        state_manager,
        sample_result,
    ):
        """Test that cache is invalidated when content changes."""
        run_id = "test-run-invalidate"
        original_content = "# Original\n\nContent."
        modified_content = "# Modified\n\nDifferent content."

        await state_manager.start_run(
            run_id=run_id,
            run_mode="full",
            target_path="/docs",
        )

        # Cache with original content
        await state_manager.cache_result(
            run_id=run_id,
            article_path="test/article.md",
            article_content=original_content,
            result=sample_result,
        )

        # Try to retrieve with modified content
        cached = await state_manager.get_cached_result(
            run_id=run_id,
            article_path="test/article.md",
            article_content=modified_content,
        )

        assert cached is None

    @pytest.mark.asyncio
    async def test_get_evaluated_paths(self, state_manager, sample_result):
        """Test getting all evaluated paths."""
        run_id = "test-run-paths"

        await state_manager.start_run(
            run_id=run_id,
            run_mode="full",
            target_path="/docs",
        )

        # Cache multiple results
        for i in range(3):
            result = EvaluationResult(
                article_path=f"test/article{i}.md",
                article_title=f"Article {i}",
                dimension_scores=[],
                issues=[],
                llm_model_used="test",
            )
            await state_manager.cache_result(
                run_id=run_id,
                article_path=result.article_path,
                article_content=f"Content {i}",
                result=result,
            )

        paths = await state_manager.get_evaluated_paths(run_id)

        assert len(paths) == 3
        assert "test/article0.md" in paths
        assert "test/article1.md" in paths
        assert "test/article2.md" in paths

    @pytest.mark.asyncio
    async def test_get_run_results(self, state_manager, sample_result):
        """Test getting all results for a run."""
        run_id = "test-run-results"

        await state_manager.start_run(
            run_id=run_id,
            run_mode="full",
            target_path="/docs",
        )

        await state_manager.cache_result(
            run_id=run_id,
            article_path="test/article.md",
            article_content="Content",
            result=sample_result,
        )

        results = await state_manager.get_run_results(run_id)

        assert len(results) == 1
        assert results[0].article_path == "test/article.md"


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_set_total(self):
        """Test setting total items."""
        tracker = ProgressTracker(show_progress=False)
        tracker.set_total(100)

        assert tracker._total == 100
        assert tracker._completed == 0

    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker(show_progress=False)
        tracker.set_total(10)
        tracker.update(5, "test/article.md")

        assert tracker._completed == 5

    def test_update_with_error(self):
        """Test updating with error."""
        tracker = ProgressTracker(show_progress=False)
        tracker.set_total(10)
        tracker.update(5, "test/article.md", is_error=True)

        assert tracker._completed == 5
        assert tracker._errors == 1

    def test_advance(self):
        """Test advancing progress."""
        tracker = ProgressTracker(show_progress=False)
        tracker.set_total(10)
        tracker.advance(3)

        assert tracker._completed == 3

    def test_finish(self, capsys):
        """Test finishing progress."""
        tracker = ProgressTracker(show_progress=False)
        tracker.set_total(10)
        tracker._completed = 10
        tracker._errors = 2
        tracker.finish()

        # Progress is finished, errors are tracked
        assert tracker._errors == 2

    def test_display_summary_table(self):
        """Test displaying summary table."""
        tracker = ProgressTracker(show_progress=False)

        results = [
            EvaluationResult(
                article_path="test1.md",
                article_title="Test 1",
                dimension_scores=[
                    DimensionScore(
                        dimension=Dimension.STRUCTURE,
                        score=5.0,
                        confidence=0.9,
                        rationale="Good",
                    ),
                ],
                issues=[],
                llm_model_used="test",
            ),
            EvaluationResult(
                article_path="test2.md",
                article_title="Test 2",
                dimension_scores=[
                    DimensionScore(
                        dimension=Dimension.STRUCTURE,
                        score=3.0,
                        confidence=0.9,
                        rationale="Needs work",
                    ),
                ],
                issues=[],
                llm_model_used="test",
            ),
        ]

        # Should not raise
        tracker.display_summary_table(results, threshold=4.0)

    def test_display_summary_table_empty(self):
        """Test displaying summary table with no results."""
        tracker = ProgressTracker(show_progress=False)

        # Should handle empty results
        tracker.display_summary_table([])
