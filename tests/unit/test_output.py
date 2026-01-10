"""Tests for output writers."""

import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from foundry_eval.models.enums import Dimension, Severity
from foundry_eval.models.evaluation import (
    BatchEvaluationResult,
    DimensionScore,
    EvaluationResult,
    Issue,
)
from foundry_eval.output.csv_writer import CSVWriter, CSV_COLUMNS
from foundry_eval.output.markdown_writer import MarkdownWriter, ReportStatistics


@pytest.fixture
def sample_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        article_path="docs/test-article.md",
        article_title="Test Article",
        dimension_scores=[
            DimensionScore(
                dimension=Dimension.PATTERN_COMPLIANCE,
                score=5.0,
                rationale="Good pattern compliance",
            ),
            DimensionScore(
                dimension=Dimension.DEV_FOCUS,
                score=4.0,
                rationale="Developer focused",
            ),
            DimensionScore(
                dimension=Dimension.CODE_QUALITY,
                score=6.0,
                rationale="High quality code",
            ),
            DimensionScore(
                dimension=Dimension.CODE_COMPLETENESS,
                score=3.0,
                rationale="Missing some examples",
            ),
            DimensionScore(
                dimension=Dimension.STRUCTURE,
                score=5.0,
                rationale="Well structured",
            ),
            DimensionScore(
                dimension=Dimension.ACCURACY,
                score=6.0,
                rationale="Accurate content",
            ),
        ],
        issues=[
            Issue(
                id=1,
                dimension=Dimension.CODE_COMPLETENESS,
                severity=Severity.MEDIUM,
                title="Missing code example",
                description="Article lacks a complete code example",
            ),
            Issue(
                id=2,
                dimension=Dimension.DEV_FOCUS,
                severity=Severity.LOW,
                title="Minor formatting issue",
                description="Some inconsistent formatting",
            ),
        ],
    )


@pytest.fixture
def sample_batch_result(sample_result):
    """Create a sample batch result."""
    return BatchEvaluationResult(
        run_id="test-run-001",
        run_mode="full",
        started_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
        target_path="/docs",
        results=[sample_result],
        total_articles=1,
        evaluated_articles=1,
    )


class TestCSVWriter:
    """Tests for CSVWriter."""

    def test_csv_columns_defined(self):
        """Test that CSV columns are properly defined."""
        assert len(CSV_COLUMNS) > 0
        assert "article_path" in CSV_COLUMNS
        assert "overall_score" in CSV_COLUMNS
        assert "pattern_compliance" in CSV_COLUMNS

    def test_write_results(self, sample_result):
        """Test writing results to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = CSVWriter(output_path)

            csv_path = writer.write_results([sample_result])

            assert csv_path.exists()
            assert csv_path.name == "scores.csv"

            # Read and verify CSV content
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["article_path"] == "docs/test-article.md"
            assert rows[0]["title"] == "Test Article"
            assert float(rows[0]["overall_score"]) > 0

    def test_write_batch_result(self, sample_batch_result):
        """Test writing batch result to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = CSVWriter(output_path)

            csv_path = writer.write(sample_batch_result)

            assert csv_path.exists()

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1

    def test_append_result(self, sample_result):
        """Test appending results to existing CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = CSVWriter(output_path)

            # Write first result
            writer.append_result(sample_result)

            # Create another result
            result2 = EvaluationResult(
                article_path="docs/another-article.md",
                article_title="Another Article",
                dimension_scores=[],
                issues=[],
            )

            # Append second result
            writer.append_result(result2)

            # Verify both are in CSV
            with open(writer.filepath, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

    def test_custom_filename(self):
        """Test using custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = CSVWriter(output_path, filename="custom.csv")

            assert writer.filepath.name == "custom.csv"


class TestReportStatistics:
    """Tests for ReportStatistics."""

    def test_empty_results(self):
        """Test statistics with empty results."""
        stats = ReportStatistics([])

        assert stats.total_articles == 0
        assert stats.average_score == 0.0
        assert stats.below_threshold == 0
        assert stats.dimension_averages == {}

    def test_calculate_statistics(self, sample_result):
        """Test statistics calculation."""
        stats = ReportStatistics([sample_result], threshold=4.0)

        assert stats.total_articles == 1
        assert stats.average_score > 0
        assert stats.min_score > 0
        assert stats.max_score > 0

    def test_issues_by_severity(self, sample_result):
        """Test counting issues by severity."""
        stats = ReportStatistics([sample_result])

        assert stats.issues_by_severity["medium"] == 1
        assert stats.issues_by_severity["low"] == 1
        assert stats.issues_by_severity["critical"] == 0
        assert stats.issues_by_severity["high"] == 0

    def test_dimension_averages(self, sample_result):
        """Test dimension average calculation."""
        stats = ReportStatistics([sample_result])

        assert "pattern_compliance" in stats.dimension_averages
        assert stats.dimension_averages["pattern_compliance"] == 5.0


class TestMarkdownWriter:
    """Tests for MarkdownWriter."""

    def test_write_results(self, sample_result):
        """Test writing results to Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = MarkdownWriter(output_path)

            report_path = writer.write_results([sample_result])

            assert report_path.exists()
            assert report_path.name == "report.md"

            content = report_path.read_text()
            assert "Content Evaluation Report" in content
            assert "Executive Summary" in content

    def test_write_batch_result(self, sample_batch_result):
        """Test writing batch result to Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = MarkdownWriter(output_path)

            report_path = writer.write(sample_batch_result)

            assert report_path.exists()

            content = report_path.read_text()
            assert "test-run-001" in content or "full" in content.lower()

    def test_custom_filename(self):
        """Test using custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = MarkdownWriter(output_path, filename="custom_report.md")

            assert writer.filepath.name == "custom_report.md"

    def test_threshold_in_report(self, sample_result):
        """Test that threshold is reflected in report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = MarkdownWriter(output_path)

            report_path = writer.write_results([sample_result], threshold=5.0)

            content = report_path.read_text()
            assert "5.0" in content or "5" in content
