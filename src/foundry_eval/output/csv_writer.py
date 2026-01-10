"""CSV output writer for evaluation scores.

Generates scores.csv per spec section 8.1.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from foundry_eval.models.enums import Dimension
from foundry_eval.models.evaluation import BatchEvaluationResult, EvaluationResult

logger = logging.getLogger("foundry_eval.output.csv")

# CSV column definitions per spec 8.1
CSV_COLUMNS = [
    "article_path",
    "title",
    "ms_author",
    "ms_subservice",
    "ms_topic",
    "ms_date",
    "pattern_compliance",
    "dev_focus",
    "code_quality",
    "code_completeness",
    "structure",
    "accuracy",
    "jtbd_alignment",
    "overall_score",
    "priority_rank",
    "top_issues",
    "issue_count",
    "has_code_samples",
    "sample_quality_issues",
    "coverage_gaps",
]


class CSVWriter:
    """Writes evaluation results to CSV format."""

    def __init__(self, output_path: Path, filename: str = "scores.csv"):
        """Initialize the CSV writer.

        Args:
            output_path: Directory to write the CSV file.
            filename: Name of the CSV file.
        """
        self._output_path = output_path
        self._filename = filename

    @property
    def filepath(self) -> Path:
        """Full path to the CSV file."""
        return self._output_path / self._filename

    def write(self, batch_result: BatchEvaluationResult) -> Path:
        """Write batch results to CSV.

        Args:
            batch_result: Batch evaluation result to write.

        Returns:
            Path to the written CSV file.
        """
        return self.write_results(batch_result.results)

    def write_results(
        self,
        results: list[EvaluationResult],
        article_metadata: Optional[dict] = None,
    ) -> Path:
        """Write evaluation results to CSV.

        Args:
            results: List of evaluation results.
            article_metadata: Optional dict mapping article_path to metadata.

        Returns:
            Path to the written CSV file.
        """
        # Ensure output directory exists
        self._output_path.mkdir(parents=True, exist_ok=True)

        # Sort by priority (highest priority first)
        sorted_results = sorted(results, key=lambda r: r.priority_rank, reverse=True)

        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            for result in sorted_results:
                row = self._result_to_row(result, article_metadata)
                writer.writerow(row)

        logger.info(f"Wrote {len(results)} results to {self.filepath}")
        return self.filepath

    def _result_to_row(
        self,
        result: EvaluationResult,
        article_metadata: Optional[dict] = None,
    ) -> dict:
        """Convert an evaluation result to a CSV row.

        Args:
            result: Evaluation result to convert.
            article_metadata: Optional metadata for the article.

        Returns:
            Dictionary representing a CSV row.
        """
        # Get metadata if available
        metadata = {}
        if article_metadata and result.article_path in article_metadata:
            metadata = article_metadata[result.article_path]

        # Extract dimension scores
        scores = {}
        for dim in Dimension:
            score = result.get_dimension_score(dim)
            scores[dim.value] = score if score is not None else ""

        # Build row
        row = {
            "article_path": result.article_path,
            "title": result.article_title,
            "ms_author": metadata.get("ms_author", ""),
            "ms_subservice": metadata.get("ms_subservice", ""),
            "ms_topic": metadata.get("ms_topic", ""),
            "ms_date": metadata.get("ms_date", ""),
            "pattern_compliance": scores.get("pattern_compliance", ""),
            "dev_focus": scores.get("dev_focus", ""),
            "code_quality": scores.get("code_quality", ""),
            "code_completeness": scores.get("code_completeness", ""),
            "structure": scores.get("structure", ""),
            "accuracy": scores.get("accuracy", ""),
            "jtbd_alignment": scores.get("jtbd_alignment", ""),
            "overall_score": round(result.overall_score, 2),
            "priority_rank": result.priority_rank,
            "top_issues": result.top_issues_summary,
            "issue_count": len(result.issues),
            "has_code_samples": metadata.get("has_code", False),
            "sample_quality_issues": "",  # Populated in samples mode
            "coverage_gaps": "",  # Populated in JTBD mode
        }

        return row

    def append_result(self, result: EvaluationResult) -> None:
        """Append a single result to an existing CSV.

        Args:
            result: Evaluation result to append.
        """
        file_exists = self.filepath.exists()

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

            if not file_exists:
                writer.writeheader()

            row = self._result_to_row(result)
            writer.writerow(row)


def results_to_csv_string(results: list[EvaluationResult]) -> str:
    """Convert results to CSV string (for testing/debugging).

    Args:
        results: List of evaluation results.

    Returns:
        CSV content as string.
    """
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    csv_writer = CSVWriter(Path("."), "temp.csv")
    for result in results:
        row = csv_writer._result_to_row(result)
        writer.writerow(row)

    return output.getvalue()
