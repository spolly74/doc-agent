"""Markdown report writer for evaluation results.

Generates report.md per spec section 8.2.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jinja2 import Environment, PackageLoader, select_autoescape

from foundry_eval.models.enums import Dimension, RunMode, Severity
from foundry_eval.models.evaluation import BatchEvaluationResult, EvaluationResult

logger = logging.getLogger("foundry_eval.output.markdown")


class ReportStatistics:
    """Statistics for the evaluation report."""

    def __init__(self, results: list[EvaluationResult], threshold: float = 4.0):
        """Calculate statistics from results.

        Args:
            results: List of evaluation results.
            threshold: Score threshold for flagging articles.
        """
        self.total_articles = len(results)
        self.threshold = threshold

        if not results:
            self.average_score = 0.0
            self.min_score = 0.0
            self.max_score = 0.0
            self.below_threshold = 0
            self.dimension_averages = {}
            self.issues_by_severity = {}
            self.score_distribution = {}
            return

        # Calculate overall scores
        scores = [r.overall_score for r in results if r.overall_score > 0]
        self.average_score = sum(scores) / len(scores) if scores else 0
        self.min_score = min(scores) if scores else 0
        self.max_score = max(scores) if scores else 0
        self.below_threshold = sum(1 for s in scores if s < threshold)

        # Calculate dimension averages
        self.dimension_averages = {}
        for dim in Dimension:
            dim_scores = [
                r.get_dimension_score(dim)
                for r in results
                if r.get_dimension_score(dim) is not None
            ]
            if dim_scores:
                self.dimension_averages[dim.value] = sum(dim_scores) / len(dim_scores)

        # Count issues by severity
        self.issues_by_severity = {s.value: 0 for s in Severity}
        for result in results:
            for issue in result.issues:
                self.issues_by_severity[issue.severity.value] += 1

        # Score distribution
        self.score_distribution = {
            "1-2": sum(1 for s in scores if 1 <= s < 2),
            "2-3": sum(1 for s in scores if 2 <= s < 3),
            "3-4": sum(1 for s in scores if 3 <= s < 4),
            "4-5": sum(1 for s in scores if 4 <= s < 5),
            "5-6": sum(1 for s in scores if 5 <= s < 6),
            "6-7": sum(1 for s in scores if 6 <= s <= 7),
        }


class MarkdownWriter:
    """Writes evaluation results to Markdown report format."""

    def __init__(
        self,
        output_path: Path,
        filename: str = "report.md",
    ):
        """Initialize the Markdown writer.

        Args:
            output_path: Directory to write the report.
            filename: Name of the report file.
        """
        self._output_path = output_path
        self._filename = filename

        # Set up Jinja2 environment
        try:
            self._env = Environment(
                loader=PackageLoader("foundry_eval.output", "templates"),
                autoescape=select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        except Exception:
            # Fallback if templates not found
            self._env = None

    @property
    def filepath(self) -> Path:
        """Full path to the report file."""
        return self._output_path / self._filename

    def write(
        self,
        batch_result: BatchEvaluationResult,
        threshold: float = 4.0,
    ) -> Path:
        """Write batch results to Markdown report.

        Args:
            batch_result: Batch evaluation result to write.
            threshold: Score threshold for highlighting.

        Returns:
            Path to the written report file.
        """
        return self.write_results(
            results=batch_result.results,
            run_mode=batch_result.run_mode,
            target_path=batch_result.target_path,
            threshold=threshold,
            started_at=batch_result.started_at,
            completed_at=batch_result.completed_at,
        )

    def write_results(
        self,
        results: list[EvaluationResult],
        run_mode: str = "full",
        target_path: str = "",
        threshold: float = 4.0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Path:
        """Write evaluation results to Markdown report.

        Args:
            results: List of evaluation results.
            run_mode: Evaluation run mode.
            target_path: Target documentation path.
            threshold: Score threshold.
            started_at: Run start time.
            completed_at: Run completion time.

        Returns:
            Path to the written report file.
        """
        # Ensure output directory exists
        self._output_path.mkdir(parents=True, exist_ok=True)

        # Calculate statistics
        stats = ReportStatistics(results, threshold)

        # Generate report content
        if self._env:
            content = self._render_template(
                results=results,
                stats=stats,
                run_mode=run_mode,
                target_path=target_path,
                threshold=threshold,
                started_at=started_at,
                completed_at=completed_at,
            )
        else:
            content = self._generate_report(
                results=results,
                stats=stats,
                run_mode=run_mode,
                target_path=target_path,
                threshold=threshold,
                started_at=started_at,
                completed_at=completed_at,
            )

        # Write to file
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Wrote report to {self.filepath}")
        return self.filepath

    def _render_template(
        self,
        results: list[EvaluationResult],
        stats: ReportStatistics,
        run_mode: str,
        target_path: str,
        threshold: float,
        started_at: Optional[datetime],
        completed_at: Optional[datetime],
    ) -> str:
        """Render report using Jinja2 template.

        Args:
            results: Evaluation results.
            stats: Calculated statistics.
            run_mode: Run mode.
            target_path: Target path.
            threshold: Score threshold.
            started_at: Start time.
            completed_at: Completion time.

        Returns:
            Rendered Markdown content.
        """
        template = self._env.get_template("report.md.j2")

        # Get top articles needing attention
        top_articles = sorted(results, key=lambda r: r.overall_score)[:10]

        # Get common issues
        common_issues = self._get_common_issues(results)

        return template.render(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            run_mode=run_mode,
            target_path=target_path,
            threshold=threshold,
            stats=stats,
            top_articles=top_articles,
            common_issues=common_issues,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _generate_report(
        self,
        results: list[EvaluationResult],
        stats: ReportStatistics,
        run_mode: str,
        target_path: str,
        threshold: float,
        started_at: Optional[datetime],
        completed_at: Optional[datetime],
    ) -> str:
        """Generate report without template (fallback).

        Args:
            results: Evaluation results.
            stats: Calculated statistics.
            run_mode: Run mode.
            target_path: Target path.
            threshold: Score threshold.
            started_at: Start time.
            completed_at: Completion time.

        Returns:
            Markdown content.
        """
        lines = []

        # Header
        lines.append("# Content Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"**Scope:** {target_path}")
        lines.append(f"**Mode:** {run_mode.title()} Evaluation")
        lines.append(f"**Articles evaluated:** {stats.total_articles}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Average overall score:** {stats.average_score:.1f} / 7")
        lines.append(f"- **Articles below threshold (< {threshold}):** {stats.below_threshold} ({stats.below_threshold * 100 // max(stats.total_articles, 1)}%)")
        lines.append(f"- **Critical issues:** {stats.issues_by_severity.get('critical', 0)}")
        lines.append(f"- **High issues:** {stats.issues_by_severity.get('high', 0)}")
        lines.append("")

        # Top 10 Articles Requiring Attention
        lines.append("## Top 10 Articles Requiring Attention")
        lines.append("")
        lines.append("| Rank | Article | Score | Top Issue |")
        lines.append("|------|---------|-------|-----------|")

        top_articles = sorted(results, key=lambda r: r.overall_score)[:10]
        for i, result in enumerate(top_articles, 1):
            path = result.article_path
            if len(path) > 40:
                path = "..." + path[-37:]
            top_issue = result.issues[0].title if result.issues else "None"
            if len(top_issue) > 30:
                top_issue = top_issue[:27] + "..."
            lines.append(f"| {i} | {path} | {result.overall_score:.1f} | {top_issue} |")

        lines.append("")

        # Score Distribution by Dimension
        lines.append("## Score Distribution by Dimension")
        lines.append("")
        lines.append("| Dimension | Avg | Min | Max | Below Threshold |")
        lines.append("|-----------|-----|-----|-----|-----------------|")

        for dim in Dimension:
            if dim.value in stats.dimension_averages:
                avg = stats.dimension_averages[dim.value]
                dim_scores = [
                    r.get_dimension_score(dim)
                    for r in results
                    if r.get_dimension_score(dim) is not None
                ]
                min_score = min(dim_scores) if dim_scores else 0
                max_score = max(dim_scores) if dim_scores else 0
                below = sum(1 for s in dim_scores if s < threshold)
                lines.append(f"| {dim.value} | {avg:.1f} | {min_score:.0f} | {max_score:.0f} | {below} |")

        lines.append("")

        # Common Issues
        lines.append("## Common Issues (Systemic)")
        lines.append("")

        common_issues = self._get_common_issues(results)
        for i, (issue_title, count, articles) in enumerate(common_issues[:5], 1):
            lines.append(f"{i}. **{issue_title}** ({count} articles)")
            lines.append(f"   - Affected: {', '.join(articles[:3])}" + (" ..." if len(articles) > 3 else ""))
            lines.append("")

        # Next Steps
        lines.append("## Next Steps")
        lines.append("")
        lines.append(f"1. Address top {min(10, stats.below_threshold)} articles below threshold")
        lines.append(f"2. Fix {stats.issues_by_severity.get('critical', 0)} critical issues")
        lines.append(f"3. Review {stats.issues_by_severity.get('high', 0)} high-priority issues")
        lines.append("")

        return "\n".join(lines)

    def _get_common_issues(
        self,
        results: list[EvaluationResult],
        min_count: int = 2,
    ) -> list[tuple[str, int, list[str]]]:
        """Find common issues across articles.

        Args:
            results: Evaluation results.
            min_count: Minimum occurrence count to include.

        Returns:
            List of (issue_title, count, affected_articles) tuples.
        """
        issue_counts: dict[str, list[str]] = {}

        for result in results:
            for issue in result.issues:
                key = issue.title.lower().strip()
                if key not in issue_counts:
                    issue_counts[key] = []
                issue_counts[key].append(result.article_path)

        # Filter and sort
        common = [
            (title, len(articles), articles)
            for title, articles in issue_counts.items()
            if len(articles) >= min_count
        ]
        common.sort(key=lambda x: x[1], reverse=True)

        return common
