"""Progress tracking with Rich console output."""

import logging
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logger = logging.getLogger("doc_agent.orchestration.progress")


class ProgressTracker:
    """Tracks and displays evaluation progress using Rich."""

    def __init__(
        self,
        console: Optional[Console] = None,
        show_progress: bool = True,
    ):
        """Initialize the progress tracker.

        Args:
            console: Rich console for output (created if not provided).
            show_progress: Whether to show progress bar.
        """
        self._console = console or Console()
        self._show_progress = show_progress
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._total = 0
        self._completed = 0
        self._errors = 0

    def set_total(self, total: int) -> None:
        """Set the total number of items to process.

        Args:
            total: Total item count.
        """
        self._total = total
        self._completed = 0
        self._errors = 0

        if self._show_progress:
            self._create_progress_bar()

    def _create_progress_bar(self) -> None:
        """Create and start the progress bar."""
        if self._progress is not None:
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            "Evaluating articles",
            total=self._total,
        )

    def update(
        self,
        completed: int,
        current_item: str = "",
        is_error: bool = False,
    ) -> None:
        """Update progress.

        Args:
            completed: Number of completed items.
            current_item: Currently processing item.
            is_error: Whether this update represents an error.
        """
        self._completed = completed
        if is_error:
            self._errors += 1

        if self._progress and self._task_id is not None:
            # Truncate long paths
            display_item = current_item
            if len(display_item) > 50:
                display_item = "..." + display_item[-47:]

            description = f"Evaluating: {display_item}" if current_item else "Evaluating articles"
            self._progress.update(
                self._task_id,
                completed=completed,
                description=description,
            )

    def advance(self, amount: int = 1) -> None:
        """Advance progress by an amount.

        Args:
            amount: Amount to advance.
        """
        self._completed += amount
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, amount)

    def finish(self) -> None:
        """Finish progress tracking and display summary."""
        if self._progress:
            self._progress.stop()
            self._progress = None

        # Display summary
        self._console.print()
        self._console.print(
            f"[bold green]✓ Completed:[/] {self._completed} articles"
        )
        if self._errors > 0:
            self._console.print(
                f"[bold red]✗ Errors:[/] {self._errors} articles"
            )

    def display_summary_table(
        self,
        results: list,
        threshold: float = 4.0,
    ) -> None:
        """Display a summary table of results.

        Args:
            results: List of EvaluationResult objects.
            threshold: Score threshold for highlighting.
        """
        table = Table(title="Evaluation Summary")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        # Calculate statistics
        total = len(results)
        if total == 0:
            self._console.print("[yellow]No results to display[/]")
            return

        scores = [r.overall_score for r in results if r.overall_score > 0]
        avg_score = sum(scores) / len(scores) if scores else 0

        below_threshold = sum(1 for s in scores if s < threshold)
        critical_issues = sum(len(r.critical_issues) for r in results)
        high_issues = sum(len(r.high_issues) for r in results)

        table.add_row("Total Articles", str(total))
        table.add_row("Average Score", f"{avg_score:.2f}")
        table.add_row(
            f"Below Threshold (<{threshold})",
            f"[red]{below_threshold}[/]" if below_threshold > 0 else "0",
        )
        table.add_row(
            "Critical Issues",
            f"[red]{critical_issues}[/]" if critical_issues > 0 else "0",
        )
        table.add_row(
            "High Issues",
            f"[yellow]{high_issues}[/]" if high_issues > 0 else "0",
        )

        self._console.print(table)

    def display_top_issues(
        self,
        results: list,
        limit: int = 10,
    ) -> None:
        """Display top articles needing attention.

        Args:
            results: List of EvaluationResult objects.
            limit: Maximum number of articles to show.
        """
        # Sort by score (ascending) to show worst first
        sorted_results = sorted(
            results,
            key=lambda r: (r.overall_score, -len(r.critical_issues)),
        )

        table = Table(title=f"Top {limit} Articles Needing Attention")

        table.add_column("Article", style="cyan", max_width=40)
        table.add_column("Score", justify="center")
        table.add_column("Issues", justify="center")
        table.add_column("Top Issue", max_width=30)

        for result in sorted_results[:limit]:
            # Determine score color
            score = result.overall_score
            if score < 3:
                score_str = f"[red]{score:.1f}[/]"
            elif score < 4:
                score_str = f"[yellow]{score:.1f}[/]"
            else:
                score_str = f"[green]{score:.1f}[/]"

            # Get top issue
            top_issue = ""
            if result.issues:
                top_issue = result.issues[0].title[:30]

            # Truncate path
            path = result.article_path
            if len(path) > 40:
                path = "..." + path[-37:]

            table.add_row(
                path,
                score_str,
                str(len(result.issues)),
                top_issue,
            )

        self._console.print(table)

    def print_status(self, message: str, style: str = "") -> None:
        """Print a status message.

        Args:
            message: Message to print.
            style: Rich style string.
        """
        if style:
            self._console.print(f"[{style}]{message}[/]")
        else:
            self._console.print(message)

    def print_error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: Error message.
        """
        self._console.print(f"[bold red]Error:[/] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: Warning message.
        """
        self._console.print(f"[bold yellow]Warning:[/] {message}")

    def print_success(self, message: str) -> None:
        """Print a success message.

        Args:
            message: Success message.
        """
        self._console.print(f"[bold green]✓[/] {message}")
