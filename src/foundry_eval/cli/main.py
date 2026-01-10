"""Main CLI entry point for foundry-eval."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from foundry_eval import __version__
from foundry_eval.config import load_config

# Create the main Typer app
app = typer.Typer(
    name="foundry-eval",
    help="Agentic content evaluation system for Microsoft Foundry documentation.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Console for rich output
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"foundry-eval version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Foundry Content Evaluation System.

    Evaluate Microsoft Foundry documentation against a developer-focused rubric.
    """
    pass


# Common options used across commands
ConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--config",
        "-c",
        help="Path to configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]

TargetOption = Annotated[
    Path,
    typer.Option(
        "--target",
        "-t",
        help="Target documentation directory to evaluate.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
]

OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        help="Output directory for results (CSV and report).",
    ),
]

VerboseOption = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        help="Verbosity level (0=quiet, 1=normal, 2=verbose, 3=debug).",
        min=0,
        max=3,
        count=True,
    ),
]


@app.command()
def full(
    target: TargetOption,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results.",
        ),
    ] = Path("./eval-output"),
    config: ConfigOption = None,
    verbose: VerboseOption = 1,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use for evaluation.",
        ),
    ] = None,
    concurrency: Annotated[
        Optional[int],
        typer.Option(
            "--concurrency",
            help="Maximum concurrent LLM requests.",
            min=1,
            max=50,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force re-evaluation, ignoring cached results.",
        ),
    ] = False,
) -> None:
    """Run full evaluation on all articles in target directory.

    Evaluates every article against the complete 7-dimension rubric,
    checks code sample quality, and produces comprehensive reports.

    Example:
        foundry-eval full --target ./docs --output ./results
    """
    cfg = load_config(
        config_path=config,
        target=target,
        output=output,
        verbose=verbose,
        model=model,
        concurrency=concurrency,
    )

    console.print(f"[bold green]Starting full evaluation[/bold green]")
    console.print(f"  Target: {target}")
    console.print(f"  Output: {output}")
    console.print(f"  Model: {cfg.llm.model}")

    # TODO: Implement full evaluation
    console.print("[yellow]Full evaluation not yet implemented[/yellow]")


@app.command()
def tiered(
    target: TargetOption,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results.",
        ),
    ] = Path("./eval-output"),
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Score threshold for deep evaluation (articles below this get full eval).",
            min=1.0,
            max=7.0,
        ),
    ] = 4.0,
    config: ConfigOption = None,
    verbose: VerboseOption = 1,
) -> None:
    """Run tiered evaluation (quick scan + deep evaluation where needed).

    First performs a quick structural scan of all articles, then runs
    full evaluation only on articles that score below the threshold
    or have critical issues.

    Example:
        foundry-eval tiered --target ./docs --threshold 3.5
    """
    cfg = load_config(
        config_path=config,
        target=target,
        output=output,
        threshold=threshold,
        verbose=verbose,
    )

    console.print(f"[bold green]Starting tiered evaluation[/bold green]")
    console.print(f"  Target: {target}")
    console.print(f"  Threshold: {threshold}")

    # TODO: Implement tiered evaluation
    console.print("[yellow]Tiered evaluation not yet implemented[/yellow]")


@app.command()
def jtbd(
    jtbd_file: Annotated[
        Path,
        typer.Option(
            "--jtbd-file",
            help="Path to JTBD data file (CSV or JSON).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    jtbd_id: Annotated[
        str,
        typer.Option(
            "--jtbd-id",
            help="ID of the specific JTBD to analyze.",
        ),
    ],
    target: Annotated[
        Optional[Path],
        typer.Option(
            "--target",
            "-t",
            help="Target documentation directory (defaults to config or current dir).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results.",
        ),
    ] = Path("./eval-output"),
    config: ConfigOption = None,
    verbose: VerboseOption = 1,
) -> None:
    """Run JTBD-scoped evaluation for a specific job-to-be-done.

    Loads the specified JTBD, maps existing articles to its steps,
    evaluates coverage, and identifies gaps.

    Example:
        foundry-eval jtbd --jtbd-file ./jtbd.csv --jtbd-id jtbd-001
    """
    cfg = load_config(
        config_path=config,
        target=target,
        output=output,
        jtbd_file=jtbd_file,
        verbose=verbose,
    )

    console.print(f"[bold green]Starting JTBD-scoped evaluation[/bold green]")
    console.print(f"  JTBD File: {jtbd_file}")
    console.print(f"  JTBD ID: {jtbd_id}")

    # TODO: Implement JTBD evaluation
    console.print("[yellow]JTBD evaluation not yet implemented[/yellow]")


@app.command()
def samples(
    articles_repo: Annotated[
        Path,
        typer.Option(
            "--articles-repo",
            help="Path to the documentation repository.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    samples_repo: Annotated[
        Path,
        typer.Option(
            "--samples-repo",
            help="Path to the code samples repository.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results.",
        ),
    ] = Path("./eval-output"),
    config: ConfigOption = None,
    verbose: VerboseOption = 1,
) -> None:
    """Validate code samples and check references.

    Indexes the samples repository, validates sample quality,
    checks for orphaned samples, and identifies articles that
    need samples but don't have them.

    Example:
        foundry-eval samples --articles-repo ./docs --samples-repo ./samples
    """
    cfg = load_config(
        config_path=config,
        target=articles_repo,
        samples_repo=samples_repo,
        output=output,
        verbose=verbose,
    )

    console.print(f"[bold green]Starting code sample validation[/bold green]")
    console.print(f"  Articles: {articles_repo}")
    console.print(f"  Samples: {samples_repo}")

    # TODO: Implement samples validation
    console.print("[yellow]Samples validation not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
