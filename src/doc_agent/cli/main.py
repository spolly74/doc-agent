"""Main CLI entry point for doc-agent."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from doc_agent import __version__
from doc_agent.config import load_config
from doc_agent.orchestration.runner import EvaluationRunner
from doc_agent.utils.logging import setup_logging


def _load_dotenv() -> None:
    """Load .env file if it exists."""
    # Look for .env in current directory and parent directories
    current = Path.cwd()
    for path in [current, *current.parents]:
        env_file = path / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            break


# Load .env file before anything else
_load_dotenv()

# Create the main Typer app
app = typer.Typer(
    name="doc-agent",
    help="""Agentic content evaluation system for technical documentation.

Evaluates articles against an 8-dimension rubric:
  [cyan]pattern_compliance[/]  - Follows correct article pattern (how-to, quickstart, etc.)
  [cyan]dev_focus[/]           - Minimizes time-to-first-success for developers
  [cyan]code_quality[/]        - Well-written, production-ready code samples
  [cyan]code_completeness[/]   - All necessary code samples present
  [cyan]structure[/]           - Well-organized, scannable content
  [cyan]accuracy[/]            - Technically correct and current
  [cyan]branding_compliance[/] - Correct Microsoft terminology
  [cyan]jtbd_alignment[/]      - Supports stated customer intent

[bold]Examples:[/bold]

  [dim]# Full evaluation with Claude (default)[/dim]
  doc-agent full --target ./docs --output ./results

  [dim]# Use local Ollama for evaluation[/dim]
  doc-agent full --target ./docs --provider ollama --model llama3.2

  [dim]# Tiered evaluation (quick scan, then deep eval where needed)[/dim]
  doc-agent tiered --target ./docs --threshold 4.0

  [dim]# JTBD coverage analysis[/dim]
  doc-agent jtbd --jtbd-file ./jtbd.csv --target ./docs

  [dim]# Code sample validation[/dim]
  doc-agent samples --articles-repo ./docs --samples-repo ./samples

[bold]Configuration:[/bold]

  Create [cyan]doc-agent.config.json[/cyan] in your project root, or use CLI flags.
  Set [cyan]ANTHROPIC_API_KEY[/cyan] env var for Claude, or use --provider ollama for local LLM.
""",
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
    """Evaluate technical documentation against a developer-focused rubric."""
    pass


def _print_summary(batch_result) -> None:
    """Print a summary table of evaluation results."""
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Articles", str(batch_result.total_articles))
    table.add_row("Evaluated", str(batch_result.evaluated_articles))
    table.add_row("Failed", str(batch_result.failed_articles))
    table.add_row("Average Score", f"{batch_result.average_score:.2f}")
    table.add_row("Below Threshold", str(len(batch_result.articles_below_threshold)))

    if batch_result.duration_seconds:
        minutes = batch_result.duration_seconds / 60
        table.add_row("Duration", f"{minutes:.1f} minutes")

    if batch_result.total_input_tokens > 0:
        table.add_row("Input Tokens", f"{batch_result.total_input_tokens:,}")
        table.add_row("Output Tokens", f"{batch_result.total_output_tokens:,}")

    console.print(table)


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
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-p",
            help="LLM provider to use (claude, ollama).",
        ),
    ] = None,
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

    Evaluates every article against the complete 8-dimension rubric
    and produces scores.csv + report.md.

    [bold]Examples:[/bold]

      [dim]# Basic evaluation with Claude[/dim]
      doc-agent full --target ./docs

      [dim]# Specify output directory[/dim]
      doc-agent full --target ./docs --output ./my-results

      [dim]# Use Ollama for local evaluation[/dim]
      doc-agent full --target ./docs --provider ollama --model llama3.2

      [dim]# Force re-evaluation (ignore cache)[/dim]
      doc-agent full --target ./docs --force

      [dim]# Increase concurrency for faster evaluation[/dim]
      doc-agent full --target ./docs --concurrency 5
    """
    cfg = load_config(
        config_path=config,
        target=target,
        output=output,
        verbose=verbose,
        model=model,
        concurrency=concurrency,
        provider=provider,
    )

    # Set up logging based on verbosity
    setup_logging(verbosity=verbose)

    console.print(f"[bold green]Starting full evaluation[/bold green]")
    console.print(f"  Target: {target}")
    console.print(f"  Output: {output}")
    console.print(f"  Provider: {cfg.llm.provider.value}")
    console.print(f"  Model: {cfg.llm.model}")
    console.print()

    try:
        runner = EvaluationRunner(config=cfg, output_path=output)
        batch_result = asyncio.run(runner.run_full(target_path=target, force=force))

        # Print summary
        _print_summary(batch_result)

        # Print output paths
        paths = runner.get_output_paths()
        console.print()
        console.print("[bold]Output files:[/bold]")
        console.print(f"  CSV:    {paths['csv']}")
        console.print(f"  Report: {paths['report']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


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
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-p",
            help="LLM provider to use (claude, ollama).",
        ),
    ] = None,
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
    """Run tiered evaluation (quick scan + deep evaluation where needed).

    First performs a quick structural scan of all articles, then runs
    full evaluation only on articles that score below the threshold
    or have critical issues. More efficient for large doc sets.

    [bold]Examples:[/bold]

      [dim]# Default threshold (4.0)[/dim]
      doc-agent tiered --target ./docs

      [dim]# Stricter threshold (more articles get deep eval)[/dim]
      doc-agent tiered --target ./docs --threshold 5.0

      [dim]# Lenient threshold (fewer deep evaluations)[/dim]
      doc-agent tiered --target ./docs --threshold 3.0

      [dim]# Use Ollama with tiered evaluation[/dim]
      doc-agent tiered --target ./docs --provider ollama --model qwen2.5
    """
    cfg = load_config(
        config_path=config,
        target=target,
        output=output,
        threshold=threshold,
        verbose=verbose,
        provider=provider,
        model=model,
        concurrency=concurrency,
    )

    # Set up logging based on verbosity
    setup_logging(verbosity=verbose)

    console.print(f"[bold green]Starting tiered evaluation[/bold green]")
    console.print(f"  Target: {target}")
    console.print(f"  Threshold: {threshold}")
    console.print(f"  Output: {output}")
    console.print(f"  Provider: {cfg.llm.provider.value}")
    console.print(f"  Model: {cfg.llm.model}")
    console.print()

    try:
        runner = EvaluationRunner(config=cfg, output_path=output)
        batch_result = asyncio.run(
            runner.run_tiered(target_path=target, threshold=threshold, force=force)
        )

        # Print summary
        _print_summary(batch_result)

        # Print output paths
        paths = runner.get_output_paths()
        console.print()
        console.print("[bold]Output files:[/bold]")
        console.print(f"  CSV:    {paths['csv']}")
        console.print(f"  Report: {paths['report']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


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
        Optional[str],
        typer.Option(
            "--jtbd-id",
            help="ID of the specific JTBD to analyze (analyzes all if not specified).",
        ),
    ] = None,
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

    Loads the JTBD data, maps existing articles to job steps,
    evaluates coverage, and identifies content gaps.

    [bold]Examples:[/bold]

      [dim]# Analyze all JTBDs in a file[/dim]
      doc-agent jtbd --jtbd-file ./jtbd.csv --target ./docs

      [dim]# Analyze a specific JTBD[/dim]
      doc-agent jtbd --jtbd-file ./jtbd.json --jtbd-id deploy-model

      [dim]# JSON format JTBD file[/dim]
      doc-agent jtbd --jtbd-file ./jobs-to-be-done.json --target ./docs

    [bold]JTBD File Formats:[/bold]

      CSV: id,title,description columns (Golden Path format)
      JSON: Array of {id, title, steps: [{id, description}]} objects
    """
    from doc_agent.context.jtbd_loader import JTBDLoader
    from doc_agent.context.toc_parser import TOCParser
    from doc_agent.evaluators.jtbd import JTBDAnalyzer
    from doc_agent.llm.factory import create_llm_provider

    cfg = load_config(
        config_path=config,
        target=target,
        output=output,
        jtbd_file=jtbd_file,
        verbose=verbose,
    )

    setup_logging(verbosity=verbose)

    console.print(f"[bold green]Starting JTBD-scoped evaluation[/bold green]")
    console.print(f"  JTBD File: {jtbd_file}")
    if jtbd_id:
        console.print(f"  JTBD ID: {jtbd_id}")
    else:
        console.print("  Analyzing all JTBDs in file")
    console.print(f"  Target: {target or cfg.articles_path or Path('.')}")
    console.print()

    async def run_jtbd_analysis():
        # Create components
        llm_provider = create_llm_provider(cfg.llm)
        jtbd_loader = JTBDLoader(jtbd_file)
        target_path = target or cfg.articles_path or Path(".")
        toc_parser = TOCParser(target_path)

        analyzer = JTBDAnalyzer(llm_provider)

        if jtbd_id:
            # Analyze specific JTBD
            jtbd_obj = await jtbd_loader.load_by_id(jtbd_id)
            if not jtbd_obj:
                console.print(f"[red]JTBD not found: {jtbd_id}[/red]")
                return None

            def on_progress(msg):
                console.print(f"  {msg}")

            result = await analyzer.analyze_jtbd(jtbd_obj, toc_parser, on_progress)
            return [result]
        else:
            # Analyze all JTBDs
            def on_progress(completed, total, msg):
                console.print(f"  [{completed}/{total}] {msg}")

            results = await analyzer.analyze_all_jtbds(jtbd_loader, toc_parser, on_progress)
            return results

    try:
        results = asyncio.run(run_jtbd_analysis())

        if results:
            # Print summary
            console.print()
            table = Table(title="JTBD Coverage Summary")
            table.add_column("JTBD", style="cyan")
            table.add_column("Coverage", style="green")
            table.add_column("Gaps", style="yellow")
            table.add_column("Critical", style="red")

            for result in results:
                coverage = f"{result.overall_coverage_score * 100:.0f}%"
                gap_counts = result.gap_count_by_severity
                table.add_row(
                    result.jtbd.title[:40],
                    coverage,
                    str(len(result.gaps)),
                    str(gap_counts.get("critical", 0)),
                )

            console.print(table)

            # Save detailed results
            output.mkdir(parents=True, exist_ok=True)
            import json
            results_file = output / "jtbd-analysis.json"
            with open(results_file, "w") as f:
                json.dump([r.model_dump() for r in results], f, indent=2, default=str)
            console.print(f"\nDetailed results saved to: {results_file}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


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
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate code samples with LLM (slower but more thorough).",
        ),
    ] = False,
    config: ConfigOption = None,
    verbose: VerboseOption = 1,
) -> None:
    """Validate code samples and check references.

    Indexes the samples repository, validates sample quality,
    checks for orphaned samples, and identifies broken references.

    [bold]Examples:[/bold]

      [dim]# Basic sample validation (no LLM)[/dim]
      doc-agent samples --articles-repo ./docs --samples-repo ./samples

      [dim]# With LLM validation for code quality[/dim]
      doc-agent samples --articles-repo ./docs --samples-repo ./samples --validate

      [dim]# Verbose output to see details[/dim]
      doc-agent samples --articles-repo ./docs --samples-repo ./samples -v

    [bold]Output:[/bold]

      Produces samples-report.json with orphaned samples, broken references,
      and language breakdown.
    """
    from doc_agent.context.samples_index import SamplesIndex
    from doc_agent.context.toc_parser import TOCParser
    from doc_agent.evaluators.code_samples import CodeSampleValidator
    from doc_agent.llm.factory import create_llm_provider

    cfg = load_config(
        config_path=config,
        target=articles_repo,
        samples_repo=samples_repo,
        output=output,
        verbose=verbose,
    )

    setup_logging(verbosity=verbose)

    console.print(f"[bold green]Starting code sample validation[/bold green]")
    console.print(f"  Articles: {articles_repo}")
    console.print(f"  Samples: {samples_repo}")
    console.print(f"  LLM Validation: {'enabled' if validate else 'disabled'}")
    console.print()

    async def run_samples_analysis():
        # Create components
        samples_index = SamplesIndex(samples_repo, articles_repo)
        toc_parser = TOCParser(articles_repo)

        # Index samples
        console.print("Indexing samples repository...")
        sample_count = await samples_index.index()
        console.print(f"  Found {sample_count} code samples")

        # Get orphaned samples
        orphaned = await samples_index.get_orphaned_samples()
        console.print(f"  Orphaned samples: {len(orphaned)}")

        # Get broken references
        broken_refs = [r for r in samples_index.references if not r.is_valid]
        console.print(f"  Broken references: {len(broken_refs)}")

        # If validation requested, run LLM validation
        report = None
        if validate:
            console.print("\nValidating samples with LLM...")
            llm_provider = create_llm_provider(cfg.llm)
            validator = CodeSampleValidator(llm_provider)

            def on_progress(completed, total, msg):
                console.print(f"  [{completed}/{total}] {msg}")

            report = await validator.validate_samples(
                samples_index, toc_parser, on_progress
            )

        return samples_index, orphaned, broken_refs, report

    try:
        samples_index, orphaned, broken_refs, report = asyncio.run(run_samples_analysis())

        # Print summary
        console.print()
        table = Table(title="Sample Validation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Samples", str(len(samples_index.samples)))
        table.add_row("Orphaned Samples", str(len(orphaned)))
        table.add_row("Broken References", str(len(broken_refs)))

        if report:
            table.add_row("Valid Samples", str(report.valid_samples))
            table.add_row("Invalid Samples", str(report.invalid_samples))
            table.add_row("Validation Rate", f"{report.validation_pass_rate:.1f}%")

        console.print(table)

        # Show orphaned samples if any
        if orphaned and verbose >= 1:
            console.print("\n[yellow]Orphaned Samples (not referenced by any article):[/yellow]")
            for sample in orphaned[:10]:
                console.print(f"  - {sample.file_path}")
            if len(orphaned) > 10:
                console.print(f"  ... and {len(orphaned) - 10} more")

        # Show broken references if any
        if broken_refs and verbose >= 1:
            console.print("\n[red]Broken References:[/red]")
            for ref in broken_refs[:10]:
                console.print(f"  - {ref.article_path}: {ref.sample_id}")
            if len(broken_refs) > 10:
                console.print(f"  ... and {len(broken_refs) - 10} more")

        # Save detailed results
        output.mkdir(parents=True, exist_ok=True)
        import json

        results = {
            "total_samples": len(samples_index.samples),
            "orphaned_samples": [s.file_path for s in orphaned],
            "broken_references": [
                {"article": r.article_path, "sample": r.sample_id}
                for r in broken_refs
            ],
            "samples_by_language": {},
        }

        # Count by language
        for sample in samples_index.samples:
            lang = sample.language.value
            results["samples_by_language"][lang] = results["samples_by_language"].get(lang, 0) + 1

        if report:
            results["validation_report"] = {
                "valid_samples": report.valid_samples,
                "invalid_samples": report.invalid_samples,
                "pass_rate": report.validation_pass_rate,
            }

        results_file = output / "samples-report.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nDetailed results saved to: {results_file}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    app()
