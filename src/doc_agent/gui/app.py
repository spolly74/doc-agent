"""Streamlit GUI for Doc Agent evaluation system.

Run with: streamlit run src/doc_agent/gui/app.py
Or after installing: doc-agent-gui
"""

import asyncio
import os
from pathlib import Path

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Doc Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from doc_agent import __version__
from doc_agent.config import load_config
from doc_agent.models.enums import Dimension
from doc_agent.orchestration.runner import EvaluationRunner


def load_dotenv() -> None:
    """Load .env file if present."""
    env_file = Path.cwd() / ".env"
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


# Load environment
load_dotenv()


def init_session_state():
    """Initialize session state variables."""
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    if "target_browse_path" not in st.session_state:
        st.session_state.target_browse_path = str(Path.home())
    if "output_browse_path" not in st.session_state:
        st.session_state.output_browse_path = str(Path.home())


def render_directory_browser(key: str, label: str) -> str | None:
    """Render a directory browser and return the selected path.

    Args:
        key: Unique key for session state (e.g., 'target' or 'output')
        label: Label to display in the expander

    Returns:
        Selected directory path, or None if nothing selected
    """
    browse_key = f"{key}_browse_path"
    expanded_key = f"{key}_browser_expanded"
    current_path = Path(st.session_state.get(browse_key, str(Path.home())))

    # Track if browser should be expanded
    is_expanded = st.session_state.get(expanded_key, False)

    with st.expander(f"📁 Browse for {label}", expanded=is_expanded):
        # Show current location
        st.caption(f"Current: {current_path}")

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🏠 Home", key=f"{key}_home"):
                st.session_state[browse_key] = str(Path.home())
                st.session_state[expanded_key] = True  # Keep open
                st.rerun()
        with col2:
            if st.button("⬆️ Up", key=f"{key}_up"):
                parent = current_path.parent
                if parent != current_path:
                    st.session_state[browse_key] = str(parent)
                    st.session_state[expanded_key] = True  # Keep open
                    st.rerun()
        with col3:
            if st.button(f"✅ Select This Folder", key=f"{key}_select", type="primary"):
                st.session_state[expanded_key] = False  # Close after selection
                return str(current_path)

        # List directories
        try:
            dirs = sorted([
                d for d in current_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ], key=lambda x: x.name.lower())

            if not dirs:
                st.info("No subdirectories found")
            else:
                # Show as clickable buttons in a scrollable container
                for d in dirs[:20]:  # Limit to 20 to avoid overwhelming
                    if st.button(f"📂 {d.name}", key=f"{key}_{d.name}", use_container_width=True):
                        st.session_state[browse_key] = str(d)
                        st.session_state[expanded_key] = True  # Keep open
                        st.rerun()

                if len(dirs) > 20:
                    st.caption(f"... and {len(dirs) - 20} more folders")

        except PermissionError:
            st.error("Permission denied to access this directory")
        except Exception as e:
            st.error(f"Error listing directory: {e}")

    return None


def render_sidebar():
    """Render the sidebar configuration."""
    with st.sidebar:
        st.title("📄 Doc Agent")
        st.caption(f"v{__version__}")
        st.divider()

        # Target directory
        st.subheader("Target")

        # Initialize target path in session state if not set
        if "target_path" not in st.session_state:
            st.session_state.target_path = "./docs"

        target_path = st.text_input(
            "Documentation Directory",
            value=st.session_state.target_path,
            help="Path to the documentation directory to evaluate",
            key="target_input",
        )
        st.session_state.target_path = target_path

        # Directory browser for target
        selected_target = render_directory_browser("target", "Target Directory")
        if selected_target:
            st.session_state.target_path = selected_target
            st.rerun()

        # Output directory
        if "output_path" not in st.session_state:
            st.session_state.output_path = "./eval-output"

        output_path = st.text_input(
            "Output Directory",
            value=st.session_state.output_path,
            help="Path to save evaluation results",
            key="output_input",
        )
        st.session_state.output_path = output_path

        # Directory browser for output
        selected_output = render_directory_browser("output", "Output Directory")
        if selected_output:
            st.session_state.output_path = selected_output
            st.rerun()

        st.divider()

        # LLM Configuration
        st.subheader("LLM Configuration")

        provider = st.selectbox(
            "Provider",
            options=["claude", "ollama"],
            index=0,
            help="LLM provider to use for evaluation",
        )

        if provider == "claude":
            default_model = "claude-sonnet-4-20250514"
            model_help = "Claude model to use"
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                st.warning("ANTHROPIC_API_KEY not set in environment")
        else:
            default_model = "llama3.2"
            model_help = "Ollama model (must be pulled locally)"

        model = st.text_input("Model", value=default_model, help=model_help)

        concurrency = st.slider(
            "Concurrency",
            min_value=1,
            max_value=10,
            value=2,
            help="Number of concurrent LLM requests",
        )

        st.divider()

        # Evaluation mode
        st.subheader("Evaluation Mode")
        mode = st.radio(
            "Mode",
            options=["Full", "Tiered"],
            index=0,
            help="Full: evaluate all articles. Tiered: quick scan then deep eval where needed.",
        )

        threshold = 4.0
        if mode == "Tiered":
            threshold = st.slider(
                "Threshold",
                min_value=1.0,
                max_value=7.0,
                value=4.0,
                step=0.5,
                help="Articles below this score get full evaluation",
            )

        st.divider()

        # Run button
        run_disabled = st.session_state.is_running
        if st.button(
            "🚀 Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        ):
            return {
                "run": True,
                "target": st.session_state.target_path,
                "output": st.session_state.output_path,
                "provider": provider,
                "model": model,
                "concurrency": concurrency,
                "mode": mode.lower(),
                "threshold": threshold,
            }

        if st.session_state.is_running:
            st.info(f"⏳ {st.session_state.status_message}")

        return None


def render_results():
    """Render evaluation results."""
    results = st.session_state.evaluation_results

    if results is None:
        st.info("👈 Configure options in the sidebar and click 'Run Evaluation' to start.")

        # Show dimension info
        st.subheader("Evaluation Dimensions")

        dimensions = [
            ("pattern_compliance", "Follows correct article pattern (how-to, quickstart, etc.)"),
            ("dev_focus", "Minimizes time-to-first-success for developers"),
            ("code_quality", "Well-written, production-ready code samples"),
            ("code_completeness", "All necessary code samples present"),
            ("structure", "Well-organized, scannable content"),
            ("accuracy", "Technically correct and current"),
            ("branding_compliance", "Correct Microsoft terminology"),
            ("jtbd_alignment", "Supports stated customer intent"),
        ]

        cols = st.columns(2)
        for i, (name, desc) in enumerate(dimensions):
            with cols[i % 2]:
                st.markdown(f"**{name}**: {desc}")

        return

    # Summary metrics
    st.subheader("Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Articles", results.total_articles)
    with col2:
        st.metric("Evaluated", results.evaluated_articles)
    with col3:
        st.metric("Failed", results.failed_articles)
    with col4:
        st.metric("Average Score", f"{results.average_score:.2f}")
    with col5:
        st.metric("Below Threshold", len(results.articles_below_threshold))

    if results.duration_seconds:
        st.caption(f"⏱️ Completed in {results.duration_seconds / 60:.1f} minutes")

    st.divider()

    # Results table
    st.subheader("Article Scores")

    if not results.results:
        st.warning("No results to display")
        return

    # Build dataframe
    import pandas as pd

    rows = []
    for r in results.results:
        row = {
            "Article": r.article_path,
            "Title": r.article_title[:50] + "..." if len(r.article_title) > 50 else r.article_title,
            "Overall": round(r.overall_score, 1),
            "Issues": len(r.issues),
        }

        # Add dimension scores
        for dim in Dimension:
            score = r.get_dimension_score(dim)
            row[dim.value.replace("_", " ").title()] = score if score else "-"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by overall score (lowest first)
    df = df.sort_values("Overall", ascending=True)

    # Color code the overall score
    def color_score(val):
        if isinstance(val, (int, float)):
            if val < 3:
                return "background-color: #ffcccc"
            elif val < 5:
                return "background-color: #ffffcc"
            else:
                return "background-color: #ccffcc"
        return ""

    styled_df = df.style.applymap(color_score, subset=["Overall"])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
    )

    st.divider()

    # Score distribution chart
    st.subheader("Score Distribution")

    try:
        import plotly.express as px

        scores = [r.overall_score for r in results.results]
        fig = px.histogram(
            x=scores,
            nbins=14,
            labels={"x": "Score", "y": "Count"},
            title="Distribution of Overall Scores",
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for score distribution chart: pip install plotly")

    st.divider()

    # Issues breakdown
    st.subheader("Issues by Severity")

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in results.results:
        for issue in r.issues:
            sev = issue.severity.value.lower()
            if sev in severity_counts:
                severity_counts[sev] += 1

    cols = st.columns(4)
    with cols[0]:
        st.metric("🔴 Critical", severity_counts["critical"])
    with cols[1]:
        st.metric("🟠 High", severity_counts["high"])
    with cols[2]:
        st.metric("🟡 Medium", severity_counts["medium"])
    with cols[3]:
        st.metric("🟢 Low", severity_counts["low"])

    st.divider()

    # Detailed view
    st.subheader("Article Details")

    article_options = [f"{r.article_path} (Score: {r.overall_score:.1f})" for r in results.results]
    selected = st.selectbox("Select an article to view details:", article_options)

    if selected:
        idx = article_options.index(selected)
        result = results.results[idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Dimension Scores:**")
            for dim in Dimension:
                score = result.get_dimension_score(dim)
                if score:
                    bar_length = int(score * 10)
                    bar = "█" * bar_length + "░" * (70 - bar_length)
                    st.text(f"{dim.value:20} {score:.1f}/7  {bar}")
                else:
                    st.text(f"{dim.value:20} N/A")

        with col2:
            st.markdown("**Issues:**")
            if result.issues:
                for issue in result.issues[:10]:
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢",
                    }.get(issue.severity.value.lower(), "⚪")
                    st.markdown(f"{severity_emoji} **{issue.title}**")
                    st.caption(issue.description[:200])
                if len(result.issues) > 10:
                    st.caption(f"... and {len(result.issues) - 10} more issues")
            else:
                st.success("No issues found!")


async def run_evaluation(config_params: dict):
    """Run the evaluation asynchronously."""
    st.session_state.is_running = True
    st.session_state.status_message = "Loading configuration..."

    try:
        # Load config
        cfg = load_config(
            target=Path(config_params["target"]),
            output=Path(config_params["output"]),
            provider=config_params["provider"],
            model=config_params["model"],
            concurrency=config_params["concurrency"],
            threshold=config_params["threshold"],
        )

        st.session_state.status_message = "Initializing evaluator..."

        # Create runner
        runner = EvaluationRunner(
            config=cfg,
            output_path=Path(config_params["output"]),
        )

        st.session_state.status_message = "Running evaluation..."

        # Run evaluation
        if config_params["mode"] == "tiered":
            batch_result = await runner.run_tiered(
                target_path=Path(config_params["target"]),
                threshold=config_params["threshold"],
            )
        else:
            batch_result = await runner.run_full(
                target_path=Path(config_params["target"]),
            )

        st.session_state.evaluation_results = batch_result
        st.session_state.status_message = "Complete!"

    except Exception as e:
        st.session_state.status_message = f"Error: {e}"
        raise

    finally:
        st.session_state.is_running = False


def main():
    """Main Streamlit app entry point."""
    init_session_state()

    # Sidebar
    run_config = render_sidebar()

    # Main content
    st.title("Doc Agent Evaluation")

    # Handle run
    if run_config and run_config.get("run"):
        # Validate inputs
        target = Path(run_config["target"])
        if not target.exists():
            st.error(f"Target directory does not exist: {target}")
        else:
            with st.spinner("Running evaluation..."):
                try:
                    asyncio.run(run_evaluation(run_config))
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    # Results
    render_results()


if __name__ == "__main__":
    main()
