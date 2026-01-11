"""Streamlit GUI for Doc Agent evaluation system.

Run with: streamlit run src/doc_agent/gui/app.py
Or after installing: doc-agent-gui
"""

import os
import subprocess
import sys
import time
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
from doc_agent.jobs.manager import JobManager
from doc_agent.jobs.models import JobConfig, JobStatus
from doc_agent.jobs.worker import is_worker_running, LOG_FILE


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

# Initialize job manager (singleton-ish via module level)
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get or create the job manager singleton."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def ensure_worker_running() -> bool:
    """Ensure the background worker is running.

    Returns:
        True if worker is running, False if failed to start.
    """
    if is_worker_running():
        return True

    try:
        # Start worker as a detached subprocess
        worker_script = Path(__file__).parent.parent / "jobs" / "worker.py"

        if sys.platform == "win32":
            # Windows: use CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(
                [sys.executable, str(worker_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            # Unix: use nohup-style detachment
            subprocess.Popen(
                [sys.executable, str(worker_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait briefly for worker to start
        time.sleep(0.5)
        return is_worker_running()

    except Exception as e:
        st.warning(f"Could not start background worker: {e}")
        return False


def init_session_state():
    """Initialize session state variables."""
    if "current_job_id" not in st.session_state:
        st.session_state.current_job_id = None
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
            if st.button("✅ Select This Folder", key=f"{key}_select", type="primary"):
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

        # Worker status indicator
        if is_worker_running():
            st.caption("🟢 Worker running")
        else:
            st.caption("🔴 Worker stopped")

        # Show log file location
        with st.expander("📋 Worker Logs"):
            st.caption(f"Log file: `{LOG_FILE}`")
            if LOG_FILE.exists():
                # Show last 20 lines of log
                try:
                    lines = LOG_FILE.read_text().strip().split("\n")
                    recent_lines = lines[-20:] if len(lines) > 20 else lines
                    st.code("\n".join(recent_lines), language=None)
                except Exception as e:
                    st.error(f"Could not read log: {e}")
            else:
                st.info("No log file yet. Start an evaluation to create logs.")

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

        # Check if there's an active job
        has_active_job = st.session_state.current_job_id is not None
        if has_active_job:
            job = get_job_manager().get_job(st.session_state.current_job_id)
            has_active_job = job is not None and not job.is_terminal

        # Run button
        if st.button(
            "🚀 Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=has_active_job,
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

        if has_active_job:
            st.info("⏳ Evaluation in progress...")

            # Cancel button
            if st.button("🛑 Cancel Job", use_container_width=True):
                get_job_manager().cancel_job(st.session_state.current_job_id)
                st.session_state.current_job_id = None
                st.rerun()

        return None


def render_job_progress(job):
    """Render job progress display.

    Args:
        job: The Job object to display progress for.
    """
    st.subheader("Evaluation Progress")

    # Status indicator
    status_colors = {
        JobStatus.PENDING: "🟡",
        JobStatus.RUNNING: "🔵",
        JobStatus.COMPLETE: "🟢",
        JobStatus.FAILED: "🔴",
        JobStatus.CANCELLED: "⚪",
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", f"{status_colors.get(job.status, '⚪')} {job.status.value.title()}")
    with col2:
        st.metric("Job ID", job.job_id)
    with col3:
        if job.duration_seconds:
            mins = job.duration_seconds / 60
            st.metric("Duration", f"{mins:.1f} min")
        else:
            st.metric("Duration", "-")

    # Progress bar
    if job.progress_total > 0:
        progress = job.progress_current / job.progress_total
        st.progress(progress, text=f"Processing: {job.progress_current}/{job.progress_total} articles")
    elif job.status == JobStatus.PENDING:
        st.progress(0, text="Waiting to start...")
    elif job.status == JobStatus.RUNNING:
        st.progress(0, text="Initializing...")

    # Error message
    if job.error_message:
        st.error(f"Error: {job.error_message}")

    # Auto-refresh while job is active
    if not job.is_terminal:
        time.sleep(2)  # Brief pause before refresh
        st.rerun()


def render_job_results(job):
    """Render results from a completed job.

    Args:
        job: The completed Job object.
    """
    if job.result_summary is None:
        st.warning("No results available")
        return

    results = job.result_summary

    # Summary metrics
    st.subheader("Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Articles", results.get("total_articles", 0))
    with col2:
        st.metric("Evaluated", results.get("evaluated_articles", 0))
    with col3:
        st.metric("Failed", results.get("failed_articles", 0))
    with col4:
        avg = results.get("average_score", 0)
        st.metric("Average Score", f"{avg:.2f}" if avg else "-")
    with col5:
        st.metric("Below Threshold", results.get("below_threshold_count", 0))

    if results.get("duration_seconds"):
        st.caption(f"⏱️ Completed in {results['duration_seconds'] / 60:.1f} minutes")

    # Link to output files
    output_path = results.get("output_path")
    if output_path:
        st.info(f"📁 Results saved to: {output_path}")

    st.divider()

    # Option to view detailed results
    st.subheader("View Detailed Results")
    st.markdown("""
    The detailed evaluation results (CSV and Markdown report) have been saved to your output directory.

    To view them:
    - Open the CSV file in Excel or a spreadsheet application
    - Open the Markdown report in a text editor or Markdown viewer
    """)


def render_results_placeholder():
    """Render the placeholder when no job is active."""
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


def render_job_history():
    """Render recent job history."""
    manager = get_job_manager()
    jobs = manager.list_jobs(limit=10)

    if not jobs:
        return

    st.divider()
    st.subheader("Recent Jobs")

    for job in jobs:
        status_emoji = {
            JobStatus.PENDING: "🟡",
            JobStatus.RUNNING: "🔵",
            JobStatus.COMPLETE: "🟢",
            JobStatus.FAILED: "🔴",
            JobStatus.CANCELLED: "⚪",
        }.get(job.status, "⚪")

        # Check if this is the currently selected job
        is_current = st.session_state.current_job_id == job.job_id

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            # Highlight current job
            if is_current:
                st.markdown(f"**{status_emoji} {job.job_id}** ◀")
            else:
                st.text(f"{status_emoji} {job.job_id}")
        with col2:
            st.text(job.status.value)
        with col3:
            st.text(job.created_at.strftime("%H:%M:%S"))
        with col4:
            # Show button for all jobs (different label based on status)
            if not is_current:
                if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    button_label = "Monitor"
                elif job.status == JobStatus.COMPLETE:
                    button_label = "View"
                else:
                    button_label = "Details"

                if st.button(button_label, key=f"view_{job.job_id}"):
                    st.session_state.current_job_id = job.job_id
                    st.rerun()


def count_markdown_files(target_path: Path) -> int:
    """Count markdown files in the target directory."""
    count = 0
    for f in target_path.rglob("*.md"):
        # Skip common non-article files
        if f.name.lower() not in ("readme.md", "changelog.md", "contributing.md"):
            count += 1
    return count


def main():
    """Main Streamlit app entry point."""
    init_session_state()

    # Sidebar
    run_config = render_sidebar()

    # Main content
    st.title("Doc Agent Evaluation")

    # Handle new job request
    if run_config and run_config.get("run"):
        # Validate inputs
        target = Path(run_config["target"])
        if not target.exists():
            st.error(f"Target directory does not exist: {target}")
        else:
            # Count articles first
            article_count = count_markdown_files(target)
            if article_count == 0:
                st.error(f"No markdown files found in: {target}")
            else:
                # Ensure worker is running
                if not ensure_worker_running():
                    st.error("Could not start background worker. Please try again or run from CLI.")
                else:
                    # Create job
                    job_config = JobConfig(
                        target_path=run_config["target"],
                        output_path=run_config["output"],
                        provider=run_config["provider"],
                        model=run_config["model"],
                        concurrency=run_config["concurrency"],
                        mode=run_config["mode"],
                        threshold=run_config["threshold"],
                    )
                    job = get_job_manager().create_job(job_config)
                    st.session_state.current_job_id = job.job_id
                    st.success(f"Created job {job.job_id} - evaluating {article_count} articles")
                    st.rerun()

    # Display current job or results
    if st.session_state.current_job_id:
        job = get_job_manager().get_job(st.session_state.current_job_id)

        if job is None:
            st.warning("Job not found")
            st.session_state.current_job_id = None
        elif job.is_terminal:
            if job.status == JobStatus.COMPLETE:
                render_job_results(job)
            elif job.status == JobStatus.FAILED:
                st.error(f"Job failed: {job.error_message}")
            elif job.status == JobStatus.CANCELLED:
                st.warning("Job was cancelled")

            # Button to start new evaluation
            if st.button("🔄 Start New Evaluation"):
                st.session_state.current_job_id = None
                st.rerun()
        else:
            render_job_progress(job)
    else:
        render_results_placeholder()

    # Show job history
    render_job_history()


if __name__ == "__main__":
    main()
