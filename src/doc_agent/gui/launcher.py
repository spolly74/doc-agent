"""Launcher for the Streamlit GUI.

This module provides an entry point for running the Streamlit GUI
via the `doc-agent-gui` command.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit GUI."""
    # Get the path to the app.py file
    app_path = Path(__file__).parent / "app.py"

    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless=true"]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit not found. Install with: pip install doc-agent[gui]")
        sys.exit(1)


if __name__ == "__main__":
    main()
