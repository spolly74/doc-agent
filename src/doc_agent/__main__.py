"""Entry point for running foundry-eval as a module.

Usage:
    python -m doc_agent [command] [options]
"""

from doc_agent.cli.main import app

if __name__ == "__main__":
    app()
