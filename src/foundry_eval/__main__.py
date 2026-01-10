"""Entry point for running foundry-eval as a module.

Usage:
    python -m foundry_eval [command] [options]
"""

from foundry_eval.cli.main import app

if __name__ == "__main__":
    app()
