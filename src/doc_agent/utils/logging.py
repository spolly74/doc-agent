"""Logging configuration for the Foundry evaluation system."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    verbosity: int = 1,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Set up logging with Rich formatting.

    Args:
        verbosity: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG, 3=DEBUG+libs).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    # Map verbosity to log levels
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,
    }
    level = level_map.get(verbosity, logging.INFO)

    # Create logger
    logger = logging.getLogger("doc_agent")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with Rich
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=verbosity >= 2,
        show_path=verbosity >= 2,
        rich_tracebacks=True,
        tracebacks_show_locals=verbosity >= 3,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)

    # Quiet noisy libraries unless verbosity is 3
    if verbosity < 3:
        for lib_logger in ["httpx", "anthropic", "aiosqlite"]:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)

    return logger


def get_logger(name: str = "doc_agent") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (will be prefixed with "doc_agent.").

    Returns:
        Logger instance.
    """
    if not name.startswith("doc_agent"):
        name = f"doc_agent.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level changes."""

    def __init__(self, logger: logging.Logger, level: int):
        """Initialize the context.

        Args:
            logger: Logger to modify.
            level: Temporary log level.
        """
        self.logger = logger
        self.level = level
        self.original_level = logger.level

    def __enter__(self) -> logging.Logger:
        """Enter the context."""
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        self.logger.setLevel(self.original_level)
