"""Default configuration values for the Doc Agent evaluation system."""

from pathlib import Path

# Default configuration file name
DEFAULT_CONFIG_FILENAME = "foundry-eval.config.json"

# Default output directory name
DEFAULT_OUTPUT_DIR = "eval-output"

# Default state database filename
DEFAULT_STATE_DB = ".foundry-eval-state.db"

# Search paths for configuration file (in order of priority)
CONFIG_SEARCH_PATHS = [
    Path.cwd() / DEFAULT_CONFIG_FILENAME,
    Path.home() / ".config" / "foundry-eval" / "config.json",
]

# Default LLM model
DEFAULT_LLM_MODEL = "claude-sonnet-4-20250514"

# Default scoring threshold (articles below this need attention)
DEFAULT_THRESHOLD = 4.0

# Default concurrent LLM requests
DEFAULT_CONCURRENCY = 5

# Dimensions that can be null (when not applicable)
NULLABLE_DIMENSIONS = ["code_quality", "jtbd_alignment"]
