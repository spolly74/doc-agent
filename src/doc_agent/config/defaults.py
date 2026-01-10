"""Default configuration values for the Doc Agent evaluation system."""

from pathlib import Path

# Default configuration file name
DEFAULT_CONFIG_FILENAME = "doc-agent.config.json"

# Default output directory name
DEFAULT_OUTPUT_DIR = "eval-output"

# Default state database filename
DEFAULT_STATE_DB = ".doc-agent-state.db"

# Search paths for configuration file (in order of priority)
CONFIG_SEARCH_PATHS = [
    Path.cwd() / DEFAULT_CONFIG_FILENAME,
    Path.home() / ".config" / "doc-agent" / "config.json",
]

# Default LLM model
DEFAULT_LLM_MODEL = "claude-sonnet-4-20250514"

# Default scoring threshold (articles below this need attention)
DEFAULT_THRESHOLD = 4.0

# Default concurrent LLM requests (reduced to avoid rate limits)
DEFAULT_CONCURRENCY = 2

# Dimensions that can be null (when not applicable)
NULLABLE_DIMENSIONS = ["code_quality", "jtbd_alignment"]
