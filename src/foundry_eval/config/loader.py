"""Configuration loading and merging logic."""

import json
import os
from pathlib import Path
from typing import Any, Optional

from foundry_eval.config.defaults import CONFIG_SEARCH_PATHS, DEFAULT_CONFIG_FILENAME
from foundry_eval.config.models import FoundryEvalConfig


def find_config_file(explicit_path: Optional[Path] = None) -> Optional[Path]:
    """Find the configuration file.

    Args:
        explicit_path: Explicitly specified config file path (from CLI).

    Returns:
        Path to config file if found, None otherwise.
    """
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Config file not found: {explicit_path}")

    # Search default locations
    for search_path in CONFIG_SEARCH_PATHS:
        if search_path.exists():
            return search_path

    return None


def load_config_file(path: Path) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return json.load(f)


def merge_cli_overrides(
    config: FoundryEvalConfig,
    target: Optional[Path] = None,
    output: Optional[Path] = None,
    samples_repo: Optional[Path] = None,
    jtbd_file: Optional[Path] = None,
    threshold: Optional[float] = None,
    verbose: Optional[int] = None,
    model: Optional[str] = None,
    concurrency: Optional[int] = None,
) -> FoundryEvalConfig:
    """Merge CLI overrides into the configuration.

    CLI arguments take precedence over config file values.

    Args:
        config: Base configuration from file.
        target: Target documentation path.
        output: Output directory path.
        samples_repo: Samples repository path.
        jtbd_file: JTBD data file path.
        threshold: Scoring threshold override.
        verbose: Verbosity level override.
        model: LLM model override.
        concurrency: Max concurrent requests override.

    Returns:
        Configuration with CLI overrides applied.
    """
    # Create a copy to avoid mutating the original
    data = config.model_dump()

    # Apply path overrides
    if target is not None:
        data["articles_path"] = target
    if output is not None:
        data["output_path"] = output
    if samples_repo is not None:
        data["samples_repo_path"] = samples_repo
    if jtbd_file is not None:
        data["jtbd_data_path"] = jtbd_file

    # Apply evaluation overrides
    if threshold is not None:
        data["evaluation"]["threshold"] = threshold

    # Apply output overrides
    if verbose is not None:
        data["output"]["verbosity"] = verbose

    # Apply LLM overrides
    if model is not None:
        data["llm"]["model"] = model
    if concurrency is not None:
        data["llm"]["max_concurrent_requests"] = concurrency

    return FoundryEvalConfig.model_validate(data)


def load_config(
    config_path: Optional[Path] = None,
    **cli_overrides: Any,
) -> FoundryEvalConfig:
    """Load configuration with CLI overrides.

    Configuration is loaded from the following sources (in order of priority):
    1. CLI arguments (highest priority)
    2. Config file (if found)
    3. Environment variables
    4. Default values (lowest priority)

    Args:
        config_path: Explicit config file path (from --config CLI option).
        **cli_overrides: CLI argument overrides.

    Returns:
        Merged configuration object.
    """
    # Start with default configuration
    config = FoundryEvalConfig()

    # Try to load from config file
    found_config = find_config_file(config_path)
    if found_config is not None:
        file_data = load_config_file(found_config)
        config = FoundryEvalConfig.model_validate(file_data)

    # Check for environment variable overrides
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        # API key is handled separately by the LLM provider
        pass

    if model := os.environ.get("FOUNDRY_EVAL_MODEL"):
        cli_overrides.setdefault("model", model)

    # Apply CLI overrides
    config = merge_cli_overrides(config, **cli_overrides)

    return config
