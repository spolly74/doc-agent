"""File system utilities for the Foundry evaluation system."""

import fnmatch
from pathlib import Path
from typing import Optional

import aiofiles


def find_files(
    root: Path,
    pattern: str,
    exclude_patterns: Optional[list[str]] = None,
) -> list[Path]:
    """Find files matching a glob pattern.

    Args:
        root: Root directory to search in.
        pattern: Glob pattern (e.g., "**/*.md").
        exclude_patterns: Patterns for files to exclude.

    Returns:
        List of matching file paths.
    """
    exclude_patterns = exclude_patterns or []
    matches = []

    for path in root.rglob(pattern.lstrip("**/").lstrip("/")):
        if not path.is_file():
            continue

        # Check exclusion patterns
        relative = str(path.relative_to(root))
        excluded = False

        for exclude in exclude_patterns:
            if fnmatch.fnmatch(relative, exclude):
                excluded = True
                break

        if not excluded:
            matches.append(path)

    return matches


def read_file(path: Path, encoding: str = "utf-8") -> str:
    """Read a file synchronously.

    Args:
        path: Path to the file.
        encoding: File encoding.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IOError: If the file can't be read.
    """
    return path.read_text(encoding=encoding)


async def read_file_async(path: Path, encoding: str = "utf-8") -> str:
    """Read a file asynchronously.

    Args:
        path: Path to the file.
        encoding: File encoding.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IOError: If the file can't be read.
    """
    async with aiofiles.open(path, "r", encoding=encoding) as f:
        return await f.read()


async def write_file_async(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    mkdir: bool = True,
) -> None:
    """Write content to a file asynchronously.

    Args:
        path: Path to the file.
        content: Content to write.
        encoding: File encoding.
        mkdir: Whether to create parent directories.
    """
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(path, "w", encoding=encoding) as f:
        await f.write(content)


def resolve_path(
    path: str | Path,
    base: Optional[Path] = None,
) -> Path:
    """Resolve a path, optionally relative to a base.

    Args:
        path: Path to resolve.
        base: Base directory for relative paths.

    Returns:
        Resolved absolute path.
    """
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    if base is not None:
        return (base / path).resolve()

    return path.resolve()


def get_relative_path(
    path: Path,
    root: Path,
) -> str:
    """Get the relative path from root to path.

    Args:
        path: Target path.
        root: Root directory.

    Returns:
        Relative path as string.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        # Path is not relative to root, return absolute
        return str(path)


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        The directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_markdown_file(path: Path) -> bool:
    """Check if a path is a markdown file.

    Args:
        path: File path to check.

    Returns:
        True if the file is a markdown file.
    """
    return path.suffix.lower() in (".md", ".markdown", ".mdown", ".mkd")


def is_yaml_file(path: Path) -> bool:
    """Check if a path is a YAML file.

    Args:
        path: File path to check.

    Returns:
        True if the file is a YAML file.
    """
    return path.suffix.lower() in (".yml", ".yaml")


def get_file_age_days(path: Path) -> Optional[int]:
    """Get the age of a file in days based on modification time.

    Args:
        path: File path.

    Returns:
        Age in days, or None if path doesn't exist.
    """
    if not path.exists():
        return None

    import time
    mtime = path.stat().st_mtime
    now = time.time()
    age_seconds = now - mtime
    return int(age_seconds / 86400)  # 86400 seconds in a day
