"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_article_path(fixtures_dir: Path) -> Path:
    """Get the path to the sample article fixture."""
    return fixtures_dir / "sample_article.md"


@pytest.fixture
def sample_article_content(sample_article_path: Path) -> str:
    """Load the sample article content."""
    return sample_article_path.read_text(encoding="utf-8")


@pytest.fixture
def sample_toc_path(fixtures_dir: Path) -> Path:
    """Get the path to the sample TOC fixture."""
    return fixtures_dir / "sample_toc.yml"


@pytest.fixture
def sample_toc_content(sample_toc_path: Path) -> str:
    """Load the sample TOC content."""
    return sample_toc_path.read_text(encoding="utf-8")
