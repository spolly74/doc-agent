"""Parse YAML Table of Contents (TOC) files for documentation structure."""

import asyncio
from pathlib import Path
from typing import Optional

import aiofiles
import yaml

from foundry_eval.models.article import Article
from foundry_eval.context.metadata_extractor import MetadataExtractor


class TOCEntry:
    """Represents an entry in the table of contents."""

    def __init__(
        self,
        name: str,
        href: Optional[str] = None,
        items: Optional[list["TOCEntry"]] = None,
        expanded: bool = False,
        display_name: Optional[str] = None,
    ):
        self.name = name
        self.href = href
        self.items = items or []
        self.expanded = expanded
        self.display_name = display_name or name

    @property
    def is_section(self) -> bool:
        """Whether this entry is a section (has children, no href)."""
        return self.href is None and len(self.items) > 0

    @property
    def is_article(self) -> bool:
        """Whether this entry is an article (has href)."""
        return self.href is not None

    def all_articles(self) -> list[tuple[str, str]]:
        """Get all article hrefs with their section paths.

        Returns:
            List of (href, section_path) tuples.
        """
        articles = []
        if self.is_article:
            articles.append((self.href, self.name))
        for item in self.items:
            for href, path in item.all_articles():
                articles.append((href, f"{self.name} > {path}"))
        return articles

    def __repr__(self) -> str:
        if self.is_section:
            return f"TOCEntry(section='{self.name}', items={len(self.items)})"
        return f"TOCEntry(article='{self.name}', href='{self.href}')"


class TOCParser:
    """Parses YAML TOC files and provides article discovery."""

    # Common TOC file names
    TOC_FILENAMES = ["toc.yml", "toc.yaml", "TOC.yml", "TOC.yaml"]

    def __init__(self, docs_root: Path):
        """Initialize the TOC parser.

        Args:
            docs_root: Root directory of the documentation.
        """
        self.docs_root = docs_root
        self._toc_cache: dict[Path, list[TOCEntry]] = {}
        self._extractor = MetadataExtractor()

    async def find_toc_files(self) -> list[Path]:
        """Find all TOC files in the documentation root.

        Returns:
            List of paths to TOC files.
        """
        toc_files = []

        # Check root directory
        for filename in self.TOC_FILENAMES:
            toc_path = self.docs_root / filename
            if toc_path.exists():
                toc_files.append(toc_path)

        # Check subdirectories (one level deep)
        if self.docs_root.is_dir():
            for subdir in self.docs_root.iterdir():
                if subdir.is_dir():
                    for filename in self.TOC_FILENAMES:
                        toc_path = subdir / filename
                        if toc_path.exists():
                            toc_files.append(toc_path)

        return toc_files

    async def parse_toc(self, toc_path: Path) -> list[TOCEntry]:
        """Parse a TOC file into TOCEntry objects.

        Args:
            toc_path: Path to the TOC file.

        Returns:
            List of top-level TOCEntry objects.
        """
        if toc_path in self._toc_cache:
            return self._toc_cache[toc_path]

        async with aiofiles.open(toc_path, "r", encoding="utf-8") as f:
            content = await f.read()

        data = yaml.safe_load(content)
        if data is None:
            return []

        # Handle different TOC formats
        if isinstance(data, list):
            entries = [self._parse_entry(item) for item in data]
        elif isinstance(data, dict) and "items" in data:
            entries = [self._parse_entry(item) for item in data["items"]]
        else:
            entries = []

        self._toc_cache[toc_path] = entries
        return entries

    async def get_all_article_paths(self) -> list[Path]:
        """Get all article paths from all TOC files.

        Returns:
            List of resolved article paths.
        """
        toc_files = await self.find_toc_files()
        all_paths = []

        for toc_file in toc_files:
            entries = await self.parse_toc(toc_file)
            toc_dir = toc_file.parent

            for entry in entries:
                for href, _ in entry.all_articles():
                    # Resolve the path relative to TOC file location
                    article_path = self._resolve_href(href, toc_dir)
                    if article_path and article_path.exists():
                        all_paths.append(article_path)

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in all_paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(path)

        return unique_paths

    async def get_articles_with_toc_info(
        self,
    ) -> list[tuple[Path, str, int]]:
        """Get articles with their TOC section and order.

        Returns:
            List of (path, section, order) tuples.
        """
        toc_files = await self.find_toc_files()
        articles = []
        order = 0

        for toc_file in toc_files:
            entries = await self.parse_toc(toc_file)
            toc_dir = toc_file.parent

            for entry in entries:
                for href, section_path in entry.all_articles():
                    article_path = self._resolve_href(href, toc_dir)
                    if article_path and article_path.exists():
                        articles.append((article_path, section_path, order))
                        order += 1

        return articles

    async def load_all_articles(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> list[Article]:
        """Load all articles from TOC or by pattern matching.

        If TOC files exist, uses them. Otherwise falls back to glob patterns.

        Args:
            include_patterns: Glob patterns for files to include.
            exclude_patterns: Glob patterns for files to exclude.

        Returns:
            List of Article objects.
        """
        # Try TOC-based discovery first
        toc_files = await self.find_toc_files()

        if toc_files:
            return await self._load_articles_from_toc()
        else:
            return await self._load_articles_from_patterns(
                include_patterns or ["**/*.md"],
                exclude_patterns or [],
            )

    async def _load_articles_from_toc(self) -> list[Article]:
        """Load articles using TOC structure.

        Returns:
            List of Article objects with TOC metadata.
        """
        articles_info = await self.get_articles_with_toc_info()
        articles = []

        for path, section, order in articles_info:
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()

                article = self._extractor.create_article(
                    path=path,
                    docs_root=self.docs_root,
                    content=content,
                )
                article.toc_section = section
                article.toc_order = order
                articles.append(article)
            except Exception as e:
                # Log error but continue with other articles
                print(f"Warning: Failed to load {path}: {e}")

        return articles

    async def _load_articles_from_patterns(
        self,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> list[Article]:
        """Load articles using glob patterns.

        Args:
            include_patterns: Patterns for files to include.
            exclude_patterns: Patterns for files to exclude.

        Returns:
            List of Article objects.
        """
        from foundry_eval.utils.file_utils import find_files

        all_paths = set()
        for pattern in include_patterns:
            paths = find_files(self.docs_root, pattern)
            all_paths.update(paths)

        # Remove excluded files
        for pattern in exclude_patterns:
            excluded = find_files(self.docs_root, pattern)
            all_paths -= set(excluded)

        # Sort for consistent ordering
        sorted_paths = sorted(all_paths)

        # Load articles concurrently
        articles = []
        for i, path in enumerate(sorted_paths):
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()

                article = self._extractor.create_article(
                    path=path,
                    docs_root=self.docs_root,
                    content=content,
                )
                article.toc_order = i
                articles.append(article)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        return articles

    def _parse_entry(self, data: dict) -> TOCEntry:
        """Parse a single TOC entry from YAML data.

        Args:
            data: Dictionary representing a TOC entry.

        Returns:
            TOCEntry object.
        """
        name = data.get("name", "Unnamed")
        href = data.get("href")
        expanded = data.get("expanded", False)
        display_name = data.get("displayName")

        # Parse child items recursively
        items = []
        if "items" in data:
            items = [self._parse_entry(item) for item in data["items"]]

        return TOCEntry(
            name=name,
            href=href,
            items=items,
            expanded=expanded,
            display_name=display_name,
        )

    def _resolve_href(self, href: str, toc_dir: Path) -> Optional[Path]:
        """Resolve an href to an absolute path.

        Args:
            href: The href from the TOC (may be relative path or URL).
            toc_dir: Directory containing the TOC file.

        Returns:
            Resolved Path or None if href is external/invalid.
        """
        # Skip external URLs
        if href.startswith(("http://", "https://", "mailto:")):
            return None

        # Skip anchors without file
        if href.startswith("#"):
            return None

        # Remove any anchor from the href
        href = href.split("#")[0]
        if not href:
            return None

        # Resolve relative path
        resolved = toc_dir / href
        return resolved.resolve()
