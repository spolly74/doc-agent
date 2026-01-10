"""Resolve [!INCLUDE] directives in documentation articles."""

import re
from pathlib import Path
from typing import Optional

import aiofiles


class IncludeResolver:
    """Resolves [!INCLUDE] directives to their content."""

    # Pattern for include directives: [!INCLUDE [name](path)]
    INCLUDE_PATTERN = re.compile(
        r"\[!INCLUDE\s*\[([^\]]*)\]\(([^)]+)\)\]",
        re.IGNORECASE
    )

    # Maximum recursion depth to prevent infinite loops
    MAX_DEPTH = 10

    def __init__(self, docs_root: Path):
        """Initialize the include resolver.

        Args:
            docs_root: Root directory of the documentation.
        """
        self.docs_root = docs_root
        self._cache: dict[Path, str] = {}

    async def resolve(
        self,
        content: str,
        source_path: Path,
        depth: int = 0,
    ) -> tuple[str, list[str]]:
        """Resolve all include directives in content.

        Args:
            content: Markdown content with include directives.
            source_path: Path of the source file (for relative path resolution).
            depth: Current recursion depth.

        Returns:
            Tuple of (resolved content, list of included file paths).
        """
        if depth >= self.MAX_DEPTH:
            return content, []

        included_files = []
        resolved_content = content

        # Find all includes
        matches = list(self.INCLUDE_PATTERN.finditer(content))

        # Process in reverse order to maintain correct positions
        for match in reversed(matches):
            include_name = match.group(1)
            include_path = match.group(2)

            # Resolve the include path
            resolved_path = self._resolve_path(include_path, source_path)

            if resolved_path is None:
                # Leave the include directive as-is if path can't be resolved
                continue

            # Load include content
            include_content = await self._load_include(resolved_path)

            if include_content is None:
                # Leave the include directive as-is if file not found
                continue

            # Track the included file
            included_files.append(str(resolved_path))

            # Recursively resolve includes in the included content
            nested_content, nested_files = await self.resolve(
                include_content,
                resolved_path,
                depth + 1,
            )
            included_files.extend(nested_files)

            # Replace the include directive with the content
            resolved_content = (
                resolved_content[:match.start()]
                + nested_content
                + resolved_content[match.end():]
            )

        return resolved_content, included_files

    async def get_include_paths(
        self,
        content: str,
        source_path: Path,
    ) -> list[str]:
        """Get all include file paths without resolving content.

        Args:
            content: Markdown content with include directives.
            source_path: Path of the source file.

        Returns:
            List of include file paths.
        """
        paths = []

        for match in self.INCLUDE_PATTERN.finditer(content):
            include_path = match.group(2)
            resolved_path = self._resolve_path(include_path, source_path)

            if resolved_path and resolved_path.exists():
                paths.append(str(resolved_path))

        return paths

    def find_includes_sync(self, content: str) -> list[tuple[str, str]]:
        """Find all include directives synchronously (no resolution).

        Args:
            content: Markdown content.

        Returns:
            List of (name, path) tuples.
        """
        includes = []
        for match in self.INCLUDE_PATTERN.finditer(content):
            name = match.group(1)
            path = match.group(2)
            includes.append((name, path))
        return includes

    async def _load_include(self, path: Path) -> Optional[str]:
        """Load content from an include file.

        Args:
            path: Path to the include file.

        Returns:
            File content or None if not found.
        """
        # Check cache first
        if path in self._cache:
            return self._cache[path]

        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()

            self._cache[path] = content
            return content
        except Exception:
            return None

    def _resolve_path(
        self,
        include_path: str,
        source_path: Path,
    ) -> Optional[Path]:
        """Resolve an include path to an absolute path.

        Args:
            include_path: The path from the include directive.
            source_path: Path of the source file.

        Returns:
            Resolved absolute path or None.
        """
        # Remove any tilde prefix (common in docs)
        include_path = include_path.lstrip("~")

        # Handle absolute paths (starting with /)
        if include_path.startswith("/"):
            resolved = self.docs_root / include_path.lstrip("/")
        else:
            # Relative path - resolve from source file's directory
            resolved = source_path.parent / include_path

        try:
            return resolved.resolve()
        except Exception:
            return None

    def clear_cache(self) -> None:
        """Clear the include file cache."""
        self._cache.clear()


class IncludeUsageAnalyzer:
    """Analyzes include file usage across documentation."""

    def __init__(self, resolver: IncludeResolver):
        """Initialize the analyzer.

        Args:
            resolver: IncludeResolver instance to use.
        """
        self.resolver = resolver
        self._usage_map: dict[str, list[str]] = {}  # include_path -> [article_paths]

    async def analyze_usage(
        self,
        articles: list[tuple[Path, str]],
    ) -> dict[str, list[str]]:
        """Analyze which articles use which include files.

        Args:
            articles: List of (path, content) tuples for articles.

        Returns:
            Dictionary mapping include paths to lists of article paths.
        """
        usage_map: dict[str, list[str]] = {}

        for article_path, content in articles:
            include_paths = await self.resolver.get_include_paths(
                content, article_path
            )

            for include_path in include_paths:
                if include_path not in usage_map:
                    usage_map[include_path] = []
                usage_map[include_path].append(str(article_path))

        self._usage_map = usage_map
        return usage_map

    def get_unused_includes(
        self,
        include_dir: Path,
    ) -> list[Path]:
        """Find include files that aren't used by any article.

        Args:
            include_dir: Directory containing include files.

        Returns:
            List of unused include file paths.
        """
        if not include_dir.exists():
            return []

        used_paths = set(self._usage_map.keys())
        unused = []

        for path in include_dir.rglob("*.md"):
            if str(path.resolve()) not in used_paths:
                unused.append(path)

        return unused

    def get_most_used_includes(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get the most frequently used include files.

        Args:
            limit: Maximum number of results.

        Returns:
            List of (include_path, usage_count) tuples, sorted by count.
        """
        usage_counts = [
            (path, len(articles))
            for path, articles in self._usage_map.items()
        ]
        usage_counts.sort(key=lambda x: x[1], reverse=True)
        return usage_counts[:limit]
