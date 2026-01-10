"""Samples repository indexer for code sample tracking."""

import logging
import re
from pathlib import Path
from typing import Optional

from doc_agent.models.code_sample import (
    CodeSample,
    SampleLanguage,
    SampleReference,
)

logger = logging.getLogger("doc_agent.context.samples_index")

# Language detection by file extension
EXTENSION_TO_LANGUAGE = {
    ".py": SampleLanguage.PYTHON,
    ".js": SampleLanguage.JAVASCRIPT,
    ".ts": SampleLanguage.TYPESCRIPT,
    ".tsx": SampleLanguage.TYPESCRIPT,
    ".cs": SampleLanguage.CSHARP,
    ".java": SampleLanguage.JAVA,
    ".go": SampleLanguage.GO,
    ".rs": SampleLanguage.RUST,
    ".sh": SampleLanguage.SHELL,
    ".bash": SampleLanguage.SHELL,
    ".ps1": SampleLanguage.POWERSHELL,
    ".json": SampleLanguage.JSON,
    ".yaml": SampleLanguage.YAML,
    ".yml": SampleLanguage.YAML,
}

# Patterns for finding sample references in articles
SAMPLE_REFERENCE_PATTERNS = [
    # GitHub-style includes: :::code language="python" source="~/samples/file.py":::
    r':::code\s+[^:]*source="([^"]+)"[^:]*:::',
    # Docfx-style includes: [!code-python[](~/samples/file.py)]
    r'\[!code-\w+\[\]\(([^)]+)\)\]',
    # Markdown-style links to samples: [Sample](../samples/file.py)
    r'\[(?:Sample|Code|Example)[^\]]*\]\(([^)]+\.(?:py|js|ts|cs|java|go|rs))\)',
    # Direct file references in code blocks
    r'#\s*(?:file|source):\s*([^\s]+)',
]


class SamplesIndex:
    """Index of code samples from a samples repository."""

    def __init__(self, samples_repo: Path, articles_repo: Optional[Path] = None):
        """Initialize the samples index.

        Args:
            samples_repo: Path to the samples repository.
            articles_repo: Optional path to articles repository for reference tracking.
        """
        self._samples_repo = samples_repo
        self._articles_repo = articles_repo
        self._samples: dict[str, CodeSample] = {}
        self._references: list[SampleReference] = []
        self._indexed = False

    @property
    def samples(self) -> list[CodeSample]:
        """Get all indexed samples."""
        return list(self._samples.values())

    @property
    def references(self) -> list[SampleReference]:
        """Get all found references."""
        return self._references

    async def index(self) -> int:
        """Index all samples in the repository.

        Returns:
            Number of samples indexed.
        """
        if self._indexed:
            return len(self._samples)

        self._samples.clear()
        self._references.clear()

        # Find all code files
        count = 0
        for ext, lang in EXTENSION_TO_LANGUAGE.items():
            for file_path in self._samples_repo.rglob(f"*{ext}"):
                # Skip common non-sample directories
                if self._should_skip(file_path):
                    continue

                sample = await self._index_file(file_path, lang)
                if sample:
                    self._samples[sample.sample_id] = sample
                    count += 1

        # If articles repo provided, find references
        if self._articles_repo:
            await self._find_references()

        self._indexed = True
        logger.info(f"Indexed {count} samples from {self._samples_repo}")
        return count

    async def get_sample(self, sample_id: str) -> Optional[CodeSample]:
        """Get a sample by ID.

        Args:
            sample_id: The sample ID (usually relative path).

        Returns:
            CodeSample if found, None otherwise.
        """
        if not self._indexed:
            await self.index()
        return self._samples.get(sample_id)

    async def get_orphaned_samples(self) -> list[CodeSample]:
        """Get samples that are not referenced by any article.

        Returns:
            List of orphaned samples.
        """
        if not self._indexed:
            await self.index()

        return [s for s in self._samples.values() if s.is_orphaned]

    async def get_samples_by_language(self, language: SampleLanguage) -> list[CodeSample]:
        """Get all samples of a specific language.

        Args:
            language: The programming language.

        Returns:
            List of samples.
        """
        if not self._indexed:
            await self.index()

        return [s for s in self._samples.values() if s.language == language]

    def _should_skip(self, file_path: Path) -> bool:
        """Check if a file should be skipped during indexing.

        Args:
            file_path: Path to check.

        Returns:
            True if the file should be skipped.
        """
        skip_patterns = [
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".tox",
            "build",
            "dist",
            ".eggs",
            "*.egg-info",
        ]

        path_str = str(file_path)
        for pattern in skip_patterns:
            if pattern in path_str:
                return True

        return False

    async def _index_file(self, file_path: Path, language: SampleLanguage) -> Optional[CodeSample]:
        """Index a single code file.

        Args:
            file_path: Path to the code file.
            language: Detected language.

        Returns:
            CodeSample object or None if indexing fails.
        """
        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8")

            # Generate sample ID from relative path
            relative_path = file_path.relative_to(self._samples_repo)
            sample_id = str(relative_path).replace("\\", "/")

            # Extract title and description from file header comments
            title, description = self._extract_metadata(content, language)

            return CodeSample(
                sample_id=sample_id,
                file_path=sample_id,
                language=language,
                title=title or file_path.stem,
                description=description,
                source_file=str(file_path),
                start_line=1,
                end_line=len(content.splitlines()),
                content=content,
            )
        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")
            return None

    def _extract_metadata(self, content: str, language: SampleLanguage) -> tuple[str, str]:
        """Extract title and description from file header comments.

        Args:
            content: File content.
            language: Programming language.

        Returns:
            Tuple of (title, description).
        """
        title = ""
        description = ""

        lines = content.splitlines()
        if not lines:
            return title, description

        # Python-style docstring
        if language == SampleLanguage.PYTHON:
            match = re.match(r'^"""(.+?)"""', content, re.DOTALL)
            if match:
                doc = match.group(1).strip()
                parts = doc.split("\n", 1)
                title = parts[0].strip()
                if len(parts) > 1:
                    description = parts[1].strip()
                return title, description

        # C-style comments
        if language in (SampleLanguage.JAVASCRIPT, SampleLanguage.TYPESCRIPT,
                       SampleLanguage.CSHARP, SampleLanguage.JAVA, SampleLanguage.GO):
            match = re.match(r'^/\*\*?\s*(.+?)\s*\*/', content, re.DOTALL)
            if match:
                doc = match.group(1).strip()
                # Clean up asterisks
                doc = re.sub(r'^\s*\*\s*', '', doc, flags=re.MULTILINE)
                parts = doc.split("\n", 1)
                title = parts[0].strip()
                if len(parts) > 1:
                    description = parts[1].strip()
                return title, description

        # Check for title in first line comment
        first_line = lines[0].strip()
        comment_prefixes = ["#", "//", "--"]
        for prefix in comment_prefixes:
            if first_line.startswith(prefix):
                title = first_line[len(prefix):].strip()
                break

        return title, description

    async def _find_references(self) -> None:
        """Find all sample references in articles."""
        if not self._articles_repo:
            return

        for md_file in self._articles_repo.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                article_path = str(md_file.relative_to(self._articles_repo))

                for pattern in SAMPLE_REFERENCE_PATTERNS:
                    for match in re.finditer(pattern, content):
                        sample_path = match.group(1)
                        # Normalize the path
                        sample_id = self._normalize_sample_path(sample_path)

                        ref = SampleReference(
                            article_path=article_path,
                            sample_id=sample_id,
                            reference_type="include",
                        )

                        # Check if sample exists
                        if sample_id in self._samples:
                            self._samples[sample_id].referenced_by.append(article_path)
                            self._samples[sample_id].reference_count += 1
                            ref.is_valid = True
                        else:
                            ref.is_valid = False
                            ref.validation_error = f"Sample not found: {sample_id}"

                        self._references.append(ref)

            except Exception as e:
                logger.warning(f"Failed to scan {md_file}: {e}")

    def _normalize_sample_path(self, path: str) -> str:
        """Normalize a sample path reference.

        Args:
            path: The raw path from the article.

        Returns:
            Normalized sample ID.
        """
        # Remove common prefixes
        path = path.strip()
        for prefix in ["~/", "../", "./", "~/"]:
            if path.startswith(prefix):
                path = path[len(prefix):]

        # Remove samples/ prefix if present
        if path.startswith("samples/"):
            path = path[8:]

        return path.replace("\\", "/")


def get_samples_index(
    samples_repo: Path,
    articles_repo: Optional[Path] = None,
) -> SamplesIndex:
    """Factory function to create a samples index.

    Args:
        samples_repo: Path to samples repository.
        articles_repo: Optional path to articles repository.

    Returns:
        SamplesIndex instance.
    """
    return SamplesIndex(samples_repo, articles_repo)
