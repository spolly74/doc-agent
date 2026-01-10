"""Extract metadata and structure from documentation articles."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import frontmatter

from doc_agent.models.article import (
    Article,
    ArticleMetadata,
    ArticleStructure,
    CodeBlock,
)


class MetadataExtractor:
    """Extracts metadata and structural information from markdown articles."""

    # Regex patterns for structural analysis
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(
        r"```(\w+)?\s*\n(.*?)```", re.DOTALL | re.MULTILINE
    )
    INCLUDE_PATTERN = re.compile(
        r"\[!INCLUDE\s*\[([^\]]*)\]\(([^)]+)\)\]", re.IGNORECASE
    )
    CALLOUT_PATTERN = re.compile(
        r"^>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]", re.MULTILINE | re.IGNORECASE
    )
    LIST_PATTERN = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
    NUMBERED_LIST_PATTERN = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)
    TABLE_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    # Section patterns for structure detection
    PREREQUISITES_PATTERNS = [
        re.compile(r"^#{1,3}\s*prerequisites?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#{1,3}\s*before\s+you\s+begin\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#{1,3}\s*requirements?\s*$", re.IGNORECASE | re.MULTILINE),
    ]
    NEXT_STEPS_PATTERNS = [
        re.compile(r"^#{1,3}\s*next\s+steps?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#{1,3}\s*what'?s?\s+next\s*$", re.IGNORECASE | re.MULTILINE),
    ]
    SEE_ALSO_PATTERNS = [
        re.compile(r"^#{1,3}\s*see\s+also\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#{1,3}\s*related\s+(content|articles?|topics?)\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    def extract(self, content: str) -> ArticleMetadata:
        """Extract metadata from article content.

        Args:
            content: Raw markdown content including frontmatter.

        Returns:
            Parsed ArticleMetadata object.
        """
        # Parse frontmatter
        post = frontmatter.loads(content)
        fm = post.metadata

        # Extract standard fields
        title = fm.get("title", "Untitled")

        # Handle ms.date parsing
        ms_date = None
        if date_value := fm.get("ms.date"):
            ms_date = self._parse_date(date_value)

        # Build metadata object
        metadata = ArticleMetadata(
            title=title,
            description=fm.get("description"),
            ms_author=fm.get("ms.author"),
            ms_date=ms_date,
            ms_topic=fm.get("ms.topic"),
            ms_service=fm.get("ms.service"),
            ms_subservice=fm.get("ms.subservice"),
            ms_custom=self._ensure_list(fm.get("ms.custom", [])),
            customer_intent=fm.get("customer-intent"),
            zone_pivot_groups=self._ensure_list(fm.get("zone_pivot_groups", [])),
            custom_fields={
                k: v for k, v in fm.items()
                if not k.startswith("ms.") and k not in (
                    "title", "description", "customer-intent", "zone_pivot_groups"
                )
            },
        )

        return metadata

    def extract_structure(self, content: str) -> ArticleStructure:
        """Extract structural information from article content.

        Args:
            content: Raw markdown content.

        Returns:
            ArticleStructure with heading hierarchy, counts, etc.
        """
        # Remove frontmatter for content analysis
        post = frontmatter.loads(content)
        body = post.content

        # Extract headings
        headings = []
        heading_levels = []
        for match in self.HEADING_PATTERN.finditer(body):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append(f"H{level}: {text}")
            heading_levels.append(level)

        # Check heading hierarchy validity
        hierarchy_valid = self._check_heading_hierarchy(heading_levels)

        # Extract code blocks
        code_blocks = list(self.CODE_BLOCK_PATTERN.finditer(body))
        code_block_count = len(code_blocks)
        first_code_line = None
        if code_blocks:
            first_code_line = body[:code_blocks[0].start()].count("\n") + 1

        # Count other elements
        callout_count = len(self.CALLOUT_PATTERN.findall(body))
        list_count = len(self.LIST_PATTERN.findall(body)) + len(
            self.NUMBERED_LIST_PATTERN.findall(body)
        )
        table_count = len(set(self.TABLE_PATTERN.findall(body))) // 2  # Rough estimate
        image_count = len(self.IMAGE_PATTERN.findall(body))
        link_count = len(self.LINK_PATTERN.findall(body))
        include_count = len(self.INCLUDE_PATTERN.findall(body))

        # Count words and paragraphs
        # Remove code blocks for word count
        text_only = self.CODE_BLOCK_PATTERN.sub("", body)
        words = text_only.split()
        word_count = len(words)

        # Count paragraphs (sequences of non-empty lines)
        paragraphs = re.split(r"\n\s*\n", text_only)
        paragraph_count = len([p for p in paragraphs if p.strip()])

        # Check for standard sections
        has_prerequisites = any(p.search(body) for p in self.PREREQUISITES_PATTERNS)
        has_next_steps = any(p.search(body) for p in self.NEXT_STEPS_PATTERNS)
        has_see_also = any(p.search(body) for p in self.SEE_ALSO_PATTERNS)

        return ArticleStructure(
            headings=headings,
            heading_levels=heading_levels,
            heading_hierarchy_valid=hierarchy_valid,
            has_prerequisites=has_prerequisites,
            has_next_steps=has_next_steps,
            has_see_also=has_see_also,
            code_block_count=code_block_count,
            first_code_block_line=first_code_line,
            word_count=word_count,
            paragraph_count=paragraph_count,
            list_count=list_count,
            table_count=table_count,
            callout_count=callout_count,
            image_count=image_count,
            link_count=link_count,
            include_count=include_count,
        )

    def extract_code_blocks(self, content: str) -> list[CodeBlock]:
        """Extract all code blocks from article content.

        Args:
            content: Raw markdown content.

        Returns:
            List of CodeBlock objects.
        """
        post = frontmatter.loads(content)
        body = post.content

        code_blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(body):
            language = match.group(1)
            code_content = match.group(2)

            # Calculate line numbers
            start_pos = match.start()
            end_pos = match.end()
            line_start = body[:start_pos].count("\n") + 1
            line_end = body[:end_pos].count("\n") + 1

            # Check for interactive markers
            is_interactive = self._is_interactive_block(language, code_content)

            # Check for source reference (GitHub link in preceding text)
            source_ref = self._find_source_reference(body, start_pos)

            code_blocks.append(CodeBlock(
                language=language,
                content=code_content.strip(),
                line_start=line_start,
                line_end=line_end,
                is_interactive=is_interactive,
                source_reference=source_ref,
            ))

        return code_blocks

    def extract_includes(self, content: str) -> list[str]:
        """Extract include file references from article content.

        Args:
            content: Raw markdown content.

        Returns:
            List of include file paths.
        """
        post = frontmatter.loads(content)
        body = post.content

        includes = []
        for match in self.INCLUDE_PATTERN.finditer(body):
            include_path = match.group(2)
            includes.append(include_path)

        return includes

    def create_article(
        self,
        path: Path,
        docs_root: Path,
        content: Optional[str] = None,
    ) -> Article:
        """Create a complete Article object from a file.

        Args:
            path: Path to the markdown file.
            docs_root: Root directory of the documentation.
            content: Optional pre-loaded content (loaded from file if not provided).

        Returns:
            Fully populated Article object.
        """
        if content is None:
            content = path.read_text(encoding="utf-8")

        # Extract all components
        metadata = self.extract(content)
        structure = self.extract_structure(content)
        code_blocks = self.extract_code_blocks(content)
        includes = self.extract_includes(content)

        # Get content without frontmatter
        post = frontmatter.loads(content)

        # Calculate relative path
        try:
            relative_path = str(path.relative_to(docs_root))
        except ValueError:
            relative_path = str(path)

        return Article(
            path=path,
            relative_path=relative_path,
            content=content,
            content_without_frontmatter=post.content,
            metadata=metadata,
            structure=structure,
            code_blocks=code_blocks,
            includes=includes,
        )

    def _parse_date(self, date_value: str | datetime) -> Optional[datetime]:
        """Parse a date value from frontmatter.

        Args:
            date_value: Date string or datetime object.

        Returns:
            Parsed datetime or None if parsing fails.
        """
        if isinstance(date_value, datetime):
            return date_value

        if not isinstance(date_value, str):
            return None

        # Try common date formats
        formats = [
            "%m/%d/%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_value, fmt)
            except ValueError:
                continue

        return None

    def _ensure_list(self, value: str | list) -> list:
        """Ensure a value is a list.

        Args:
            value: String or list value.

        Returns:
            List (wrapping string in list if needed).
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        return []

    def _check_heading_hierarchy(self, levels: list[int]) -> bool:
        """Check if heading hierarchy is valid (no skipped levels).

        Args:
            levels: List of heading levels (1-6).

        Returns:
            True if hierarchy is valid, False otherwise.
        """
        if not levels:
            return True

        # First heading should be H1 or H2
        if levels[0] > 2:
            return False

        # Check for skipped levels
        for i in range(1, len(levels)):
            # Can go deeper by 1 level, or go up any amount
            if levels[i] > levels[i - 1] + 1:
                return False

        return True

    def _is_interactive_block(self, language: Optional[str], content: str) -> bool:
        """Check if a code block is interactive (Cloud Shell, Try It, etc.).

        Args:
            language: Language tag on the code block.
            content: Code block content.

        Returns:
            True if the block appears to be interactive.
        """
        if not language:
            return False

        language = language.lower()

        # Check for interactive language tags
        interactive_tags = ["azurecli-interactive", "azurepowershell-interactive", "cloudshell"]
        if language in interactive_tags:
            return True

        # Check for Try It annotations
        if "try it" in content.lower():
            return True

        return False

    def _find_source_reference(self, content: str, code_block_start: int) -> Optional[str]:
        """Find a GitHub source reference near a code block.

        Args:
            content: Full article content.
            code_block_start: Start position of the code block.

        Returns:
            GitHub URL if found, None otherwise.
        """
        # Look in the 500 characters before the code block
        search_start = max(0, code_block_start - 500)
        search_region = content[search_start:code_block_start]

        # Look for GitHub links
        github_pattern = re.compile(
            r"https://github\.com/[^\s\)\]]+",
            re.IGNORECASE
        )
        matches = github_pattern.findall(search_region)

        if matches:
            return matches[-1]  # Return the closest one

        return None
