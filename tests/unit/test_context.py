"""Tests for the context layer components."""

from datetime import datetime
from pathlib import Path

import pytest

from foundry_eval.context.metadata_extractor import MetadataExtractor
from foundry_eval.context.toc_parser import TOCParser, TOCEntry
from foundry_eval.context.include_resolver import IncludeResolver
from foundry_eval.models.enums import ContentPattern


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        return MetadataExtractor()

    def test_extract_basic_metadata(self, extractor: MetadataExtractor, sample_article_content: str):
        """Test extracting basic metadata from article."""
        metadata = extractor.extract(sample_article_content)

        assert metadata.title == "Guardrails and controls overview in Microsoft Foundry"
        assert metadata.ms_topic == "conceptual"
        assert metadata.ms_author == "ssalgado"
        assert metadata.ms_service == "azure-ai-foundry"
        assert "azure-ai-guardrails" in metadata.ms_custom

    def test_extract_customer_intent(self, extractor: MetadataExtractor, sample_article_content: str):
        """Test extracting customer intent."""
        metadata = extractor.extract(sample_article_content)

        assert metadata.customer_intent is not None
        assert "developer" in metadata.customer_intent.lower()
        assert "guardrails" in metadata.customer_intent.lower()

    def test_extract_date(self, extractor: MetadataExtractor, sample_article_content: str):
        """Test extracting and parsing ms.date."""
        metadata = extractor.extract(sample_article_content)

        assert metadata.ms_date is not None
        assert isinstance(metadata.ms_date, datetime)
        assert metadata.ms_date.year == 2025
        assert metadata.ms_date.month == 11

    def test_content_pattern(self, extractor: MetadataExtractor, sample_article_content: str):
        """Test content pattern detection."""
        metadata = extractor.extract(sample_article_content)

        assert metadata.content_pattern == ContentPattern.CONCEPT

    def test_extract_structure(self, extractor: MetadataExtractor, sample_article_content: str):
        """Test extracting article structure."""
        structure = extractor.extract_structure(sample_article_content)

        assert len(structure.headings) > 0
        assert structure.heading_hierarchy_valid is True
        assert structure.has_prerequisites is True
        assert structure.has_next_steps is True
        assert structure.code_block_count == 2
        assert structure.table_count >= 1

    def test_extract_code_blocks(self, extractor: MetadataExtractor, sample_article_content: str):
        """Test extracting code blocks."""
        code_blocks = extractor.extract_code_blocks(sample_article_content)

        assert len(code_blocks) == 2
        assert code_blocks[0].language == "python"
        assert "GuardrailConfig" in code_blocks[0].content
        assert code_blocks[1].language == "python"
        assert "apply_guardrail" in code_blocks[1].content

    def test_create_article(
        self,
        extractor: MetadataExtractor,
        sample_article_path: Path,
        fixtures_dir: Path,
    ):
        """Test creating a complete Article object."""
        article = extractor.create_article(
            path=sample_article_path,
            docs_root=fixtures_dir,
        )

        assert article.path == sample_article_path
        assert article.relative_path == "sample_article.md"
        assert article.metadata.title == "Guardrails and controls overview in Microsoft Foundry"
        assert len(article.code_blocks) == 2
        assert article.has_code is True
        assert article.word_count > 0


class TestTOCParser:
    """Tests for TOCParser."""

    def test_parse_entry(self):
        """Test parsing a single TOC entry."""
        entry = TOCEntry(
            name="Getting started",
            href="getting-started.md",
        )

        assert entry.name == "Getting started"
        assert entry.href == "getting-started.md"
        assert entry.is_article is True
        assert entry.is_section is False

    def test_parse_section(self):
        """Test parsing a section with children."""
        entry = TOCEntry(
            name="Concepts",
            items=[
                TOCEntry(name="Overview", href="overview.md"),
                TOCEntry(name="Architecture", href="architecture.md"),
            ],
        )

        assert entry.is_section is True
        assert entry.is_article is False
        assert len(entry.items) == 2

    def test_all_articles(self):
        """Test getting all articles from a TOC structure."""
        entry = TOCEntry(
            name="Docs",
            items=[
                TOCEntry(name="Overview", href="index.md"),
                TOCEntry(
                    name="Guides",
                    items=[
                        TOCEntry(name="Setup", href="setup.md"),
                        TOCEntry(name="Deploy", href="deploy.md"),
                    ],
                ),
            ],
        )

        articles = entry.all_articles()
        assert len(articles) == 3
        hrefs = [href for href, _ in articles]
        assert "index.md" in hrefs
        assert "setup.md" in hrefs
        assert "deploy.md" in hrefs

    @pytest.mark.asyncio
    async def test_parse_toc_file(self, fixtures_dir: Path):
        """Test parsing a TOC file."""
        parser = TOCParser(fixtures_dir)
        entries = await parser.parse_toc(fixtures_dir / "sample_toc.yml")

        assert len(entries) > 0
        # First entry should be Overview
        assert entries[0].name == "Overview"
        assert entries[0].href == "index.md"


class TestIncludeResolver:
    """Tests for IncludeResolver."""

    @pytest.fixture
    def resolver(self, fixtures_dir: Path) -> IncludeResolver:
        return IncludeResolver(fixtures_dir)

    def test_find_includes(self, resolver: IncludeResolver):
        """Test finding include directives."""
        content = """
# Article

Some text before.

[!INCLUDE [prereqs](../includes/prereqs.md)]

More text.

[!INCLUDE [note](./includes/note.md)]
"""
        includes = resolver.find_includes_sync(content)

        assert len(includes) == 2
        assert includes[0] == ("prereqs", "../includes/prereqs.md")
        assert includes[1] == ("note", "./includes/note.md")

    def test_no_includes(self, resolver: IncludeResolver):
        """Test content with no includes."""
        content = """
# Article

Just regular content with no includes.
"""
        includes = resolver.find_includes_sync(content)
        assert len(includes) == 0

    @pytest.mark.asyncio
    async def test_get_include_paths(
        self,
        resolver: IncludeResolver,
        sample_article_path: Path,
        sample_article_content: str,
    ):
        """Test getting include paths from article."""
        paths = await resolver.get_include_paths(
            sample_article_content,
            sample_article_path,
        )
        # Sample article has no includes
        assert len(paths) == 0
