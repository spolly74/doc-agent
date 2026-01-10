"""Article domain models."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from foundry_eval.models.enums import ContentPattern


class CodeBlock(BaseModel):
    """Represents a code block within an article."""

    language: Optional[str] = None
    content: str
    line_start: int
    line_end: int
    is_interactive: bool = False  # Azure Cloud Shell, Try It, etc.
    source_reference: Optional[str] = None  # Link to external sample

    @property
    def line_count(self) -> int:
        """Number of lines in the code block."""
        return self.line_end - self.line_start + 1


class ArticleMetadata(BaseModel):
    """Frontmatter metadata extracted from an article."""

    # Required fields
    title: str

    # Common ms.* fields
    description: Optional[str] = None
    ms_author: Optional[str] = Field(default=None, alias="ms.author")
    ms_date: Optional[datetime] = Field(default=None, alias="ms.date")
    ms_topic: Optional[str] = Field(default=None, alias="ms.topic")
    ms_service: Optional[str] = Field(default=None, alias="ms.service")
    ms_subservice: Optional[str] = Field(default=None, alias="ms.subservice")
    ms_custom: list[str] = Field(default_factory=list, alias="ms.custom")

    # Customer intent
    customer_intent: Optional[str] = Field(default=None, alias="customer-intent")

    # Zone pivots (for multi-language content)
    zone_pivot_groups: list[str] = Field(default_factory=list)

    # Additional metadata stored as dict
    custom_fields: dict = Field(default_factory=dict)

    @property
    def content_pattern(self) -> ContentPattern:
        """Get the content pattern from ms.topic."""
        return ContentPattern.from_ms_topic(self.ms_topic)

    @property
    def age_days(self) -> Optional[int]:
        """Days since last update (based on ms.date)."""
        if self.ms_date is None:
            return None
        delta = datetime.now() - self.ms_date
        return delta.days

    model_config = ConfigDict(populate_by_name=True)  # Allow both alias and field name


class ArticleStructure(BaseModel):
    """Structural analysis of an article."""

    headings: list[str] = Field(default_factory=list)
    heading_levels: list[int] = Field(default_factory=list)
    heading_hierarchy_valid: bool = True
    has_prerequisites: bool = False
    has_next_steps: bool = False
    has_see_also: bool = False
    code_block_count: int = 0
    first_code_block_line: Optional[int] = None
    word_count: int = 0
    paragraph_count: int = 0
    list_count: int = 0
    table_count: int = 0
    callout_count: int = 0  # Note, Warning, Tip, etc.
    image_count: int = 0
    link_count: int = 0
    include_count: int = 0


class Article(BaseModel):
    """Represents a documentation article."""

    # File information
    path: Path
    relative_path: str  # Relative to docs root

    # Content
    content: str
    content_without_frontmatter: str = ""

    # Parsed data
    metadata: ArticleMetadata
    structure: ArticleStructure = Field(default_factory=ArticleStructure)
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)  # Resolved include paths

    # TOC position (for navigation context)
    toc_section: Optional[str] = None
    toc_order: Optional[int] = None

    @property
    def word_count(self) -> int:
        """Total word count of the article content."""
        return self.structure.word_count

    @property
    def has_code(self) -> bool:
        """Whether the article contains any code blocks."""
        return len(self.code_blocks) > 0

    @property
    def code_to_prose_ratio(self) -> float:
        """Ratio of code lines to prose lines."""
        if self.structure.word_count == 0:
            return 0.0
        code_lines = sum(cb.line_count for cb in self.code_blocks)
        # Rough estimate: ~10 words per line of prose
        prose_lines = self.structure.word_count / 10
        if prose_lines == 0:
            return float("inf") if code_lines > 0 else 0.0
        return code_lines / prose_lines

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow Path
