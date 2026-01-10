"""Job-To-Be-Done (JTBD) models for coverage analysis."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JTBDCoverageStatus(str, Enum):
    """Coverage status for a JTBD step."""

    FULLY_COVERED = "fully_covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    OUTDATED = "outdated"


class JTBDStep(BaseModel):
    """A single step within a Job-To-Be-Done."""

    step_id: str
    step_number: int
    title: str
    description: str
    required_skills: list[str] = Field(default_factory=list)
    expected_outcomes: list[str] = Field(default_factory=list)

    # Coverage tracking
    coverage_status: JTBDCoverageStatus = JTBDCoverageStatus.NOT_COVERED
    mapped_articles: list[str] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_notes: str = ""


class JTBD(BaseModel):
    """A complete Job-To-Be-Done definition."""

    jtbd_id: str
    title: str
    description: str
    persona: str = ""
    category: str = ""
    priority: int = Field(default=0, ge=0, le=10)

    # Steps in the journey (optional - may be empty for simple JTBDs)
    steps: list[JTBDStep] = Field(default_factory=list)

    # Prerequisites
    prerequisites: list[str] = Field(default_factory=list)

    # Tags for categorization
    tags: list[str] = Field(default_factory=list)

    # Additional metadata from Golden Path format
    url: str = ""  # GitHub issue or documentation URL
    phase: str = ""  # e.g., "Idea to Proto", "Build to Deploy"
    area: str = ""  # e.g., "Provision", "Data & Pipeline"
    status: str = ""  # e.g., "Writing in Progress", "Published"
    target_grade: str = ""  # Target quality grade
    assignees: list[str] = Field(default_factory=list)

    # Coverage tracking at JTBD level (for step-less JTBDs)
    coverage_status: JTBDCoverageStatus = JTBDCoverageStatus.NOT_COVERED
    mapped_articles: list[str] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def total_steps(self) -> int:
        """Total number of steps."""
        return len(self.steps)

    @property
    def covered_steps(self) -> int:
        """Number of fully or partially covered steps."""
        return sum(
            1 for s in self.steps
            if s.coverage_status in (
                JTBDCoverageStatus.FULLY_COVERED,
                JTBDCoverageStatus.PARTIALLY_COVERED,
            )
        )

    @property
    def coverage_percentage(self) -> float:
        """Overall coverage percentage."""
        # For step-less JTBDs, use JTBD-level coverage
        if not self.steps:
            if self.coverage_status in (
                JTBDCoverageStatus.FULLY_COVERED,
                JTBDCoverageStatus.PARTIALLY_COVERED,
            ):
                return self.coverage_score * 100
            return 0.0
        return (self.covered_steps / self.total_steps) * 100

    @property
    def is_stepless(self) -> bool:
        """Check if this JTBD has no steps defined."""
        return len(self.steps) == 0


class JTBDMapping(BaseModel):
    """Mapping between a JTBD step and documentation articles."""

    jtbd_id: str
    step_id: str
    article_path: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    covers_fully: bool = False
    mapping_notes: str = ""

    # Which aspects of the step are covered
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)


class CoverageGap(BaseModel):
    """A gap in documentation coverage for a JTBD."""

    jtbd_id: str
    step_id: str
    gap_type: str  # "missing", "incomplete", "outdated"
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str

    # Recommendations
    suggested_article_title: Optional[str] = None
    suggested_content_outline: list[str] = Field(default_factory=list)
    estimated_effort: str = ""  # "small", "medium", "large"

    # Related existing articles that could be expanded
    related_articles: list[str] = Field(default_factory=list)


class JTBDAnalysisResult(BaseModel):
    """Result of analyzing JTBD coverage."""

    jtbd: JTBD
    mappings: list[JTBDMapping] = Field(default_factory=list)
    gaps: list[CoverageGap] = Field(default_factory=list)

    # Summary statistics
    total_articles_mapped: int = 0
    overall_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Recommendations
    priority_gaps: list[str] = Field(default_factory=list)
    suggested_new_articles: list[str] = Field(default_factory=list)

    @property
    def has_critical_gaps(self) -> bool:
        """Check if there are any critical gaps."""
        return any(g.severity == "critical" for g in self.gaps)

    @property
    def gap_count_by_severity(self) -> dict[str, int]:
        """Count gaps by severity."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for gap in self.gaps:
            if gap.severity in counts:
                counts[gap.severity] += 1
        return counts
