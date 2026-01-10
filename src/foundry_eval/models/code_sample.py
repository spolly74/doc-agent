"""Code sample models for validation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SampleStatus(str, Enum):
    """Status of a code sample."""

    VALID = "valid"
    INVALID = "invalid"
    ORPHANED = "orphaned"
    MISSING = "missing"
    OUTDATED = "outdated"


class SampleLanguage(str, Enum):
    """Supported programming languages for samples."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    SHELL = "shell"
    POWERSHELL = "powershell"
    JSON = "json"
    YAML = "yaml"
    OTHER = "other"


class CodeSample(BaseModel):
    """A code sample from the samples repository."""

    sample_id: str
    file_path: str
    language: SampleLanguage
    title: str = ""
    description: str = ""

    # Source information
    source_file: str  # Full path to the sample file
    start_line: int = 1
    end_line: Optional[int] = None
    content: str = ""

    # Metadata
    tags: list[str] = Field(default_factory=list)
    sdk_version: str = ""
    api_version: str = ""

    # Reference tracking
    referenced_by: list[str] = Field(default_factory=list)  # Article paths
    reference_count: int = 0

    @property
    def is_orphaned(self) -> bool:
        """Check if the sample is not referenced by any article."""
        return self.reference_count == 0

    @property
    def line_count(self) -> int:
        """Number of lines in the sample."""
        return len(self.content.splitlines()) if self.content else 0


class SampleReference(BaseModel):
    """A reference to a code sample from an article."""

    article_path: str
    sample_id: str
    reference_type: str  # "include", "link", "inline"
    line_number: Optional[int] = None

    # Validation
    is_valid: bool = True
    validation_error: str = ""


class SampleValidationIssue(BaseModel):
    """An issue found during sample validation."""

    sample_id: str
    issue_type: str  # "syntax", "deprecated_api", "missing_import", "incomplete", etc.
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    line_number: Optional[int] = None
    suggestion: str = ""
    auto_fixable: bool = False


class SampleValidationResult(BaseModel):
    """Result of validating a single code sample."""

    sample: CodeSample
    status: SampleStatus
    issues: list[SampleValidationIssue] = Field(default_factory=list)

    # Quality scores
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    correctness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_practices_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Analysis
    has_error_handling: bool = False
    has_comments: bool = False
    is_runnable: bool = False

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (
            self.completeness_score * 0.4 +
            self.correctness_score * 0.4 +
            self.best_practices_score * 0.2
        )

    @property
    def is_valid(self) -> bool:
        """Check if the sample passes validation."""
        return self.status == SampleStatus.VALID and not any(
            i.severity == "critical" for i in self.issues
        )


class SamplesCoverageReport(BaseModel):
    """Report on code sample coverage across documentation."""

    # Counts
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    orphaned_samples: int = 0

    # Article coverage
    total_articles: int = 0
    articles_with_samples: int = 0
    articles_needing_samples: int = 0

    # Validation results
    validation_results: list[SampleValidationResult] = Field(default_factory=list)

    # Orphaned samples (not referenced by any article)
    orphaned_sample_paths: list[str] = Field(default_factory=list)

    # Articles that should have samples but don't
    articles_missing_samples: list[str] = Field(default_factory=list)

    # Broken references
    broken_references: list[SampleReference] = Field(default_factory=list)

    @property
    def sample_coverage_percentage(self) -> float:
        """Percentage of articles that have samples when needed."""
        if self.articles_needing_samples == 0:
            return 100.0
        return (
            (self.articles_needing_samples - len(self.articles_missing_samples))
            / self.articles_needing_samples
        ) * 100

    @property
    def validation_pass_rate(self) -> float:
        """Percentage of samples that pass validation."""
        if self.total_samples == 0:
            return 100.0
        return (self.valid_samples / self.total_samples) * 100
