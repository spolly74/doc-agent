"""Evaluation result models."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from doc_agent.models.enums import Dimension, Severity


class Issue(BaseModel):
    """An issue detected during evaluation."""

    id: int = 0
    dimension: Dimension
    severity: Severity
    title: str
    description: str
    location: Optional[str] = None  # Line number, section, or "general"
    suggestion: Optional[str] = None
    auto_fixable: bool = False

    @property
    def is_critical(self) -> bool:
        """Whether this is a critical issue."""
        return self.severity == Severity.CRITICAL

    @property
    def is_blocking(self) -> bool:
        """Whether this issue should block publishing."""
        return self.severity in (Severity.CRITICAL, Severity.HIGH)


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: Dimension
    score: Optional[float] = Field(default=None, ge=1.0, le=7.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""
    issues: list[Issue] = Field(default_factory=list)

    @property
    def is_applicable(self) -> bool:
        """Whether this dimension is applicable (score is not None)."""
        return self.score is not None

    @property
    def is_below_threshold(self) -> bool:
        """Whether this score is below the default threshold (4.0)."""
        return self.score is not None and self.score < 4.0


class EvaluationResult(BaseModel):
    """Complete evaluation result for an article."""

    # Article identification
    article_path: str
    article_title: str

    # Article metadata (from frontmatter)
    ms_author: Optional[str] = None
    ms_date: Optional[str] = None  # Stored as string for serialization
    ms_topic: Optional[str] = None
    ms_subservice: Optional[str] = None
    has_code_samples: bool = False

    # Timing
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Scores
    dimension_scores: list[DimensionScore] = Field(default_factory=list)

    # Issues (flattened from all dimensions)
    issues: list[Issue] = Field(default_factory=list)

    # Metadata
    is_quick_scan: bool = False
    needs_deep_evaluation: bool = False
    llm_model_used: str = ""
    token_usage: Optional[dict] = None
    evaluation_duration_seconds: Optional[float] = None

    @computed_field
    @property
    def overall_score(self) -> float:
        """Calculate the weighted overall score.

        Uses the weights from the spec:
        - pattern_compliance: 0.15
        - dev_focus: 0.25
        - code_quality: 0.20
        - code_completeness: 0.15
        - structure: 0.10
        - accuracy: 0.15
        """
        weights = {
            Dimension.PATTERN_COMPLIANCE: 0.15,
            Dimension.DEV_FOCUS: 0.25,
            Dimension.CODE_QUALITY: 0.20,
            Dimension.CODE_COMPLETENESS: 0.15,
            Dimension.STRUCTURE: 0.10,
            Dimension.ACCURACY: 0.15,
            Dimension.JTBD_ALIGNMENT: 0.0,  # Not included in overall
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for dim_score in self.dimension_scores:
            if dim_score.score is not None and dim_score.dimension in weights:
                weight = weights[dim_score.dimension]
                weighted_sum += dim_score.score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 2)

    @property
    def priority_rank(self) -> int:
        """Calculate priority rank (1 = highest priority for remediation).

        Lower scores = higher priority.
        Critical issues boost priority.
        """
        base_priority = int((7.0 - self.overall_score) * 10)
        critical_boost = len(self.critical_issues) * 5
        high_boost = len(self.high_issues) * 2
        return base_priority + critical_boost + high_boost

    @property
    def critical_issues(self) -> list[Issue]:
        """Get all critical issues."""
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def high_issues(self) -> list[Issue]:
        """Get all high-severity issues."""
        return [i for i in self.issues if i.severity == Severity.HIGH]

    @property
    def medium_issues(self) -> list[Issue]:
        """Get all medium-severity issues."""
        return [i for i in self.issues if i.severity == Severity.MEDIUM]

    @property
    def low_issues(self) -> list[Issue]:
        """Get all low-severity issues."""
        return [i for i in self.issues if i.severity == Severity.LOW]

    @property
    def top_issues_summary(self) -> str:
        """Get a semicolon-separated summary of top 3 issues."""
        # Prioritize by severity
        sorted_issues = sorted(
            self.issues,
            key=lambda i: (
                list(Severity).index(i.severity),
                i.id,
            ),
        )
        top_3 = sorted_issues[:3]
        return ";".join(i.title for i in top_3)

    def get_dimension_score(self, dimension: Dimension) -> Optional[float]:
        """Get the score for a specific dimension."""
        for dim_score in self.dimension_scores:
            if dim_score.dimension == dimension:
                return dim_score.score
        return None

    def get_dimension_score_by_name(self, name: str) -> Optional[float]:
        """Get the score for a dimension by its string name."""
        for dim_score in self.dimension_scores:
            if dim_score.dimension.value == name:
                return dim_score.score
        return None

    def is_below_threshold(self, threshold: float = 4.0) -> bool:
        """Check if the overall score is below the given threshold."""
        return self.overall_score < threshold


class BatchEvaluationResult(BaseModel):
    """Result of evaluating a batch of articles."""

    # Run metadata
    run_id: str
    run_mode: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    target_path: str

    # Results
    results: list[EvaluationResult] = Field(default_factory=list)

    # Statistics
    total_articles: int = 0
    evaluated_articles: int = 0
    failed_articles: int = 0
    skipped_articles: int = 0

    # Token usage totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def success_rate(self) -> float:
        """Percentage of articles successfully evaluated."""
        if self.total_articles == 0:
            return 0.0
        return (self.evaluated_articles / self.total_articles) * 100

    @property
    def average_score(self) -> float:
        """Average overall score across all evaluated articles."""
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    @property
    def articles_below_threshold(self) -> list[EvaluationResult]:
        """Articles scoring below the default threshold."""
        return [r for r in self.results if r.is_below_threshold()]

    @property
    def duration_seconds(self) -> Optional[float]:
        """Total evaluation duration in seconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds()
