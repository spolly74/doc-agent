"""Pydantic configuration models for the Foundry evaluation system."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMProviderType(str, Enum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    # Future providers can be added here
    # OPENAI = "openai"
    # AZURE = "azure"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: LLMProviderType = LLMProviderType.CLAUDE
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_concurrent_requests: int = Field(default=5, ge=1, le=50)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)


class ScoringWeights(BaseModel):
    """Weights for calculating the overall score from dimension scores."""

    pattern_compliance: float = Field(default=0.15, ge=0.0, le=1.0)
    dev_focus: float = Field(default=0.25, ge=0.0, le=1.0)
    code_quality: float = Field(default=0.20, ge=0.0, le=1.0)
    code_completeness: float = Field(default=0.15, ge=0.0, le=1.0)
    structure: float = Field(default=0.10, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.15, ge=0.0, le=1.0)

    def validate_sum(self) -> bool:
        """Check that weights sum to 1.0 (within floating point tolerance)."""
        total = (
            self.pattern_compliance
            + self.dev_focus
            + self.code_quality
            + self.code_completeness
            + self.structure
            + self.accuracy
        )
        return abs(total - 1.0) < 0.001


class EvaluationConfig(BaseModel):
    """Evaluation behavior configuration."""

    dimensions: list[str] = Field(
        default=[
            "pattern_compliance",
            "dev_focus",
            "code_quality",
            "code_completeness",
            "structure",
            "accuracy",
            "jtbd_alignment",
        ]
    )
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    threshold: float = Field(default=4.0, ge=1.0, le=7.0)
    quick_scan_threshold: float = Field(default=3.5, ge=1.0, le=7.0)
    max_age_months: int = Field(default=12, ge=1, le=60)
    include_patterns: list[str] = Field(default=["**/*.md"])
    exclude_patterns: list[str] = Field(
        default=["**/includes/**", "**/snippets/**", "**/media/**"]
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    csv_filename: str = "scores.csv"
    report_filename: str = "report.md"
    include_raw_responses: bool = False
    verbosity: int = Field(default=1, ge=0, le=3)
    gitignore_output: bool = True


class FoundryEvalConfig(BaseModel):
    """Root configuration model for the evaluation system."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Paths (optional, can be overridden by CLI)
    articles_path: Optional[Path] = None
    samples_repo_path: Optional[Path] = None
    jtbd_data_path: Optional[Path] = None
    output_path: Optional[Path] = None

    # State management
    state_db_path: Optional[str] = ".foundry-eval-state.db"

    model_config = ConfigDict(extra="ignore")  # Ignore unknown fields in config file
