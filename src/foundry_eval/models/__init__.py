"""Domain models for the Foundry evaluation system."""

from foundry_eval.models.article import Article, ArticleMetadata, CodeBlock
from foundry_eval.models.code_sample import (
    CodeSample,
    SampleReference,
    SamplesCoverageReport,
    SampleStatus,
    SampleValidationResult,
)
from foundry_eval.models.enums import Dimension, RunMode, Severity
from foundry_eval.models.evaluation import DimensionScore, EvaluationResult, Issue
from foundry_eval.models.jtbd import (
    CoverageGap,
    JTBD,
    JTBDAnalysisResult,
    JTBDCoverageStatus,
    JTBDMapping,
    JTBDStep,
)

__all__ = [
    "Article",
    "ArticleMetadata",
    "CodeBlock",
    "CodeSample",
    "CoverageGap",
    "Dimension",
    "DimensionScore",
    "EvaluationResult",
    "Issue",
    "JTBD",
    "JTBDAnalysisResult",
    "JTBDCoverageStatus",
    "JTBDMapping",
    "JTBDStep",
    "RunMode",
    "SampleReference",
    "SamplesCoverageReport",
    "SampleStatus",
    "SampleValidationResult",
    "Severity",
]
