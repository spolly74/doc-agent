"""Domain models for the Doc Agent evaluation system."""

from doc_agent.models.article import Article, ArticleMetadata, CodeBlock
from doc_agent.models.code_sample import (
    CodeSample,
    SampleReference,
    SamplesCoverageReport,
    SampleStatus,
    SampleValidationResult,
)
from doc_agent.models.enums import Dimension, RunMode, Severity
from doc_agent.models.evaluation import DimensionScore, EvaluationResult, Issue
from doc_agent.models.jtbd import (
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
