"""Parse and validate LLM responses for evaluation results."""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from foundry_eval.models.enums import Dimension, Severity
from foundry_eval.models.evaluation import DimensionScore, EvaluationResult, Issue

logger = logging.getLogger("foundry_eval.llm.response_parser")


class RawDimensionScore(BaseModel):
    """Raw dimension score from LLM response."""

    dimension: str
    score: Optional[float] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class RawIssue(BaseModel):
    """Raw issue from LLM response."""

    dimension: str
    severity: str
    title: str
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


class RawEvaluationResponse(BaseModel):
    """Raw evaluation response from LLM."""

    dimension_scores: list[RawDimensionScore]
    issues: list[RawIssue] = Field(default_factory=list)
    summary: str = ""


class RawQuickScanResponse(BaseModel):
    """Raw quick scan response from LLM."""

    quick_score: float
    needs_deep_evaluation: bool
    flags: list[str] = Field(default_factory=list)
    pattern_match: bool = True
    structure_valid: bool = True
    has_code: bool = False
    is_recent: bool = True
    has_prerequisites: bool = False


def extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response that may contain markdown.

    Args:
        content: Raw LLM response content.

    Returns:
        Extracted JSON string.
    """
    content = content.strip()

    # Try to find JSON in code blocks
    json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(json_block_pattern, content)
    if matches:
        return matches[0].strip()

    # If no code block, try to find JSON object directly
    # Look for content starting with { and ending with }
    if content.startswith("{"):
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(content):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return content[: i + 1]

    return content


def parse_evaluation_response(
    content: str,
    article_path: str,
    article_title: str,
    model_used: str,
) -> EvaluationResult:
    """Parse an LLM evaluation response into an EvaluationResult.

    Args:
        content: Raw LLM response content.
        article_path: Path of the evaluated article.
        article_title: Title of the evaluated article.
        model_used: LLM model that generated the response.

    Returns:
        Parsed EvaluationResult.

    Raises:
        ValueError: If response cannot be parsed.
    """
    # Extract JSON
    json_str = extract_json_from_response(content)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Content was: {content[:500]}")
        raise ValueError(f"Invalid JSON in LLM response: {e}")

    # Parse into raw model
    try:
        raw = RawEvaluationResponse.model_validate(data)
    except ValidationError as e:
        logger.error(f"Failed to validate response structure: {e}")
        raise ValueError(f"Invalid response structure: {e}")

    # Convert to typed models
    dimension_scores = []
    for raw_score in raw.dimension_scores:
        try:
            dimension = Dimension(raw_score.dimension)
        except ValueError:
            logger.warning(f"Unknown dimension: {raw_score.dimension}")
            continue

        dimension_scores.append(
            DimensionScore(
                dimension=dimension,
                score=raw_score.score,
                confidence=raw_score.confidence,
                rationale=raw_score.rationale,
            )
        )

    # Convert issues
    issues = []
    for i, raw_issue in enumerate(raw.issues):
        try:
            dimension = Dimension(raw_issue.dimension)
        except ValueError:
            logger.warning(f"Unknown dimension in issue: {raw_issue.dimension}")
            dimension = Dimension.STRUCTURE  # Default fallback

        try:
            severity = Severity(raw_issue.severity.lower())
        except ValueError:
            logger.warning(f"Unknown severity: {raw_issue.severity}")
            severity = Severity.MEDIUM  # Default fallback

        issues.append(
            Issue(
                id=i + 1,
                dimension=dimension,
                severity=severity,
                title=raw_issue.title,
                description=raw_issue.description,
                location=raw_issue.location,
                suggestion=raw_issue.suggestion,
            )
        )

    return EvaluationResult(
        article_path=article_path,
        article_title=article_title,
        dimension_scores=dimension_scores,
        issues=issues,
        llm_model_used=model_used,
    )


def parse_quick_scan_response(content: str) -> RawQuickScanResponse:
    """Parse a quick scan response from LLM.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed RawQuickScanResponse.

    Raises:
        ValueError: If response cannot be parsed.
    """
    json_str = extract_json_from_response(content)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise ValueError(f"Invalid JSON in LLM response: {e}")

    try:
        return RawQuickScanResponse.model_validate(data)
    except ValidationError as e:
        logger.error(f"Failed to validate quick scan response: {e}")
        raise ValueError(f"Invalid quick scan response: {e}")


def quick_scan_to_evaluation_result(
    scan: RawQuickScanResponse,
    article_path: str,
    article_title: str,
    model_used: str,
) -> EvaluationResult:
    """Convert a quick scan response to a minimal EvaluationResult.

    Args:
        scan: Parsed quick scan response.
        article_path: Path of the evaluated article.
        article_title: Title of the evaluated article.
        model_used: LLM model that generated the response.

    Returns:
        Minimal EvaluationResult with quick scan data.
    """
    # Create dimension scores based on quick scan flags
    dimension_scores = []

    # Estimate structure score
    structure_score = 5.0 if scan.structure_valid else 3.0
    dimension_scores.append(
        DimensionScore(
            dimension=Dimension.STRUCTURE,
            score=structure_score,
            confidence=0.6,  # Lower confidence for quick scan
            rationale="Quick scan structural assessment",
        )
    )

    # Estimate pattern compliance
    pattern_score = 5.0 if scan.pattern_match else 3.0
    dimension_scores.append(
        DimensionScore(
            dimension=Dimension.PATTERN_COMPLIANCE,
            score=pattern_score,
            confidence=0.6,
            rationale="Quick scan pattern match assessment",
        )
    )

    # Convert flags to issues
    issues = []
    for i, flag in enumerate(scan.flags):
        issues.append(
            Issue(
                id=i + 1,
                dimension=Dimension.STRUCTURE,  # Default
                severity=Severity.MEDIUM,
                title=flag[:50] + "..." if len(flag) > 50 else flag,
                description=flag,
                location="general",
            )
        )

    return EvaluationResult(
        article_path=article_path,
        article_title=article_title,
        dimension_scores=dimension_scores,
        issues=issues,
        is_quick_scan=True,
        needs_deep_evaluation=scan.needs_deep_evaluation,
        llm_model_used=model_used,
    )
