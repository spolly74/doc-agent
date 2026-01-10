"""Code samples validator."""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from foundry_eval.context.samples_index import SamplesIndex
from foundry_eval.context.toc_parser import TOCParser
from foundry_eval.llm.prompts.code_review import (
    CODE_REVIEW_SYSTEM_PROMPT,
    build_orphan_sample_prompt,
    build_sample_validation_prompt,
    build_samples_cross_reference_prompt,
)
from foundry_eval.llm.protocol import LLMMessage, LLMProvider
from foundry_eval.llm.response_parser import extract_json_from_response
from foundry_eval.models.code_sample import (
    CodeSample,
    SampleReference,
    SamplesCoverageReport,
    SampleStatus,
    SampleValidationIssue,
    SampleValidationResult,
)

logger = logging.getLogger("foundry_eval.evaluators.code_samples")


class CodeSampleValidator:
    """Validates code samples and checks documentation coverage."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_concurrency: int = 5,
    ):
        """Initialize the validator.

        Args:
            llm_provider: LLM provider for code review.
            max_concurrency: Maximum concurrent LLM calls.
        """
        self._llm = llm_provider
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def validate_samples(
        self,
        samples_index: SamplesIndex,
        toc_parser: Optional[TOCParser] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> SamplesCoverageReport:
        """Validate all samples and generate a coverage report.

        Args:
            samples_index: Indexed samples repository.
            toc_parser: Optional TOC parser for cross-referencing articles.
            on_progress: Optional progress callback (completed, total, message).

        Returns:
            SamplesCoverageReport with validation results.
        """
        # Ensure index is built
        await samples_index.index()

        samples = samples_index.samples
        total = len(samples)

        if on_progress:
            on_progress(0, total, "Starting sample validation...")

        # Validate each sample
        validation_results = []
        for i, sample in enumerate(samples):
            if on_progress:
                on_progress(i, total, f"Validating: {sample.sample_id}")

            result = await self._validate_sample(sample)
            validation_results.append(result)

        # Get orphaned samples
        orphaned = await samples_index.get_orphaned_samples()

        # Find articles needing samples (if TOC parser provided)
        articles_missing = []
        articles_with_samples = 0
        total_articles = 0

        if toc_parser:
            articles = await toc_parser.load_all_articles()
            total_articles = len(articles)

            for article in articles:
                # Check if article has code blocks
                if article.code_blocks:
                    articles_with_samples += 1
                elif self._article_needs_samples(article):
                    articles_missing.append(article.relative_path)

        # Build report
        valid_count = sum(1 for r in validation_results if r.is_valid)

        return SamplesCoverageReport(
            total_samples=len(samples),
            valid_samples=valid_count,
            invalid_samples=len(samples) - valid_count,
            orphaned_samples=len(orphaned),
            total_articles=total_articles,
            articles_with_samples=articles_with_samples,
            articles_needing_samples=len(articles_missing) + articles_with_samples,
            validation_results=validation_results,
            orphaned_sample_paths=[s.file_path for s in orphaned],
            articles_missing_samples=articles_missing,
            broken_references=[r for r in samples_index.references if not r.is_valid],
        )

    async def validate_single_sample(self, sample: CodeSample) -> SampleValidationResult:
        """Validate a single code sample.

        Args:
            sample: The sample to validate.

        Returns:
            SampleValidationResult.
        """
        return await self._validate_sample(sample)

    async def analyze_orphaned_samples(
        self,
        samples_index: SamplesIndex,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[dict]:
        """Analyze orphaned samples and provide recommendations.

        Args:
            samples_index: Indexed samples repository.
            on_progress: Optional progress callback.

        Returns:
            List of analysis results with recommendations.
        """
        await samples_index.index()
        orphaned = await samples_index.get_orphaned_samples()

        results = []
        for i, sample in enumerate(orphaned):
            if on_progress:
                on_progress(i, len(orphaned), f"Analyzing: {sample.sample_id}")

            analysis = await self._analyze_orphan(sample)
            results.append(analysis)

        return results

    async def find_articles_needing_samples(
        self,
        toc_parser: TOCParser,
        samples_index: SamplesIndex,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[dict]:
        """Find articles that should have code samples but don't.

        Args:
            toc_parser: TOC parser with loaded articles.
            samples_index: Indexed samples for matching.
            on_progress: Optional progress callback.

        Returns:
            List of articles with sample recommendations.
        """
        await samples_index.index()
        articles = await toc_parser.load_all_articles()

        results = []
        for i, article in enumerate(articles):
            if on_progress:
                on_progress(i, len(articles), f"Checking: {article.relative_path}")

            # Skip articles that already have samples
            if article.code_blocks:
                continue

            # Check if article needs samples
            if not self._article_needs_samples(article):
                continue

            # Find relevant samples
            recommendations = await self._recommend_samples(
                article, samples_index.samples
            )
            if recommendations:
                results.append({
                    "article_path": article.relative_path,
                    "article_title": article.title,
                    "recommendations": recommendations,
                })

        return results

    async def _validate_sample(self, sample: CodeSample) -> SampleValidationResult:
        """Validate a single sample using LLM.

        Args:
            sample: The sample to validate.

        Returns:
            SampleValidationResult.
        """
        async with self._semaphore:
            prompt = build_sample_validation_prompt(
                sample_content=sample.content,
                language=sample.language.value,
            )

            messages = [
                LLMMessage(role="system", content=CODE_REVIEW_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            try:
                response = await self._llm.complete(messages)
                data = extract_json_from_response(response.content)

                if data:
                    issues = [
                        SampleValidationIssue(
                            sample_id=sample.sample_id,
                            issue_type=i.get("issue_type", "unknown"),
                            severity=i.get("severity", "low"),
                            title=i.get("title", ""),
                            description=i.get("description", ""),
                            line_number=i.get("line_number"),
                            suggestion=i.get("suggestion", ""),
                            auto_fixable=i.get("auto_fixable", False),
                        )
                        for i in data.get("issues", [])
                    ]

                    status = SampleStatus.VALID if data.get("status") == "valid" else SampleStatus.INVALID
                    analysis = data.get("analysis", {})

                    return SampleValidationResult(
                        sample=sample,
                        status=status,
                        issues=issues,
                        completeness_score=data.get("completeness_score", 0.5),
                        correctness_score=data.get("correctness_score", 0.5),
                        best_practices_score=data.get("best_practices_score", 0.5),
                        has_error_handling=analysis.get("has_error_handling", False),
                        has_comments=analysis.get("has_comments", False),
                        is_runnable=analysis.get("is_runnable", False),
                    )

            except Exception as e:
                logger.error(f"Failed to validate sample {sample.sample_id}: {e}")

        # Return minimal result on failure
        return SampleValidationResult(
            sample=sample,
            status=SampleStatus.INVALID,
            issues=[
                SampleValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="validation_error",
                    severity="high",
                    title="Validation Failed",
                    description="Failed to validate sample",
                )
            ],
        )

    async def _analyze_orphan(self, sample: CodeSample) -> dict:
        """Analyze an orphaned sample.

        Args:
            sample: The orphaned sample.

        Returns:
            Analysis dictionary with recommendations.
        """
        async with self._semaphore:
            prompt = build_orphan_sample_prompt(
                sample_path=sample.file_path,
                sample_content=sample.content,
                sample_language=sample.language.value,
            )

            messages = [
                LLMMessage(role="system", content=CODE_REVIEW_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            try:
                response = await self._llm.complete(messages)
                data = extract_json_from_response(response.content)

                if data:
                    return {
                        "sample_path": sample.file_path,
                        "sample_id": sample.sample_id,
                        **data,
                    }

            except Exception as e:
                logger.error(f"Failed to analyze orphan {sample.sample_id}: {e}")

        return {
            "sample_path": sample.file_path,
            "sample_id": sample.sample_id,
            "recommendation": "review",
            "quality_assessment": "unknown",
            "error": "Analysis failed",
        }

    async def _recommend_samples(
        self,
        article,
        available_samples: list[CodeSample],
    ) -> list[dict]:
        """Recommend samples for an article.

        Args:
            article: The article needing samples.
            available_samples: List of available samples.

        Returns:
            List of sample recommendations.
        """
        # Simple keyword matching for now
        # Could be enhanced with LLM-based matching
        recommendations = []
        article_content = article.content.lower() if article.content else ""

        for sample in available_samples:
            # Check for keyword matches
            sample_keywords = sample.title.lower().split() + sample.tags
            match_score = sum(
                1 for kw in sample_keywords
                if kw in article_content and len(kw) > 3
            )

            if match_score > 0:
                recommendations.append({
                    "sample_path": sample.file_path,
                    "sample_title": sample.title,
                    "match_score": match_score,
                    "language": sample.language.value,
                })

        # Sort by match score and return top 5
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        return recommendations[:5]

    def _article_needs_samples(self, article) -> bool:
        """Check if an article should have code samples.

        Args:
            article: The article to check.

        Returns:
            True if the article should have samples.
        """
        if not article.content:
            return False

        content_lower = article.content.lower()

        # Keywords indicating code should be present
        code_indicators = [
            "how to",
            "example",
            "sample",
            "code",
            "implement",
            "create",
            "build",
            "develop",
            "api",
            "sdk",
            "quickstart",
            "tutorial",
            "walkthrough",
        ]

        return any(indicator in content_lower for indicator in code_indicators)
