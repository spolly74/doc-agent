"""JTBD coverage analyzer."""

import asyncio
import json
import logging
from typing import Callable, Optional

from foundry_eval.context.jtbd_loader import JTBDLoader
from foundry_eval.context.toc_parser import TOCParser
from foundry_eval.llm.prompts.jtbd_analysis import (
    JTBD_ANALYSIS_SYSTEM_PROMPT,
    build_gap_analysis_prompt,
    build_jtbd_mapping_prompt,
    build_jtbd_title_mapping_prompt,
    build_stepless_gap_analysis_prompt,
)
from foundry_eval.llm.protocol import LLMMessage, LLMProvider
from foundry_eval.llm.response_parser import extract_json_from_response
from foundry_eval.models.jtbd import (
    CoverageGap,
    JTBD,
    JTBDAnalysisResult,
    JTBDCoverageStatus,
    JTBDMapping,
)

logger = logging.getLogger("foundry_eval.evaluators.jtbd")


class JTBDAnalyzer:
    """Analyzes JTBD coverage across documentation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_concurrency: int = 3,
    ):
        """Initialize the JTBD analyzer.

        Args:
            llm_provider: LLM provider for analysis.
            max_concurrency: Maximum concurrent LLM calls.
        """
        self._llm = llm_provider
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_jtbd(
        self,
        jtbd: JTBD,
        toc_parser: TOCParser,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> JTBDAnalysisResult:
        """Analyze coverage for a single JTBD.

        Supports both step-based JTBDs and step-less JTBDs (title-level mapping).

        Args:
            jtbd: The JTBD to analyze.
            toc_parser: TOC parser with loaded articles.
            on_progress: Optional progress callback.

        Returns:
            JTBDAnalysisResult with mappings and gaps.
        """
        if on_progress:
            on_progress(f"Analyzing JTBD: {jtbd.title}")

        # Load all articles
        articles = await toc_parser.load_all_articles()

        # Prepare article summaries for the prompt
        article_summaries = []
        for article in articles:
            article_summaries.append({
                "path": article.relative_path,
                "title": article.title or article.relative_path,
                "content": article.content[:1000] if article.content else "",
            })

        # Handle step-less JTBDs differently
        if jtbd.is_stepless:
            return await self._analyze_stepless_jtbd(
                jtbd, article_summaries, on_progress
            )

        # Step-based analysis
        if on_progress:
            on_progress("Mapping articles to JTBD steps...")

        mappings = await self._map_articles_to_jtbd(jtbd, article_summaries)

        # Analyze gaps
        if on_progress:
            on_progress("Analyzing coverage gaps...")

        gaps = await self._analyze_gaps(jtbd, mappings)

        # Calculate overall coverage
        covered_steps = sum(
            1 for m in mappings
            if m.get("coverage_status") in ("fully_covered", "partially_covered")
        )
        coverage_score = covered_steps / len(jtbd.steps) if jtbd.steps else 0.0

        # Build result
        result = JTBDAnalysisResult(
            jtbd=jtbd,
            mappings=[self._parse_mapping(m, jtbd.jtbd_id) for m in mappings],
            gaps=gaps,
            total_articles_mapped=len(set(
                a["article_path"]
                for m in mappings
                for a in m.get("mapped_articles", [])
            )),
            overall_coverage_score=coverage_score,
        )

        # Update JTBD step coverage
        for mapping in mappings:
            step_id = mapping.get("step_id")
            status = mapping.get("coverage_status", "not_covered")
            for step in result.jtbd.steps:
                if step.step_id == step_id:
                    step.coverage_status = JTBDCoverageStatus(status)
                    step.mapped_articles = [
                        a["article_path"]
                        for a in mapping.get("mapped_articles", [])
                    ]
                    break

        return result

    async def _analyze_stepless_jtbd(
        self,
        jtbd: JTBD,
        article_summaries: list[dict],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> JTBDAnalysisResult:
        """Analyze coverage for a step-less JTBD using title-level mapping.

        Args:
            jtbd: The step-less JTBD to analyze.
            article_summaries: List of article summaries.
            on_progress: Optional progress callback.

        Returns:
            JTBDAnalysisResult with title-level mappings.
        """
        if on_progress:
            on_progress("Finding relevant articles for JTBD...")

        # Map articles to the JTBD title
        mapping_result = await self._map_articles_to_jtbd_title(
            jtbd, article_summaries
        )

        mapped_articles = mapping_result.get("mapped_articles", [])
        coverage_status = mapping_result.get("coverage_status", "not_covered")
        coverage_score = mapping_result.get("coverage_score", 0.0)

        # Update JTBD-level coverage
        jtbd.coverage_status = JTBDCoverageStatus(coverage_status)
        jtbd.coverage_score = coverage_score
        jtbd.mapped_articles = [a["article_path"] for a in mapped_articles]

        # Create a single mapping for the JTBD itself
        mappings = []
        for article in mapped_articles:
            mappings.append(JTBDMapping(
                jtbd_id=jtbd.jtbd_id,
                step_id="",  # No step for step-less JTBD
                article_path=article.get("article_path", ""),
                relevance_score=article.get("relevance_score", 0.0),
                covers_fully=article.get("covers_fully", False),
                covered_aspects=article.get("covered_aspects", []),
                missing_aspects=article.get("missing_aspects", []),
                mapping_notes=article.get("notes", ""),
            ))

        # Analyze gaps for step-less JTBD
        if on_progress:
            on_progress("Analyzing coverage gaps...")

        gaps = await self._analyze_stepless_gaps(jtbd, mapping_result)

        return JTBDAnalysisResult(
            jtbd=jtbd,
            mappings=mappings,
            gaps=gaps,
            total_articles_mapped=len(mapped_articles),
            overall_coverage_score=coverage_score,
        )

    async def analyze_all_jtbds(
        self,
        jtbd_loader: JTBDLoader,
        toc_parser: TOCParser,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[JTBDAnalysisResult]:
        """Analyze coverage for all JTBDs.

        Args:
            jtbd_loader: JTBD data loader.
            toc_parser: TOC parser with loaded articles.
            on_progress: Optional progress callback (completed, total, message).

        Returns:
            List of JTBDAnalysisResult objects.
        """
        jtbds = await jtbd_loader.load_all()
        results = []

        for i, jtbd in enumerate(jtbds):
            if on_progress:
                on_progress(i, len(jtbds), f"Analyzing: {jtbd.title}")

            result = await self.analyze_jtbd(jtbd, toc_parser)
            results.append(result)

        return results

    async def _map_articles_to_jtbd(
        self,
        jtbd: JTBD,
        articles: list[dict],
    ) -> list[dict]:
        """Map articles to JTBD steps using LLM.

        Args:
            jtbd: The JTBD to map.
            articles: List of article summaries.

        Returns:
            List of step mappings.
        """
        async with self._semaphore:
            prompt = build_jtbd_mapping_prompt(
                jtbd=jtbd.model_dump(),
                articles=articles,
            )

            messages = [
                LLMMessage(role="system", content=JTBD_ANALYSIS_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            try:
                response = await self._llm.complete(messages)
                data = extract_json_from_response(response.content)

                if data and "mappings" in data:
                    return data["mappings"]

            except Exception as e:
                logger.error(f"Failed to map articles to JTBD: {e}")

        # Return empty mappings for each step
        return [
            {
                "step_id": step.step_id,
                "step_title": step.title,
                "coverage_status": "not_covered",
                "mapped_articles": [],
                "coverage_notes": "Analysis failed",
            }
            for step in jtbd.steps
        ]

    async def _analyze_gaps(
        self,
        jtbd: JTBD,
        mappings: list[dict],
    ) -> list[CoverageGap]:
        """Analyze coverage gaps using LLM.

        Args:
            jtbd: The JTBD.
            mappings: Step-to-article mappings.

        Returns:
            List of CoverageGap objects.
        """
        async with self._semaphore:
            prompt = build_gap_analysis_prompt(
                jtbd=jtbd.model_dump(),
                mappings=mappings,
            )

            messages = [
                LLMMessage(role="system", content=JTBD_ANALYSIS_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            try:
                response = await self._llm.complete(messages)
                data = extract_json_from_response(response.content)

                if data and "gaps" in data:
                    return [
                        CoverageGap(
                            jtbd_id=jtbd.jtbd_id,
                            step_id=g.get("step_id", ""),
                            gap_type=g.get("gap_type", "missing"),
                            severity=g.get("severity", "medium"),
                            title=g.get("title", ""),
                            description=g.get("description", ""),
                            suggested_article_title=g.get("suggested_article_title"),
                            suggested_content_outline=g.get("suggested_content_outline", []),
                            estimated_effort=g.get("estimated_effort", "medium"),
                            related_articles=g.get("related_articles", []),
                        )
                        for g in data["gaps"]
                    ]

            except Exception as e:
                logger.error(f"Failed to analyze gaps: {e}")

        return []

    async def _map_articles_to_jtbd_title(
        self,
        jtbd: JTBD,
        articles: list[dict],
    ) -> dict:
        """Map articles to a step-less JTBD using title-level matching.

        Args:
            jtbd: The step-less JTBD.
            articles: List of article summaries.

        Returns:
            Dictionary with mapping results.
        """
        async with self._semaphore:
            prompt = build_jtbd_title_mapping_prompt(
                jtbd=jtbd.model_dump(),
                articles=articles,
            )

            messages = [
                LLMMessage(role="system", content=JTBD_ANALYSIS_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            try:
                response = await self._llm.complete(messages)
                data = extract_json_from_response(response.content)

                if data:
                    return data

            except Exception as e:
                logger.error(f"Failed to map articles to JTBD title: {e}")

        return {
            "mapped_articles": [],
            "coverage_status": "not_covered",
            "coverage_score": 0.0,
            "coverage_notes": "Analysis failed",
        }

    async def _analyze_stepless_gaps(
        self,
        jtbd: JTBD,
        mapping_result: dict,
    ) -> list[CoverageGap]:
        """Analyze coverage gaps for a step-less JTBD.

        Args:
            jtbd: The step-less JTBD.
            mapping_result: Result from title-level mapping.

        Returns:
            List of CoverageGap objects.
        """
        async with self._semaphore:
            prompt = build_stepless_gap_analysis_prompt(
                jtbd=jtbd.model_dump(),
                mapping_result=mapping_result,
            )

            messages = [
                LLMMessage(role="system", content=JTBD_ANALYSIS_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            try:
                response = await self._llm.complete(messages)
                data = extract_json_from_response(response.content)

                if data and "gaps" in data:
                    return [
                        CoverageGap(
                            jtbd_id=jtbd.jtbd_id,
                            step_id="",  # No step for step-less JTBD
                            gap_type=g.get("gap_type", "missing"),
                            severity=g.get("severity", "medium"),
                            title=g.get("title", ""),
                            description=g.get("description", ""),
                            suggested_article_title=g.get("suggested_article_title"),
                            suggested_content_outline=g.get(
                                "suggested_content_outline", []
                            ),
                            estimated_effort=g.get("estimated_effort", "medium"),
                            related_articles=g.get("related_articles", []),
                        )
                        for g in data["gaps"]
                    ]

            except Exception as e:
                logger.error(f"Failed to analyze stepless gaps: {e}")

        return []

    def _parse_mapping(self, mapping_data: dict, jtbd_id: str) -> JTBDMapping:
        """Parse a mapping dictionary into JTBDMapping object.

        Args:
            mapping_data: Raw mapping data.
            jtbd_id: The JTBD ID.

        Returns:
            JTBDMapping object.
        """
        # Get the first mapped article if any
        mapped_articles = mapping_data.get("mapped_articles", [])
        if mapped_articles:
            first = mapped_articles[0]
            return JTBDMapping(
                jtbd_id=jtbd_id,
                step_id=mapping_data.get("step_id", ""),
                article_path=first.get("article_path", ""),
                relevance_score=first.get("relevance_score", 0.0),
                covers_fully=first.get("covers_fully", False),
                covered_aspects=first.get("covered_aspects", []),
                missing_aspects=first.get("missing_aspects", []),
                mapping_notes=mapping_data.get("coverage_notes", ""),
            )

        return JTBDMapping(
            jtbd_id=jtbd_id,
            step_id=mapping_data.get("step_id", ""),
            article_path="",
            mapping_notes="No articles mapped",
        )
