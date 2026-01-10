"""Prompt templates for LLM evaluations."""

from foundry_eval.llm.prompts.article_eval import (
    ARTICLE_EVALUATION_SYSTEM_PROMPT,
    build_article_evaluation_prompt,
    build_quick_scan_prompt,
)
from foundry_eval.llm.prompts.code_review import (
    CODE_REVIEW_SYSTEM_PROMPT,
    build_orphan_sample_prompt,
    build_sample_completeness_prompt,
    build_sample_validation_prompt,
    build_samples_cross_reference_prompt,
)
from foundry_eval.llm.prompts.jtbd_analysis import (
    JTBD_ANALYSIS_SYSTEM_PROMPT,
    build_article_jtbd_relevance_prompt,
    build_gap_analysis_prompt,
    build_jtbd_mapping_prompt,
    build_jtbd_title_mapping_prompt,
    build_stepless_gap_analysis_prompt,
)

__all__ = [
    # Article evaluation
    "ARTICLE_EVALUATION_SYSTEM_PROMPT",
    "build_article_evaluation_prompt",
    "build_quick_scan_prompt",
    # Code review
    "CODE_REVIEW_SYSTEM_PROMPT",
    "build_orphan_sample_prompt",
    "build_sample_completeness_prompt",
    "build_sample_validation_prompt",
    "build_samples_cross_reference_prompt",
    # JTBD analysis
    "JTBD_ANALYSIS_SYSTEM_PROMPT",
    "build_article_jtbd_relevance_prompt",
    "build_gap_analysis_prompt",
    "build_jtbd_mapping_prompt",
    "build_jtbd_title_mapping_prompt",
    "build_stepless_gap_analysis_prompt",
]
