"""Prompt templates for LLM evaluations."""

from foundry_eval.llm.prompts.article_eval import (
    build_article_evaluation_prompt,
    build_quick_scan_prompt,
    ARTICLE_EVALUATION_SYSTEM_PROMPT,
)

__all__ = [
    "ARTICLE_EVALUATION_SYSTEM_PROMPT",
    "build_article_evaluation_prompt",
    "build_quick_scan_prompt",
]
