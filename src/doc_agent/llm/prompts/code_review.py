"""Prompts for code sample validation and review."""

CODE_REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer specializing in developer documentation and code samples.

Your task is to evaluate code samples for:
1. Correctness - Does the code work and follow best practices?
2. Completeness - Is the sample self-contained or missing key parts?
3. Clarity - Is it easy to understand and follow?
4. Documentation - Are there appropriate comments and explanations?

Focus on practical issues that would affect a developer trying to use this code.
Be specific about any problems and provide actionable suggestions."""


def build_sample_validation_prompt(
    sample_content: str,
    language: str,
    context: str = "",
) -> str:
    """Build a prompt for validating a code sample.

    Args:
        sample_content: The code sample content.
        language: Programming language of the sample.
        context: Optional context about where/how the sample is used.

    Returns:
        Formatted prompt string.
    """
    context_section = ""
    if context:
        context_section = f"""## Context

{context}

"""

    return f"""{context_section}## Code Sample ({language})

```{language}
{sample_content}
```

## Your Task

Evaluate this code sample for quality and correctness. Respond with JSON:

```json
{{
    "status": "valid|invalid",
    "completeness_score": 0.8,
    "correctness_score": 0.9,
    "best_practices_score": 0.7,
    "issues": [
        {{
            "issue_type": "syntax|deprecated_api|missing_import|incomplete|security|style",
            "severity": "critical|high|medium|low",
            "title": "Short issue title",
            "description": "Detailed description of the issue",
            "line_number": 5,
            "suggestion": "How to fix this issue",
            "auto_fixable": false
        }}
    ],
    "analysis": {{
        "has_error_handling": true,
        "has_comments": false,
        "is_runnable": true,
        "uses_deprecated_apis": false,
        "missing_imports": ["list", "of", "imports"],
        "security_concerns": []
    }},
    "improvement_suggestions": [
        "Add error handling for edge case X",
        "Include import statement for Y"
    ]
}}
```

Be thorough but fair. Minor style issues should be low severity. Focus on issues that would prevent the code from working or confuse developers.
"""


def build_sample_completeness_prompt(
    sample_content: str,
    language: str,
    article_context: str,
) -> str:
    """Build a prompt to evaluate if a sample is complete for its context.

    Args:
        sample_content: The code sample content.
        language: Programming language.
        article_context: The surrounding article text for context.

    Returns:
        Formatted prompt string.
    """
    return f"""## Article Context

{article_context[:3000]}

## Code Sample ({language})

```{language}
{sample_content}
```

## Your Task

Evaluate if this code sample is complete enough for the article's purpose. Consider:
1. Does it demonstrate what the article claims?
2. Can a developer copy and run this code?
3. Are there missing setup steps or dependencies?

Respond with JSON:

```json
{{
    "is_complete": true,
    "completeness_score": 0.8,
    "missing_elements": [
        {{
            "element": "import statements",
            "importance": "high",
            "suggestion": "Add: import foo from 'bar'"
        }}
    ],
    "assumptions_made": [
        "Assumes developer has SDK installed",
        "Assumes API key is configured"
    ],
    "setup_requirements": [
        "Install package X",
        "Configure environment variable Y"
    ],
    "would_run_as_is": false,
    "improvements_needed": [
        "Add import statements at the top",
        "Include error handling for API calls"
    ]
}}
```
"""


def build_samples_cross_reference_prompt(
    article_path: str,
    article_content: str,
    available_samples: list[dict],
) -> str:
    """Build a prompt to analyze if an article needs samples or has broken references.

    Args:
        article_path: Path to the article.
        article_content: Full article content.
        available_samples: List of available samples with paths and descriptions.

    Returns:
        Formatted prompt string.
    """
    samples_list = "\n".join([
        f"- {s['path']}: {s.get('description', 'No description')[:100]}"
        for s in available_samples[:50]  # Limit to avoid token overflow
    ])

    return f"""## Article: {article_path}

{article_content[:6000]}

## Available Code Samples

{samples_list}

## Your Task

Analyze this article's code sample needs:
1. Does this article need code samples that it doesn't have?
2. Are there any broken or invalid sample references?
3. Which available samples would be relevant?

Respond with JSON:

```json
{{
    "needs_samples": true,
    "current_sample_count": 2,
    "sample_needs": [
        {{
            "topic": "Authentication example",
            "importance": "high",
            "suggested_sample": "path/to/relevant/sample.py"
        }}
    ],
    "broken_references": [
        {{
            "reference": "~/samples/nonexistent.py",
            "line_number": 45,
            "suggested_fix": "Update to ~/samples/existing.py"
        }}
    ],
    "relevant_existing_samples": [
        {{
            "sample_path": "auth/basic-auth.py",
            "relevance_reason": "Demonstrates authentication flow mentioned in article"
        }}
    ],
    "sample_quality_notes": "The existing samples are good but missing error handling examples"
}}
```
"""


def build_orphan_sample_prompt(
    sample_path: str,
    sample_content: str,
    sample_language: str,
) -> str:
    """Build a prompt to analyze an orphaned sample's purpose and potential use.

    Args:
        sample_path: Path to the sample file.
        sample_content: Content of the sample.
        sample_language: Programming language.

    Returns:
        Formatted prompt string.
    """
    return f"""## Orphaned Sample: {sample_path}

This code sample is not referenced by any documentation article.

```{sample_language}
{sample_content[:4000]}
```

## Your Task

Analyze this orphaned sample and determine:
1. What is its purpose?
2. Should it be deleted or documented?
3. What kind of article would use it?

Respond with JSON:

```json
{{
    "purpose": "Brief description of what this sample demonstrates",
    "recommendation": "document|delete|update",
    "quality_assessment": "good|needs_work|poor",
    "suggested_article_topics": [
        "Topic 1 that could use this sample",
        "Topic 2"
    ],
    "issues_if_any": [
        "Sample uses deprecated API",
        "Missing error handling"
    ],
    "deletion_reason": "Only if recommendation is 'delete': explain why"
}}
```
"""
