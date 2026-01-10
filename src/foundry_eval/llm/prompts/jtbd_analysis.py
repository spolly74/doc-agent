"""Prompts for JTBD coverage analysis."""

JTBD_ANALYSIS_SYSTEM_PROMPT = """You are an expert technical documentation analyst specializing in developer journeys and Jobs-To-Be-Done (JTBD) analysis.

Your task is to analyze how well existing documentation covers a specific developer journey (JTBD) by:
1. Mapping existing articles to journey steps
2. Identifying coverage gaps
3. Recommending content improvements

For each JTBD step, evaluate:
- Does existing documentation fully cover this step?
- What aspects are covered vs missing?
- What additional content would help developers complete this step?

Be specific and actionable in your recommendations. Focus on practical developer needs."""


def build_jtbd_mapping_prompt(jtbd: dict, articles: list[dict]) -> str:
    """Build a prompt for mapping articles to JTBD steps.

    Args:
        jtbd: JTBD definition with steps.
        articles: List of article summaries (path, title, content snippet).

    Returns:
        Formatted prompt string.
    """
    # Format JTBD
    jtbd_section = f"""## Job-To-Be-Done: {jtbd['title']}

**ID:** {jtbd['jtbd_id']}
**Persona:** {jtbd.get('persona', 'Developer')}
**Description:** {jtbd['description']}

### Steps in the Journey:
"""
    for step in jtbd.get('steps', []):
        jtbd_section += f"""
**Step {step['step_number']}: {step['title']}**
- ID: {step['step_id']}
- Description: {step['description']}
"""

    # Format articles
    articles_section = "## Available Documentation Articles:\n\n"
    for i, article in enumerate(articles, 1):
        articles_section += f"""### Article {i}: {article.get('title', 'Untitled')}
- Path: {article['path']}
- Content Preview: {article.get('content', '')[:500]}...

"""

    return f"""{jtbd_section}

{articles_section}

## Your Task

For each step in the JTBD, identify which articles (if any) cover that step. Respond with a JSON object in this format:

```json
{{
    "mappings": [
        {{
            "step_id": "step-1",
            "step_title": "Title of step",
            "coverage_status": "fully_covered|partially_covered|not_covered",
            "mapped_articles": [
                {{
                    "article_path": "path/to/article.md",
                    "relevance_score": 0.8,
                    "covers_fully": false,
                    "covered_aspects": ["aspect 1", "aspect 2"],
                    "missing_aspects": ["aspect 3"]
                }}
            ],
            "coverage_notes": "Brief explanation of coverage status"
        }}
    ],
    "overall_coverage_score": 0.65,
    "summary": "Overall assessment of documentation coverage for this JTBD"
}}
```

Be thorough but realistic in your assessment. Only mark a step as "fully_covered" if the documentation truly addresses all aspects a developer would need.
"""


def build_gap_analysis_prompt(jtbd: dict, mappings: list[dict]) -> str:
    """Build a prompt for analyzing coverage gaps and recommending improvements.

    Args:
        jtbd: JTBD definition.
        mappings: List of step-to-article mappings with coverage status.

    Returns:
        Formatted prompt string.
    """
    # Format JTBD and mappings
    context = f"""## Job-To-Be-Done: {jtbd['title']}

**Description:** {jtbd['description']}

### Current Coverage Status:
"""
    for mapping in mappings:
        context += f"""
**Step {mapping.get('step_number', '?')}: {mapping['step_title']}**
- Status: {mapping['coverage_status']}
- Mapped Articles: {len(mapping.get('mapped_articles', []))}
- Notes: {mapping.get('coverage_notes', 'N/A')}
"""

    return f"""{context}

## Your Task

Analyze the gaps in documentation coverage and provide actionable recommendations. For each gap, suggest:
1. What new content should be created
2. What existing content could be expanded
3. Priority level (critical/high/medium/low)

Respond with a JSON object:

```json
{{
    "gaps": [
        {{
            "step_id": "step-3",
            "gap_type": "missing|incomplete|outdated",
            "severity": "critical|high|medium|low",
            "title": "Short title for the gap",
            "description": "Detailed description of what's missing",
            "suggested_article_title": "Proposed title for new content",
            "suggested_content_outline": [
                "Section 1: ...",
                "Section 2: ..."
            ],
            "related_articles": ["path/to/related.md"],
            "estimated_effort": "small|medium|large"
        }}
    ],
    "priority_recommendations": [
        "1. Most important action to take",
        "2. Second priority",
        "3. Third priority"
    ],
    "new_articles_needed": [
        {{
            "title": "Suggested Article Title",
            "covering_steps": ["step-2", "step-3"],
            "outline": ["Section 1", "Section 2"]
        }}
    ]
}}
```

Focus on gaps that would have the highest impact on developer success.
"""


def build_article_jtbd_relevance_prompt(article_content: str, jtbd: dict) -> str:
    """Build a prompt to assess how relevant an article is to a JTBD.

    Args:
        article_content: Full article content.
        jtbd: JTBD definition.

    Returns:
        Formatted prompt string.
    """
    steps_list = "\n".join([
        f"- Step {s['step_number']}: {s['title']} - {s['description'][:100]}..."
        for s in jtbd.get('steps', [])
    ])

    return f"""## Article Content

{article_content[:8000]}

## JTBD: {jtbd['title']}

**Description:** {jtbd['description']}

**Steps:**
{steps_list}

## Your Task

Analyze how this article supports developers completing this JTBD. Respond with JSON:

```json
{{
    "relevance_score": 0.7,
    "relevant_steps": [
        {{
            "step_id": "step-1",
            "relevance": "high|medium|low",
            "covered_aspects": ["aspect 1", "aspect 2"],
            "missing_aspects": ["aspect 3"]
        }}
    ],
    "article_quality_for_jtbd": {{
        "provides_context": true,
        "has_actionable_steps": true,
        "includes_code_examples": false,
        "addresses_common_errors": false
    }},
    "improvement_suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ]
}}
```
"""
