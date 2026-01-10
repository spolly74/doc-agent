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


def build_jtbd_title_mapping_prompt(jtbd: dict, articles: list[dict]) -> str:
    """Build a prompt for mapping articles to a step-less JTBD by title.

    This is used for JTBDs that don't have defined steps, such as those
    from project tracking systems (Golden Path format).

    Args:
        jtbd: JTBD definition (without steps).
        articles: List of article summaries (path, title, content snippet).

    Returns:
        Formatted prompt string.
    """
    # Build metadata section
    metadata = []
    if jtbd.get("persona"):
        metadata.append(f"**Persona:** {jtbd['persona']}")
    if jtbd.get("phase"):
        metadata.append(f"**Phase:** {jtbd['phase']}")
    if jtbd.get("area"):
        metadata.append(f"**Area:** {jtbd['area']}")
    if jtbd.get("category"):
        metadata.append(f"**Category:** {jtbd['category']}")

    metadata_section = "\n".join(metadata) if metadata else "No additional metadata"

    # Format articles
    articles_section = "## Available Documentation Articles:\n\n"
    for i, article in enumerate(articles, 1):
        articles_section += f"""### Article {i}: {article.get('title', 'Untitled')}
- Path: {article['path']}
- Content Preview: {article.get('content', '')[:500]}...

"""

    return f"""## Job-To-Be-Done: {jtbd['title']}

**ID:** {jtbd['jtbd_id']}
**Description:** {jtbd.get('description', jtbd['title'])}

{metadata_section}

{articles_section}

## Your Task

This JTBD does not have predefined steps. Analyze the JTBD title and description to find articles that would help a developer accomplish this job.

Consider:
1. Which articles directly address the task described in the title?
2. Which articles provide supporting information or prerequisites?
3. Which articles cover related concepts that would be helpful?

Respond with a JSON object:

```json
{{
    "mapped_articles": [
        {{
            "article_path": "path/to/article.md",
            "article_title": "Article Title",
            "relevance_score": 0.85,
            "covers_fully": false,
            "covered_aspects": ["Setting up the environment", "Basic configuration"],
            "missing_aspects": ["Advanced scenarios", "Troubleshooting"],
            "notes": "Brief explanation of how this article helps with the JTBD"
        }}
    ],
    "coverage_status": "fully_covered|partially_covered|not_covered",
    "coverage_score": 0.65,
    "coverage_notes": "Overall assessment of how well documentation covers this JTBD",
    "suggested_reading_order": ["path/to/first.md", "path/to/second.md"],
    "key_topics_found": ["topic 1", "topic 2"],
    "key_topics_missing": ["missing topic 1", "missing topic 2"]
}}
```

Include all articles with relevance_score > 0.3 in the mapped_articles list. Order by relevance score descending.
"""


def build_stepless_gap_analysis_prompt(jtbd: dict, mapping_result: dict) -> str:
    """Build a prompt for analyzing coverage gaps in a step-less JTBD.

    Args:
        jtbd: JTBD definition (without steps).
        mapping_result: Result from title-level mapping.

    Returns:
        Formatted prompt string.
    """
    # Format mapped articles
    mapped_articles = mapping_result.get("mapped_articles", [])
    articles_section = ""
    if mapped_articles:
        articles_section = "### Mapped Articles:\n"
        for article in mapped_articles:
            articles_section += f"""
- **{article.get('article_title', article.get('article_path', 'Unknown'))}**
  - Relevance: {article.get('relevance_score', 0):.0%}
  - Covers: {', '.join(article.get('covered_aspects', ['N/A']))}
  - Missing: {', '.join(article.get('missing_aspects', ['N/A']))}
"""
    else:
        articles_section = "### No articles mapped to this JTBD.\n"

    return f"""## Job-To-Be-Done: {jtbd['title']}

**ID:** {jtbd['jtbd_id']}
**Description:** {jtbd.get('description', jtbd['title'])}
**Phase:** {jtbd.get('phase', 'N/A')}
**Area:** {jtbd.get('area', 'N/A')}

### Current Coverage:
- Status: {mapping_result.get('coverage_status', 'unknown')}
- Score: {mapping_result.get('coverage_score', 0):.0%}
- Notes: {mapping_result.get('coverage_notes', 'N/A')}

{articles_section}

### Topics Found: {', '.join(mapping_result.get('key_topics_found', ['None']))}
### Topics Missing: {', '.join(mapping_result.get('key_topics_missing', ['None']))}

## Your Task

Based on the JTBD title/description and the current documentation coverage, identify gaps that should be addressed. For this step-less JTBD, think about:

1. What information would a developer need to complete this job?
2. What topics are not covered by existing documentation?
3. What new articles or content expansions would be most valuable?

Respond with a JSON object:

```json
{{
    "gaps": [
        {{
            "gap_type": "missing|incomplete|outdated",
            "severity": "critical|high|medium|low",
            "title": "Short title for the gap",
            "description": "Detailed description of what's missing and why it matters",
            "suggested_article_title": "Proposed title for new content",
            "suggested_content_outline": [
                "Introduction and prerequisites",
                "Step-by-step instructions",
                "Code examples",
                "Troubleshooting"
            ],
            "related_articles": ["path/to/related.md"],
            "estimated_effort": "small|medium|large"
        }}
    ],
    "overall_assessment": "Summary of the documentation coverage for this JTBD",
    "priority_actions": [
        "Most important action to improve coverage",
        "Second priority action"
    ]
}}
```

Focus on gaps that would have the highest impact on developer success for this specific job.
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
