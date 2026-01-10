"""Prompts for article evaluation."""

from doc_agent.models.article import Article

ARTICLE_EVALUATION_SYSTEM_PROMPT = """You are an expert technical documentation evaluator. Your role is to evaluate documentation articles against a developer-focused rubric.

You evaluate with these principles:
1. **Developer-first**: Documentation should minimize time-to-first-success for developers
2. **Code-forward**: Working code examples should appear early and be complete
3. **Scannable**: Developers should be able to quickly find what they need
4. **Accurate**: Technical content must be correct and current
5. **Structured**: Articles should follow consistent patterns

Be specific in your feedback. Cite line numbers or section names when identifying issues. Provide actionable recommendations.

You must respond with valid JSON matching the requested schema."""


def build_article_evaluation_prompt(article: Article) -> str:
    """Build the full evaluation prompt for an article.

    Args:
        article: Article object to evaluate.

    Returns:
        Formatted evaluation prompt.
    """
    # Truncate content if very long (to stay within context limits)
    content = article.content_without_frontmatter
    if len(content) > 50000:
        content = content[:50000] + "\n\n[Content truncated due to length...]"

    return f"""Evaluate this technical documentation article against the 7-dimension rubric.

## Article Information

- **Path**: {article.relative_path}
- **Title**: {article.metadata.title}
- **Type (ms.topic)**: {article.metadata.ms_topic or "Not specified"}
- **Service**: {article.metadata.ms_service or "Not specified"}
- **Last Updated**: {article.metadata.ms_date.strftime("%Y-%m-%d") if article.metadata.ms_date else "Unknown"}
- **Customer Intent**: {article.metadata.customer_intent or "Not specified"}

## Article Content

{content}

## Evaluation Instructions

Score each dimension from 1-7 based on the criteria below. Provide specific justification and identify issues.

### Dimension 1: Pattern Compliance (pattern_compliance)
*How well does the article follow the appropriate content pattern?*

Based on ms.topic="{article.metadata.ms_topic or 'unknown'}", evaluate whether the article:
- Follows the correct pattern (how-to, quickstart, tutorial, concept, reference)
- Has all required sections for that pattern
- Uses appropriate section ordering

Score guide:
- 1-2: Wrong pattern or missing critical sections
- 3-4: Mostly correct but structural issues
- 5-6: Good pattern adherence
- 7: Exemplary, could serve as template

### Dimension 2: Developer Focus (dev_focus)
*How well does the article support "time to first success" for developers?*

Evaluate:
- Position of first code block (earlier = better)
- Presence of minimal runnable example
- Code-to-prose ratio (more code, less fluff)
- Prerequisites completeness
- Expected output documentation

Score guide:
- 1-2: No code or code buried after extensive prose
- 3-4: Code present but delayed or lacking context
- 5-6: Code-first approach with good context
- 7: Exemplary - hello world within first scroll

### Dimension 3: Code Quality (code_quality)
*How well-written and maintainable are the code samples?*

Evaluate:
- Language tags on all code blocks
- Import/using statements included
- Line length ≤80 characters
- Error handling where appropriate
- No hardcoded secrets
- Current API versions

Score guide:
- 1-2: Broken code or security issues
- 3-4: Functional but poor practices
- 5-6: Good quality, production-ready patterns
- 7: Exemplary reference implementation

If no code is present, set score to null.

### Dimension 4: Code Completeness (code_completeness)
*Does the article have all necessary code samples?*

Evaluate:
- All procedural steps have corresponding code
- Multi-language support where appropriate
- Full sample available (linked or inline)
- Reference links to SDK docs

Score guide:
- 1-2: Critical samples missing
- 3-4: Adequate but notable gaps
- 5-6: Good coverage
- 7: Complete - every operation demonstrated

### Dimension 5: Structure (structure)
*How well-organized is the article for developer consumption?*

Evaluate:
- Heading hierarchy (no skipped levels)
- Logical section ordering
- Use of lists, tables, callouts
- Paragraph length (no walls of text)
- Clear goal statement

Score guide:
- 1-2: No clear structure
- 3-4: Basic structure but issues
- 5-6: Good structure, easy to scan
- 7: Optimal information architecture

### Dimension 6: Accuracy (accuracy)
*Is the content technically correct and up-to-date?*

Evaluate:
- ms.date currency (flag if >12 months old)
- API versions current
- No deprecated parameters/methods
- Terminology consistent with product

Score guide:
- 1-2: Major errors or severely outdated
- 3-4: Some inaccuracies or outdated elements
- 5-6: Accurate and reasonably current
- 7: Verified accurate, latest APIs

### Dimension 7: JTBD Alignment (jtbd_alignment)
*How well does this article support its intended customer job?*

Based on the customer intent: "{article.metadata.customer_intent or 'Not specified'}"

Evaluate:
- Does the article deliver on its stated intent?
- Are prerequisites clear?
- Do next steps connect to natural follow-on tasks?

If no customer intent is specified, set score to null.

## Output Format

Provide your evaluation as a JSON object with this structure:

```json
{{
  "dimension_scores": [
    {{
      "dimension": "pattern_compliance",
      "score": <1-7 or null>,
      "confidence": <0.0-1.0>,
      "rationale": "<1-2 sentence justification>"
    }},
    // ... repeat for all 7 dimensions
  ],
  "issues": [
    {{
      "dimension": "<dimension_name>",
      "severity": "<critical|high|medium|low>",
      "title": "<short issue title>",
      "description": "<detailed description>",
      "location": "<line number, section name, or 'general'>",
      "suggestion": "<how to fix>"
    }}
  ],
  "summary": "<2-3 sentence overall assessment>"
}}
```

Be thorough but concise. Focus on actionable issues."""


def build_quick_scan_prompt(article: Article) -> str:
    """Build a lightweight prompt for quick structural scanning.

    This is used in tiered mode for the initial pass before deciding
    whether to do a full evaluation.

    Args:
        article: Article object to scan.

    Returns:
        Formatted quick scan prompt.
    """
    # Only include first part of content for quick scan
    content_preview = article.content_without_frontmatter[:5000]
    if len(article.content_without_frontmatter) > 5000:
        content_preview += "\n\n[Content truncated for quick scan...]"

    return f"""Perform a quick structural assessment of this documentation article.

## Article Information

- **Path**: {article.relative_path}
- **Title**: {article.metadata.title}
- **Type (ms.topic)**: {article.metadata.ms_topic or "Not specified"}
- **Last Updated**: {article.metadata.ms_date.strftime("%Y-%m-%d") if article.metadata.ms_date else "Unknown"}
- **Word Count**: ~{article.structure.word_count}
- **Code Blocks**: {article.structure.code_block_count}
- **Has Prerequisites**: {article.structure.has_prerequisites}
- **Has Next Steps**: {article.structure.has_next_steps}

## Content Preview

{content_preview}

## Quick Assessment

Evaluate these quick-check items:

1. **Pattern Match**: Does ms.topic match the content style?
2. **Structure Valid**: Are headings properly nested (H1 → H2 → H3)?
3. **Has Code**: Does a developer article have code examples?
4. **Recent**: Is ms.date within the last 12 months?
5. **Prerequisites**: Does procedural content have a prerequisites section?

## Output Format

```json
{{
  "quick_score": <1-7 overall estimate>,
  "needs_deep_evaluation": <true if score < 4 or critical issues>,
  "flags": [
    "<list of specific concerns that warrant deep evaluation>"
  ],
  "pattern_match": <true|false>,
  "structure_valid": <true|false>,
  "has_code": <true|false>,
  "is_recent": <true|false>,
  "has_prerequisites": <true|false>
}}
```

Be brief. This is a triage assessment."""
