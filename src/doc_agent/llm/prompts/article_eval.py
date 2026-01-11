"""Prompts for article evaluation.

Evaluation criteria derived from documentation standards defined in:
- copilot-instructions.md (writing style, structure)
- dev-focused.instructions.md (developer focus, code practices)
- foundry-branding.instructions.md (terminology, branding)
- Pattern templates (how-to, quickstart, tutorial, concept)
"""

from doc_agent.models.article import Article
from doc_agent.models.enums import ContentPattern

ARTICLE_EVALUATION_SYSTEM_PROMPT = """You are an expert technical documentation evaluator for Microsoft Learn content. Evaluate articles against an 8-dimension rubric focused on developer experience.

Core principles:
1. **Developer-first**: Minimize time-to-first-success
2. **Code-forward**: Working code early, complete with imports
3. **Scannable**: Short paragraphs, clear headings, logical structure
4. **Accurate**: Current APIs, correct terminology, no fabrication
5. **Pattern-compliant**: Follow the correct article pattern
6. **Brand-consistent**: Use correct Microsoft terminology

Be specific. Cite line numbers or sections. Provide actionable recommendations.
Respond with valid JSON matching the requested schema."""


def _get_pattern_criteria(ms_topic: str | None) -> str:
    """Get pattern-specific evaluation criteria based on ms.topic."""
    pattern = ContentPattern.from_ms_topic(ms_topic)

    if pattern == ContentPattern.HOW_TO:
        return """**How-to Pattern Requirements:**
- H1: "<verb> * <noun>" format (no gerunds)
- Prerequisites must be first H2
- Each H2: verb+noun title, brief intro, then ordered steps
- Validation/verification steps included
- Next step OR Related content section (not both)
- No bulleted list of sections in intro"""

    elif pattern == ContentPattern.QUICKSTART:
        return """**Quickstart Pattern Requirements:**
- H1: "Quickstart: <verb> * <noun>"
- Target: completable in <=10 minutes
- Free trial link before first H2
- Prerequisites must be first H2
- Minimal narrative, one success checkpoint
- Clean up resources section (optional)
- Single outcome focus"""

    elif pattern == ContentPattern.TUTORIAL:
        return """**Tutorial Pattern Requirements:**
- H1: "Tutorial: <verb> * <noun>"
- Target: completable in <=30 minutes
- Green checklist outline before first H2
- Free trial link before first H2
- Prerequisites must be first H2 (not numbered)
- Staged progression, each stage builds on prior
- Clean up resources section required
- Single learning objective"""

    elif pattern in (ContentPattern.CONCEPT, ContentPattern.OVERVIEW):
        return """**Concept/Overview Pattern Requirements:**
- H1: "<noun> concepts" OR "What is <noun>?" OR "<noun> overview"
- NO procedural content or numbered steps
- Define the concept early ("X is a Y that does Z")
- Use examples to illustrate abstract ideas
- Link to related how-to/tutorial content
- Code samples NOT expected (evaluate links instead)"""

    else:
        return """**General Pattern Requirements:**
- Clear H1 that sets expectations
- Prerequisites section if procedural
- Logical heading hierarchy
- Next step or Related content section"""


def build_article_evaluation_prompt(article: Article) -> str:
    """Build the evaluation prompt for an article.

    Args:
        article: Article object to evaluate.

    Returns:
        Formatted evaluation prompt.
    """
    content = article.content_without_frontmatter
    if len(content) > 50000:
        content = content[:50000] + "\n\n[Content truncated due to length...]"

    pattern_criteria = _get_pattern_criteria(article.metadata.ms_topic)
    is_concept = ContentPattern.from_ms_topic(article.metadata.ms_topic) in (
        ContentPattern.CONCEPT,
        ContentPattern.OVERVIEW,
    )

    return f"""Evaluate this Microsoft Learn article against the 8-dimension rubric.

## Article Information

- **Path**: {article.relative_path}
- **Title**: {article.metadata.title}
- **Type (ms.topic)**: {article.metadata.ms_topic or "Not specified"}
- **Service**: {article.metadata.ms_service or "Not specified"}
- **Last Updated**: {article.metadata.ms_date.strftime("%Y-%m-%d") if article.metadata.ms_date else "Unknown"}
- **Customer Intent**: {article.metadata.customer_intent or "Not specified"}

## Article Content

{content}

## Evaluation Dimensions (Score 1-7)

{pattern_criteria}

### 1. Pattern Compliance (pattern_compliance)
*Does the article follow the correct pattern for ms.topic="{article.metadata.ms_topic or 'unknown'}"?*

Check:
- Correct H1 format for the pattern
- Required sections present and ordered correctly
- Prerequisites as first H2 (for procedural content)
- Appropriate section structure

Scoring: 1-2=wrong pattern/missing sections, 3-4=structural issues, 5-6=good adherence, 7=exemplary

### 2. Developer Focus (dev_focus)
*Does the article minimize time-to-first-success?*

{"For concept articles, evaluate clarity and actionable links instead of code position." if is_concept else "Check:"}
{'''- Position of first code block (earlier is better)
- Minimal "hello world" example present
- Prerequisites complete (including RBAC roles)
- Each snippet explains: what it does, inputs, expected output
- Reference links after code blocks (classes, methods, schemas)''' if not is_concept else '''- Clear explanation of the concept upfront
- Actionable takeaways for developers
- Links to related procedural content (how-to, tutorials)'''}

Scoring: 1-2={"confusing, no actionable content" if is_concept else "code buried or missing"}, 3-4={"limited guidance" if is_concept else "code delayed"}, 5-6={"clear with good links" if is_concept else "code-first approach"}, 7=exemplary

### 3. Code Quality (code_quality)
*Are code samples well-written and production-ready?*

{"Set score to null - code not expected for concept articles." if is_concept else '''Check:
- Language tags on all code blocks
- Import/using statements included at top
- Line length <=80 characters (no horizontal scroll)
- No hardcoded secrets or credentials
- Current API versions
- Error handling where appropriate'''}

{"" if is_concept else "Scoring: 1-2=broken/security issues, 3-4=functional but poor practices, 5-6=production-ready, 7=exemplary reference"}

### 4. Code Completeness (code_completeness)
*Does the article have all necessary code samples?*

{"Set score to null - evaluate links to procedural content instead." if is_concept else '''Check:
- All procedural steps have corresponding code
- Multi-language tabs if applicable (order: Python, C#, JavaScript, Java)
- Full sample linked or available
- Reference links to SDK documentation after snippets'''}

{"" if is_concept else "Scoring: 1-2=critical samples missing, 3-4=notable gaps, 5-6=good coverage, 7=complete"}

### 5. Structure (structure)
*Is the article well-organized for scanning?*

Check:
- Heading hierarchy (no skipped levels, no stacked headings without text)
- Sentence case for headings (not Title Case)
- Short paragraphs (1-3 sentences)
- Lists/tables for scannable content
- No gerunds in headings
- Oxford comma in lists

Scoring: 1-2=no clear structure, 3-4=basic but issues, 5-6=easy to scan, 7=optimal architecture

### 6. Accuracy (accuracy)
*Is the content technically correct and current?*

Check:
- ms.date currency (flag if >12 months old)
- API versions current (no deprecated methods)
- No fabricated parameters or features
- Terminology consistent with product
- Version-specific content properly tagged with monikers

Scoring: 1-2=major errors/severely outdated, 3-4=some inaccuracies, 5-6=accurate and current, 7=verified latest

### 7. Branding Compliance (branding_compliance)
*Does the article use correct Microsoft terminology?*

Check:
- First-mention vs subsequent-mention patterns (e.g., "Microsoft Foundry" first, then "Foundry")
- Protected terms unchanged (Azure OpenAI, SDK library names)
- Historical context preserved (don't change terms in "formerly X" phrases)
- Grammar correct after name changes (a/an usage)
- Portal capitalization (lowercase: "Azure portal", "Foundry portal")

Scoring: 1-2=major terminology errors, 3-4=inconsistent usage, 5-6=mostly correct, 7=exemplary compliance

### 8. JTBD Alignment (jtbd_alignment)
*Does the article support its stated customer intent?*

Customer intent: "{article.metadata.customer_intent or 'Not specified'}"

{"If no customer intent specified, set score to null." if not article.metadata.customer_intent else '''Check:
- Article delivers on stated intent
- Prerequisites enable the customer to succeed
- Next steps connect to natural follow-on tasks'''}

## Output Format

Provide your evaluation as a JSON object:

```json
{{
  "dimension_scores": [
    {{
      "dimension": "pattern_compliance",
      "score": <1-7 or null>,
      "confidence": <0.0-1.0>,
      "rationale": "<1-2 sentence justification>"
    }},
    // ... repeat for all 8 dimensions
  ],
  "issues": [
    {{
      "dimension": "<dimension_name>",
      "severity": "<critical|high|medium|low>",
      "title": "<short issue title>",
      "description": "<detailed description>",
      "location": "<line number, section, or 'general'>",
      "suggestion": "<how to fix>"
    }}
  ],
  "summary": "<2-3 sentence overall assessment>"
}}
```

Focus on high-impact, actionable issues. Prioritize: move > modify > add."""


def build_quick_scan_prompt(article: Article) -> str:
    """Build a lightweight prompt for quick structural scanning.

    Used in tiered mode for initial triage before full evaluation.

    Args:
        article: Article object to scan.

    Returns:
        Formatted quick scan prompt.
    """
    content_preview = article.content_without_frontmatter[:5000]
    if len(article.content_without_frontmatter) > 5000:
        content_preview += "\n\n[Content truncated for quick scan...]"

    is_concept = ContentPattern.from_ms_topic(article.metadata.ms_topic) in (
        ContentPattern.CONCEPT,
        ContentPattern.OVERVIEW,
    )

    return f"""Quick structural assessment of this documentation article.

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

## Quick Checks

1. **Pattern Match**: Does ms.topic match content style? (how-to=procedural, concept=explanatory)
2. **Structure Valid**: Headings properly nested? No skipped levels?
3. **Has Code**: {"N/A for concept" if is_concept else "Does procedural content have code examples?"}
4. **Recent**: Is ms.date within 12 months?
5. **Prerequisites**: Does procedural content have prerequisites section first?
6. **Branding**: Any obvious terminology issues? (Azure AI Foundry->Microsoft Foundry)

## Output Format

```json
{{
  "quick_score": <1-7 overall estimate>,
  "needs_deep_evaluation": <true if score < 4 or critical issues>,
  "flags": ["<specific concerns>"],
  "pattern_match": <true|false>,
  "structure_valid": <true|false>,
  "has_code": <true|false>,
  "is_recent": <true|false>,
  "has_prerequisites": <true|false>
}}
```

Brief triage only."""
