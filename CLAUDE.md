# Claude Code Context

This is Doc Agent - an agentic tool for evaluating technical documentation.

## Project Structure

```
src/doc_agent/
├── cli/          # Typer CLI commands (full, tiered, jtbd, samples)
├── config/       # Pydantic configuration models and loading
├── models/       # Domain models (Article, EvaluationResult, etc.)
├── llm/          # LLM provider abstraction (Claude API)
├── context/      # Data loading (TOC parser, metadata extractor, etc.)
├── evaluators/   # Evaluation logic (article, JTBD, code samples)
├── orchestration/# Batch processing, state management, progress
├── output/       # CSV and Markdown report generation
└── utils/        # Shared utilities
```

## Key Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run CLI
doc-agent --help
doc-agent full --target ./docs --output ./results

# Run tests
pytest

# Type check
mypy src/

# Lint
ruff check src/
```

## Implementation Status

- [x] Phase 1: Project foundation (pyproject.toml, config, models, CLI skeleton)
- [x] Phase 2: Context layer (TOC parser, metadata extractor, include resolver)
- [x] Phase 3: LLM integration (Claude provider, prompts)
- [x] Phase 4: Article evaluation (7-dimension rubric scoring)
- [x] Phase 5: Output generation (CSV, Markdown reports)
- [x] Phase 6: JTBD & samples (JTBD analyzer, code sample validator)

## Key Design Decisions

1. **Async throughout** - LLM calls and file I/O are async for performance
2. **Pydantic models** - All data structures use Pydantic for validation
3. **Protocol-based LLM abstraction** - Easy to swap providers
4. **SQLite state** - Resume capability for large batch evaluations
5. **Typer CLI** - Modern, type-hint based CLI framework
6. **Dual JTBD format support** - Both step-based and Golden Path (title-only) formats
