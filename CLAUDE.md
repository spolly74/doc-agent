# Claude Code Context

This is the Foundry Content Evaluation System - an agentic tool for evaluating Microsoft Foundry documentation.

## Project Structure

```
src/foundry_eval/
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
foundry-eval --help
foundry-eval full --target ./docs --output ./results

# Run tests
pytest

# Type check
mypy src/

# Lint
ruff check src/
```

## Implementation Status

- [x] Phase 1: Project foundation (pyproject.toml, config, models, CLI skeleton)
- [ ] Phase 2: Context layer (TOC parser, metadata extractor, include resolver)
- [ ] Phase 3: LLM integration (Claude provider, prompts)
- [ ] Phase 4: Article evaluation (7-dimension rubric scoring)
- [ ] Phase 5: Output generation (CSV, Markdown reports)
- [ ] Phase 6: JTBD & samples (JTBD analyzer, code sample validator)

## Key Design Decisions

1. **Async throughout** - LLM calls and file I/O are async for performance
2. **Pydantic models** - All data structures use Pydantic for validation
3. **Protocol-based LLM abstraction** - Easy to swap providers
4. **SQLite state** - Resume capability for large batch evaluations
5. **Typer CLI** - Modern, type-hint based CLI framework

## Spec Reference

See `foundry-content-eval-spec.md` for the full technical specification.
