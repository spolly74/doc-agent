# Doc Agent

An agentic content evaluation system for technical documentation. Evaluates articles against a developer-focused rubric, performs JTBD (Job-To-Be-Done) coverage analysis, validates code samples, and produces actionable reports.

## Features

- **Single-article evaluation** - Score articles against a 7-dimension rubric
- **Cross-article JTBD analysis** - Assess how well article sets support customer jobs
- **Coverage gap detection** - Identify missing articles and propose new content
- **Code sample validation** - Check sample quality and identify unreferenced samples
- **Batch processing** - Evaluate entire repo or targeted folder paths
- **Actionable output** - CSV scoring matrix + summary report

## Installation

```bash
# Clone the repository
git clone https://github.com/spolly74/doc-agent.git
cd doc-agent

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

1. Copy the example configuration:
   ```bash
   cp doc-agent.config.json.example doc-agent.config.json
   ```

2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

## Usage

### Full Evaluation

Evaluate all articles in a documentation directory:

```bash
doc-agent full --target ./docs --output ./results
```

### Tiered Evaluation

Quick scan all articles, then deep-evaluate only those below threshold:

```bash
doc-agent tiered --target ./docs --threshold 3.5 --output ./results
```

### JTBD-Scoped Evaluation

Evaluate coverage for a specific job-to-be-done:

```bash
doc-agent jtbd --jtbd-file ./jtbd.csv --jtbd-id jtbd-001 --output ./results
```

The JTBD command supports two CSV formats:

1. **Step-based format** - Detailed JTBD definitions with explicit steps
2. **Golden Path format** - Project tracking with JTBD titles (auto-detected)

### Code Sample Validation

Validate code samples and check references:

```bash
doc-agent samples --articles-repo ./docs --samples-repo ./samples --output ./results
```

## Evaluation Dimensions

Articles are scored 1-7 on each dimension:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Pattern Compliance | 15% | Follows the correct content pattern (how-to, quickstart, etc.) |
| Developer Focus | 25% | Code-first, actionable, time-to-success optimized |
| Code Quality | 20% | Well-written, follows best practices, production-ready |
| Code Completeness | 15% | All necessary samples present, can copy-paste and run |
| Structure | 10% | Well-organized, scannable, logical flow |
| Accuracy | 15% | Technically correct, up-to-date, uses current APIs |
| JTBD Alignment | N/A | Supports its intended customer job (when applicable) |

## Output

### scores.csv

A CSV file with one row per article, including:
- All dimension scores
- Overall weighted score
- Priority rank for remediation
- Top issues summary
- Code sample status

### report.md

A Markdown summary report including:
- Executive summary with statistics
- Top articles requiring attention
- Score distribution by dimension
- Coverage gaps and missing articles
- Common issues (systemic patterns)
- Code sample status

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy src/

# Run linting
ruff check src/

# Format code
ruff format src/
```

## License

MIT
