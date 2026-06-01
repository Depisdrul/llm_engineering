# Ruff Linting Guide

## Quick Commands

```bash
# Check all files for linting errors
uv run ruff check .

# Auto-fix safe issues
uv run ruff check --fix .

# Auto-fix including unsafe fixes
uv run ruff check --fix --unsafe-fixes .

# Format code
uv run ruff format .

# Check specific files/folders
uv run ruff check week1/ week2/
```

## VS Code Integration

The `.vscode/settings.json` configures:
- **Ruff** as the default linter and formatter
- **Auto-fix on save** - imports are organized and fixable issues are corrected
- **Format on save** - code is automatically formatted
- **Community contributions excluded** from searches and linting

## What Was Fixed

### Python Scripts (week1-8)
- ✅ Import sorting (imports now follow standard → third-party → local order)
- ✅ Modern type hints (replaced `List`/`Dict` from typing with built-in `list`/`dict`)
- ✅ Simplified dict() calls to literals
- ✅ Removed unnecessary file mode arguments
- ✅ Other modernizations for Python 3.11+

### Jupyter Notebooks
- ✅ Import sorting
- ✅ Removed unused imports
- ✅ Fixed Python 3.12-only f-string syntax to work with Python 3.11

## Configuration

All settings are in `pyproject.toml`:
- **Excluded folders**: community-contributions, .venv, .git
- **Target**: Python 3.11+
- **Line length**: 120 characters
- **Ignored rules**: Rules that don't make sense for educational code

## Recommended VS Code Extensions

Install these for the best experience:
- `charliermarsh.ruff` - Ruff linter/formatter
- `ms-python.python` - Python extension
- `ms-toolsai.jupyter` - Jupyter notebooks
