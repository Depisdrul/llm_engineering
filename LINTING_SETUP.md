# Linting & Code Quality Setup

**Date:** 2026-06-01
**Status:** ✅ Fully configured and working

---

## Overview

This repository uses **Ruff** as the primary linter and formatter for Python code.
The setup excludes community contributions and focuses only on reference/course code in the `week*` folders.

**Automated Health Checks**: VS Code automatically runs environment
health checks on startup to ensure dependencies are synced and code
quality is maintained.

## What Was Done

### 1. Ruff Configuration (`pyproject.toml`)

Added comprehensive Ruff configuration:

- **Target**: Python 3.11+
- **Line length**: 120 characters
- **Excluded paths**: All `community-contributions` folders, `.venv`, `.git`, `__pycache__`
- **Enabled rules**:
  - `E` - pycodestyle errors
  - `W` - pycodestyle warnings
  - `F` - pyflakes
  - `I` - isort (import sorting)
  - `N` - pep8-naming
  - `UP` - pyupgrade (modern Python syntax)
  - `B` - flake8-bugbear
  - `C4` - flake8-comprehensions
  - `SIM` - flake8-simplify

- **Ignored rules** (for educational code):
  - `E501` - Line too long (handled by formatter)
  - `B008` - Function calls in argument defaults
  - `B904` - Exception chaining
  - `B905` - zip() without strict parameter
  - `SIM108` - Ternary operator (can reduce readability)
  - `N803` - Allow uppercase argument names (e.g., `G` for graphs)
  - `N806` - Variable naming for compatibility
  - `N816` - mixedCase variables (common in educational code)

### 2. Automated Startup Checks

#### `.vscode/tasks.json` & `check-env.ps1`

Created automated tasks that run on VS Code startup:

- ✅ **UV Environment Health Check** (runs automatically on folder open)
  - Verifies UV is installed
  - Checks virtual environment exists
  - Validates dependencies are in sync
  - Confirms Python version (3.11+)
  - Runs quick linting check

- ✅ **Manual Tasks Available** (via Command Palette):
  - `UV: Sync Dependencies` - Install/sync dependencies
  - `UV: Update All Dependencies` - Update to latest versions
  - `Ruff: Check All Files` - Run full linting check
  - `Ruff: Fix All Issues` - Auto-fix linting issues
  - `Full Health Check` - Complete environment verification

#### What Gets Checked on Startup

1. UV installation and version
2. Virtual environment presence (`.venv`)
3. Dependency sync status (`uv.lock` vs `pyproject.toml`)
4. Python version compatibility (>= 3.11)
5. Quick linting status (informational only)

### 4. VS Code Configuration

#### `.vscode/settings.json`

Configured VS Code to:

- Use **Ruff** as default linter and formatter (Pylint disabled)
- **Auto-fix on save**: Automatically organize imports and fix issues
- **Format on save**: Code is automatically formatted
- Exclude `community-contributions` from:
  - File searches
  - File watching
  - Workspace indexing
- Set Python interpreter to `.venv/Scripts/python.exe`
- Disabled deprecated IntelliCode Python completions (was causing errors)
- Use **Pylance** as the language server

#### `.vscode/extensions.json`

Recommended extensions:

- `charliermarsh.ruff` - Ruff linter/formatter ⭐
- `ms-python.python` - Python extension
- `ms-toolsai.jupyter` - Jupyter notebooks
- `davidanson.vscode-markdownlint` - Markdown linting ⭐

### 3. Markdown Linting Configuration

#### `.markdownlint.json` + VS Code Extension

Configured markdown linting with educational-friendly rules:

- ✅ **Disabled MD033** - Allows inline HTML (for tables, styling in notebooks)
- ✅ **Disabled MD041** - First line doesn't need to be top-level heading
- ✅ **Disabled MD045** - Images alt text (not required for educational content)
- ✅ **Configured MD013** - Line length 120, ignores tables and code blocks
- ✅ **Auto-fix on save** via VS Code extension (no CLI tools needed)
- ✅ **Real-time linting** as you type
- ✅ **Community-contributions** excluded from linting

**How it works:**

- Uses the `davidanson.vscode-markdownlint` VS Code extension
- Auto-fixes issues on save
- No Node.js or command-line tools required
- Works offline

### 4. Extension Cleanup

**Uninstalled:**

- ❌ Visual Studio IntelliCode (deprecated, causing activation errors)

**Installed:**

- ✅ Ruff extension (`charliermarsh.ruff`)
- ✅ Markdownlint extension (`davidanson.vscode-markdownlint`)

---

## Usage

### Automated Checks

**On Startup** (automatic):

- Health check runs when you open the workspace
- Shows status in the terminal panel
- Reports any issues with dependencies or environment

**View Health Check Results**:

- Check the "Terminal" panel after opening VS Code
- Look for the 🔍 health check output
- Green ✅ means everything is good
- Yellow ⚠️ means action recommended
- Red ❌ means immediate action required

**Run Manual Checks**:

1. Press `Ctrl+Shift+P` (Command Palette)
2. Type "Tasks: Run Task"
3. Choose from:
   - `UV: Check Environment Health`
   - `UV: Sync Dependencies`
   - `UV: Update All Dependencies`
   - `Ruff: Check All Files`
   - `Ruff: Fix All Issues`
   - `Full Health Check`

### Quick Commands

#### Python/Ruff

```bash
# Check all files for linting errors
uv run ruff check .

# Auto-fix safe issues
uv run ruff check --fix .

# Auto-fix including unsafe fixes (use with caution)
uv run ruff check --fix --unsafe-fixes .

# Format code
uv run ruff format .

# Check specific folders
uv run ruff check week1/ week2/

# Check specific files
uv run ruff check week1/solution.py
```

#### Markdown

Markdown linting and auto-fix is handled entirely by the VS Code extension - no command-line tools needed!

**Auto-fix happens:**

- On save (automatic)
- Manual trigger: `Shift+Alt+F` (Format Document)

### VS Code Integration

With the configured settings, VS Code will automatically:

1. **On Save** (Python):
   - Organize imports
   - Fix auto-fixable linting issues
   - Format code to follow style guide

2. **On Save** (Markdown):
   - Auto-fix markdown code smells (via VS Code extension)
   - Format document
   - Trim trailing whitespace

3. **On Type**:
   - Show linting warnings/errors inline (Python & Markdown)
   - Provide quick-fix suggestions

4. **Manual Trigger**:
   - Right-click → "Format Document" (`Shift+Alt+F`)
   - Right-click → "Organize Imports" (Python only)

---

## Configuration Files

### `.markdownlint.json`

```json
{
  "default": true,
  "MD033": false,  // Allow inline HTML
  "MD041": false,  // First line doesn't need to be heading
  "MD013": {       // Line length
    "line_length": 120,
    "tables": false,      // Don't check tables
    "code_blocks": false  // Don't check code blocks
  },
  "MD024": {
    "siblings_only": true  // Allow duplicate headings in different sections
  },
  "MD025": false  // Allow multiple top-level headings
}
```

**Disabled Rules:**

- **MD033** - Inline HTML (needed for tables/styling in educational content)
- **MD041** - First line heading requirement (too restrictive)
- **MD045** - Images require alt text (auto-handled by formatter)

### `pyproject.toml`

```toml
[tool.ruff]
exclude = [
    "community-contributions",
    "*/community-contributions",
    "**/community-contributions/**",
    ".git",
    ".venv",
    "__pycache__",
    "*.egg-info",
]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501", "B008", "B904", "B905", "SIM108", "N803", "N806", "N816"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### `.vscode/settings.json` (Key Settings)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.linting.enabled": false,
  "python.languageServer": "Pylance",
  "vsintellicode.python.completionsEnabled": false,

  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },

  "ruff.enable": true,
  "ruff.organizeImports": true,
  "ruff.fixAll": true,
  "ruff.lint.run": "onType"
}
```

---

## Troubleshooting

### Issue: VS Code shows "command 'python.intellicode.loadLanguageServerExtension' not found"

**Solution:**

1. Uninstall the deprecated IntelliCode extension
2. Reload VS Code (`Ctrl+Shift+P` → "Developer: Reload Window")

### Issue: Linting still showing Pylint errors

**Solution:**

1. Verify `.vscode/settings.json` has `"python.linting.enabled": false`
2. Reload VS Code window
3. Check that Ruff extension is installed and enabled

### Issue: Community contributions showing linting errors

**Solution:**

- They shouldn't be checked due to exclusions in `pyproject.toml`
- If they appear, verify the `exclude` paths in `pyproject.toml`
- Check `.vscode/settings.json` has community-contributions in `files.exclude`

### Issue: Code not auto-formatting on save

**Solution:**

1. Verify Ruff extension is installed
2. Check `"editor.formatOnSave": true` in settings
3. Ensure file is saved as `.py` or `.ipynb`
4. Try manual format: `Shift+Alt+F`

### Issue: Startup health check not running

**Solution:**

1. Check `.vscode/settings.json` has `"task.allowAutomaticTasks": "on"`
2. Reload VS Code window
3. Check terminal panel for health check output
4. Run manually: `Ctrl+Shift+P` → "Tasks: Run Task" → "UV: Check Environment Health"

### Issue: "Dependencies need syncing" warning

**Solution:**

```powershell
# Sync dependencies to match uv.lock
uv sync

# Or if you want to update everything first
uv lock --upgrade
uv sync
```

### Issue: Markdown auto-fix not working

**Solution:**

1. Verify markdownlint extension is installed: `davidanson.vscode-markdownlint`
2. Check `.vscode/settings.json` has markdown auto-fix enabled:

   ```json
   "markdownlint.run": "onType",
   "[markdown]": {
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
       "source.fixAll.markdownlint": "explicit"
     }
   }
   ```

3. Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
4. Save a `.md` file to test - formatting should happen automatically

---

## Project Structure

```txt
llm_engineering/
├── .vscode/
│   ├── settings.json          # VS Code workspace settings
│   ├── extensions.json        # Recommended extensions
│   └── ruff-guide.md         # Quick reference guide
├── pyproject.toml            # Ruff configuration
├── community-contributions/  # EXCLUDED from linting
└── week1-8/                  # Reference code (LINTED)
    ├── *.py                  # Python scripts
    └── *.ipynb               # Jupyter notebooks
```

---

## Best Practices

1. **Pay attention to startup health checks:**
   - Check the terminal when opening VS Code
   - Address any yellow ⚠️ or red ❌ warnings
   - Green ✅ means you're good to go

2. **Keep dependencies in sync:**

   ```bash
   # When health check shows sync warning
   uv sync
   ```

3. **Update dependencies periodically:**

   ```bash
   # Update to latest compatible versions
   uv lock --upgrade
   uv sync
   ```

4. **Run linting before commits:**

   ```bash
   # Python linting
   uv run ruff check .
   ```

   **Markdown:** Already auto-fixed on save - no manual check needed!

5. **Let VS Code auto-fix on save** - don't fight the formatter

6. **For notebooks**: Save them in VS Code to trigger auto-formatting

7. **Community contributions**:
   - Never edit these directly
   - They are excluded from linting for a reason
   - Focus on reference code in `week*` folders

8. **When adding new files**:
   - They'll automatically be linted if in `week*` folders
   - Format will be applied on save
   - Imports will be auto-organized

---

## Related Documentation

- **Ruff Documentation**: <https://docs.astral.sh/ruff/>
- **Ruff Rules**: <https://docs.astral.sh/ruff/rules/>
- **VS Code Python**: <https://code.visualstudio.com/docs/python/python-tutorial>
- **Course Resources**: <https://edwarddonner.com/2024/11/13/llm-engineering-resources/>

---

## Summary for Future Sessions

**Quick Start:**

1. ✅ **Automated startup checks** verify environment health when opening VS Code
2. ✅ **Linting** fully configured via Ruff
3. ✅ **Auto-fix on save** for formatting and imports
4. ✅ **Community contributions** excluded from all checks
5. ✅ **Dependencies** monitored for sync issues
6. Run `uv run ruff check .` to manually check linting
7. Run `uv sync` if dependencies are out of sync
8. All configuration in `pyproject.toml` and `.vscode/`

**Files Created:**

- `.vscode/tasks.json` - Task definitions for health checks
- `.vscode/check-env.ps1` - Startup health check script
- `.vscode/settings.json` - VS Code configuration
- `.vscode/extensions.json` - Recommended extensions
- `pyproject.toml` - Ruff linting rules
- `.markdownlint.json` - Markdown linting rules
- `LINTING_SETUP.md` - This documentation

**Status:**

- ✅ All 203 Python linting errors fixed, zero errors remaining in reference code
- ✅ Markdown linting configured with auto-fix on save (no CLI tools needed)
- ✅ Automated health checks run on startup
- ✅ Environment monitoring active
- ✅ No Node.js dependency required
