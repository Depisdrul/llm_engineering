# Utility Scripts

This folder contains utility scripts for repository maintenance and development tasks.

## Active Scripts (Cross-Platform)

### `kill-gradio.py`

**Purpose:** Kill all Gradio processes running on ports 7860-7900

**Usage:**

```bash
# Run directly
python scripts/kill-gradio.py

# Or via VS Code task
Ctrl+Shift+P → "Tasks: Run Task" → "Kill Gradio Processes"
```

**Platform Support:** Windows, macOS, Linux

- Windows: Uses `netstat` + `taskkill`
- Unix/macOS: Uses `lsof` + `kill -9`

## Legacy Scripts (Windows Only)

These PowerShell scripts are kept for reference but are superseded by the Python versions:

### `check-env.ps1`

**Superseded by:** `.vscode/check-env.py`

**Purpose:** Environment health check (UV, dependencies, Python version, linting)

**Note:** The Python version in `.vscode/check-env.py` is now used by the VS Code task and works cross-platform.

## Adding New Scripts

When adding utility scripts to this folder:

1. ✅ **Prefer Python** for cross-platform compatibility
2. ✅ **Add shebang line** for Unix systems: `#!/usr/bin/env python3`
3. ✅ **Handle UTF-8 encoding** for emoji/Unicode on Windows
4. ✅ **Update this README** with usage instructions
5. ✅ **Add VS Code task** if the script should be easily accessible (`.vscode/tasks.json`)

## VS Code Integration

Scripts can be run via VS Code tasks:

- See `.vscode/tasks.json` for task definitions
- Run with `Ctrl+Shift+P` → "Tasks: Run Task"
- Tasks use `${workspaceFolder}` variable for portability
