# UV Environment Health Check
# Runs on VS Code startup to verify environment is clean and synced

$ErrorActionPreference = "Continue"
$hasIssues = $false

Write-Host "🔍 Checking UV Environment Health..." -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
try {
    $uvVersion = uv --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ UV installed: $uvVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ UV not found - install from https://docs.astral.sh/uv/" -ForegroundColor Red
    $hasIssues = $true
    exit 1
}

# Check if Python virtual environment exists
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment exists (.venv)" -ForegroundColor Green
} else {
    Write-Host "⚠️  Virtual environment not found" -ForegroundColor Yellow
    Write-Host "   Creating virtual environment..." -ForegroundColor Yellow
    uv venv
    $hasIssues = $true
}

# Check if uv.lock is in sync with pyproject.toml
Write-Host ""
Write-Host "Checking dependency sync..." -ForegroundColor Cyan

$syncCheck = uv sync --dry-run 2>&1
if ($LASTEXITCODE -eq 0) {
    # Check if output indicates no changes needed
    if ($syncCheck -like "*Already up-to-date*" -or $syncCheck -like "*Audited*") {
        Write-Host "✅ Dependencies are in sync" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Dependencies need syncing" -ForegroundColor Yellow
        Write-Host "   Run: uv sync" -ForegroundColor Yellow
        $hasIssues = $true
    }
} else {
    Write-Host "❌ Failed to check dependency sync" -ForegroundColor Red
    $hasIssues = $true
}

# Check Python version
try {
    $pythonVersion = & .venv/Scripts/python.exe --version 2>&1
    if ($pythonVersion -match "Python (3\.\d+\.\d+)") {
        $version = $matches[1]
        if ($version -ge "3.11.0") {
            Write-Host "✅ Python version: $version (>= 3.11 required)" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Python version: $version (3.11+ recommended)" -ForegroundColor Yellow
            $hasIssues = $true
        }
    }
} catch {
    Write-Host "⚠️  Could not verify Python version" -ForegroundColor Yellow
}

# Quick ruff check (non-blocking, just informational)
Write-Host ""
Write-Host "Running quick linting check..." -ForegroundColor Cyan
try {
    $ruffOutput = uv run ruff check --quiet week1/*.py week2/*.py week3/*.py week4/*.py week5/*.py week7/*.py week8/*.py 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ No Python linting errors found" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Python linting issues detected - run 'uv run ruff check .' for details" -ForegroundColor Yellow
        # Don't mark as has issues - linting errors are expected during development
    }
} catch {
    Write-Host "ℹ️  Ruff check skipped (run manually: uv run ruff check .)" -ForegroundColor Gray
}


# Summary
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
if (-not $hasIssues) {
    Write-Host "✅ Environment is healthy and ready!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some issues detected - please review above" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Quick fixes:" -ForegroundColor Cyan
    Write-Host "  • Sync dependencies: uv sync" -ForegroundColor White
    Write-Host "  • Update dependencies: uv lock --upgrade && uv sync" -ForegroundColor White
    Write-Host "  • Fix linting: uv run ruff check --fix ." -ForegroundColor White
}
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
