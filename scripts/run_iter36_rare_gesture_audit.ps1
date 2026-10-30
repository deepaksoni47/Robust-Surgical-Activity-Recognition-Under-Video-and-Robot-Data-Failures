<#
.SYNOPSIS
    Single-Command All-in-One Runner for Rare-Gesture Prediction Audit on Mentor GPU.

.DESCRIPTION
    Loads trained checkpoints from all 4 Phase 1 loss formulations (ce, class_weighted, focal, cb_focal),
    exports all trial-level and fold-level frame predictions, computes per-class Precision/Recall/F1
    and confusion matrices, generates publication figures, and automatically commits and pushes all
    artifacts to GitHub origin main.

.EXAMPLE
    # One-click execution on mentor GPU:
    .\scripts\run_iter36_rare_gesture_audit.ps1

    # Fast smoke-test (Fold 1 only, no push):
    .\scripts\run_iter36_rare_gesture_audit.ps1 -SmokeTest
#>

param (
    [switch]$SmokeTest,
    [switch]$NoPush,
    [switch]$NoRetrain
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  ITER-36 RARE-GESTURE PREDICTION AUDIT & EXPORT (MENTOR GPU)             " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# Locate Python in virtual environment
$PythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$ScriptPath = Join-Path $PSScriptRoot "run_iter36_rare_gesture_audit.py"

$ArgsList = @()
if ($SmokeTest) { $ArgsList += "--smoke-test" }
if ($NoPush) { $ArgsList += "--no-push" }
if ($NoRetrain) { $ArgsList += "--no-retrain" }

& $PythonExe $ScriptPath @ArgsList
if ($LASTEXITCODE -ne 0) {
    Write-Error "Rare-gesture prediction audit failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "`n[SUCCESS] Rare-gesture prediction audit completed and pushed to GitHub main." -ForegroundColor Green
