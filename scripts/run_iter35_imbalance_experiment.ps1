<#
.SYNOPSIS
    Single-Command All-in-One Runner for ITER-35 Class-Imbalance Experiment on Mentor GPU.

.DESCRIPTION
    Executes all 4 Phase 1 loss formulations sequentially on Suturing (8 LOUO Folds),
    generates the comparative summary, and automatically commits and pushes all
    generated artifacts and checkpoints to GitHub origin main.

.EXAMPLE
    # Run the entire Phase 1 experiment with one command:
    .\scripts\run_iter35_imbalance_experiment.ps1

    # Rapid smoke test (1 epoch on 1 fold to verify setup in ~1 min):
    .\scripts\run_iter35_imbalance_experiment.ps1 -SmokeTest
#>

param (
    [switch]$SmokeTest,
    [switch]$NoPush,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  ITER-35 ALL-IN-ONE CLASS-IMBALANCE RUNNER (MENTOR GPU)                  " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# Locate Python in virtual environment
$PythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$ScriptPath = Join-Path $PSScriptRoot "run_iter35_imbalance_experiment.py"

$ArgsList = @()
if ($SmokeTest) { $ArgsList += "--smoke-test" }
if ($NoPush) { $ArgsList += "--no-push" }
if ($PreflightOnly) { $ArgsList += "--preflight-only" }

& $PythonExe $ScriptPath @ArgsList
if ($LASTEXITCODE -ne 0) {
    Write-Error "ITER-35 experiment execution failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "`n[SUCCESS] ITER-35 experiment workflow completed and pushed to GitHub main." -ForegroundColor Green
