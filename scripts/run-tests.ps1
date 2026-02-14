# Run tests with appropriate settings.
#
# Doc tests spawn many processes and load TableGen files, which can
# overwhelm the system. This script runs them with limited parallelism.
#
# Usage:
#   .\scripts\run-tests.ps1        # Run all tests (lib + doc)
#   .\scripts\run-tests.ps1 -Lib   # Run only library tests (fast)
#   .\scripts\run-tests.ps1 -Doc   # Run only doc tests (limited threads)
#   .\scripts\run-tests.ps1 -All   # Run all tests including ignored

param(
    [switch]$Lib,
    [switch]$Doc,
    [switch]$All
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Push-Location $ProjectDir

# Limit parallel doc test jobs to reduce system load
$DocTestJobs = if ($env:DOC_TEST_JOBS) { $env:DOC_TEST_JOBS } else { "2" }

try {
    if ($Lib) {
        Write-Host "Running library tests..."
        cargo test --lib
        if ($LASTEXITCODE -ne 0) { throw "Library tests failed" }
    }
    elseif ($Doc) {
        Write-Host "Running doc tests (jobs=$DocTestJobs)..."
        cargo test --doc -- --test-threads=$DocTestJobs
        if ($LASTEXITCODE -ne 0) { throw "Doc tests failed" }
    }
    else {
        # Default and -All: run both lib and doc tests
        Write-Host "Running library tests..."
        cargo test --lib
        if ($LASTEXITCODE -ne 0) { throw "Library tests failed" }

        Write-Host ""
        Write-Host "Running doc tests (jobs=$DocTestJobs)..."
        cargo test --doc -- --test-threads=$DocTestJobs
        if ($LASTEXITCODE -ne 0) { throw "Doc tests failed" }
    }

    Write-Host ""
    Write-Host "All tests completed."
}
finally {
    Pop-Location
}
