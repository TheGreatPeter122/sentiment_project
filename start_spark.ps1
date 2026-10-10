param([string]$Script = 'scripts/spark_data_analysis.py')
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
if (Test-Path 'S:/') { throw 'S: is already in use. Remove that mapping before running.' }
try {
    subst S: $workspace
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create temporary S: path mapping.' }
    & 'S:/sentiment_project/run.ps1' -Script $Script
    $code = $LASTEXITCODE
} finally {
    if (Test-Path 'S:/') { subst S: /D | Out-Null }
}
exit $code
