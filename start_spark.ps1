param([string]$Script = 'scripts/spark_data_analysis.py')
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
if (Test-Path 'S:/') { throw 'S: 已被占用，请关闭相关映射后再运行。' }
try {
    subst S: $workspace
    if ($LASTEXITCODE -ne 0) { throw '无法创建临时短路径 S:。' }
    & 'S:/sentiment_project/run.ps1' -Script $Script
    $code = $LASTEXITCODE
} finally {
    if (Test-Path 'S:/') { subst S: /D | Out-Null }
}
exit $code
