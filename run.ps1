param([string]$Script = 'scripts/smoke_spark.py')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $env:JAVA_HOME) {
    $javaCandidate = Join-Path $PSScriptRoot 'runtime/jdk-21.0.12.1+1'
    if (Test-Path "$javaCandidate/bin/java.exe") { $env:JAVA_HOME = $javaCandidate }
    else { throw '请准备完整 Java 17/21 并配置 JAVA_HOME。' }
}
$env:PATH = "$env:JAVA_HOME/bin;$env:PATH"
$env:SPARK_HOME = "$PSScriptRoot/.venv/Lib/site-packages/pyspark"
$env:PYSPARK_PYTHON = "$PSScriptRoot/.venv/Scripts/python.exe"
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:PYTHONIOENCODING = 'utf-8'
$env:SPARK_LOCAL_IP = '127.0.0.1'
& $env:PYSPARK_PYTHON $Script
exit $LASTEXITCODE
