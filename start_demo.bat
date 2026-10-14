@echo off
setlocal
cd /d "%~dp0"

if not exist "models\daily_review.joblib" (
    echo Model file is missing. Please download the complete project folder.
    goto failed
)
if not exist "models\daily_review.json" (
    echo Model version record is missing. Please download the complete project folder.
    goto failed
)

if not exist ".venv\Scripts\python.exe" (
    py -3.12 --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.12 is required. Install it, then double-click this file again.
        echo https://www.python.org/downloads/
        goto failed
    )
    echo Creating the Python environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -c "import numpy, sklearn, joblib, jieba" >nul 2>&1
if errorlevel 1 (
    echo Installing demo dependencies. Internet access is needed on the first run...
    ".venv\Scripts\python.exe" -m pip install -r requirements-demo.txt
    if errorlevel 1 goto failed
)

echo Starting the demo. Close this window to stop it.
".venv\Scripts\python.exe" app.py --open
if errorlevel 1 goto failed
exit /b 0

:failed
echo Startup failed. Please check the message above.
pause
exit /b 1
