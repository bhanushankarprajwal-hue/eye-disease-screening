@echo off
title Eye Disease Screening
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during the install. Then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\setup_done.txt" (
    echo First run: setting up the app. This takes 5-10 minutes and needs internet...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installing the packages failed. See README.md, section 5.
        pause
        exit /b 1
    )
    echo done> ".venv\setup_done.txt"
)

echo Starting the app... it will open in your browser at http://localhost:8501
echo Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run Code\app.py
pause
