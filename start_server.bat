@echo off
title VoiceBox Server (XPU)
cd /d "%~dp0"
if not exist "%~dp0backend\venv\Scripts\python.exe" (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
"%~dp0backend\venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --port 17493
pause