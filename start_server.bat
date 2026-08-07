@echo off
title VoiceBox Server (XPU)
cd /d "%~dp0"
call backend\venv\Scripts\activate.bat
uvicorn backend.main:app --reload --port 17493
pause