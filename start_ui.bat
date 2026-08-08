@echo off
title VoiceBox UI
cd /d "%~dp0"
bun run setup:dev
if errorlevel 1 (
    echo ERROR: bun run setup:dev failed. Is Bun installed?
    pause
    exit /b 1
)
cd tauri
if errorlevel 1 exit /b 1
bun run tauri dev
pause