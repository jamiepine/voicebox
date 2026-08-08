@echo off
title VoiceBox UI
cd /d "%~dp0tauri"
bun run tauri dev
pause