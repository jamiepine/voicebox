@echo off
start "VoiceBox Server" "%~dp0start_server.bat"
timeout /t 6 >nul
start "VoiceBox UI" "%~dp0start_ui.bat"