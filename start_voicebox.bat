@echo off
rem curl.exe is required for the readiness probe
where curl.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: curl.exe not found. Update Windows, or run start_server.bat and start_ui.bat manually.
    pause
    exit /b 1
)
rem Launch backend first
start "VoiceBox Server" "%~dp0start_server.bat"

rem --- bounded readiness polling (up to ~60s) ---
set /a wait=0
:wait_loop
timeout /t 2 >nul
curl -s -o nul http://127.0.0.1:17493/
if not errorlevel 1 goto ready
set /a wait+=2
if %wait% lss 60 goto wait_loop
echo ERROR: backend not ready after 60s. Check the server window.
pause
exit /b 1

:ready
echo Backend ready on port 17493. Starting UI...
start "VoiceBox UI" "%~dp0start_ui.bat"