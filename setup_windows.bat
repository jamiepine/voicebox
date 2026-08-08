@echo off
rem ============================================================
rem VoiceBox - one-time Windows setup (run from source)
rem Install manually first: Python 3.10-3.12, Git, Bun, Rust (rustup)
rem and Visual Studio Build Tools with "Desktop development with C++"
rem (required by the Tauri UI and to build pyopenjtalk).
rem ============================================================
cd /d "%~dp0"
if errorlevel 1 exit /b 1

rem --- backend virtual environment ---
if not exist backend\venv\Scripts\activate.bat (
    echo Creating backend venv...
    python -m venv backend\venv
    if errorlevel 1 exit /b 1
)
if not exist "%~dp0backend\venv\Scripts\python.exe" (
    echo ERROR: backend venv was not created.
    exit /b 1
)
call backend\venv\Scripts\activate.bat
if errorlevel 1 exit /b 1
"%~dp0backend\venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

rem --- PyTorch with Intel XPU support (NVIDIA users: use cu126 index) ---
"%~dp0backend\venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
if errorlevel 1 exit /b 1

rem --- backend dependencies ---
rem If pip fails on pyopenjtalk: install VS Build Tools (see above)
rem or remove ",ja" from misaki extras in backend\requirements.txt
"%~dp0backend\venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
if errorlevel 1 exit /b 1

rem --- Tauri UI dependencies ---
cd tauri
if errorlevel 1 exit /b 1
bun install
if errorlevel 1 (
    echo ERROR: bun install failed. Is Bun installed?
    exit /b 1
)
cd ..
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Double-click start_voicebox.bat to launch.
pause