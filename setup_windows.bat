@echo off
rem ============================================================
rem VoiceBox - one-time Windows setup (run from source)
rem Install manually first: Python 3.10-3.12, Git, Bun, Rust (rustup)
rem and Visual Studio Build Tools with "Desktop development with C++"
rem (required by the Tauri UI and to build pyopenjtalk).
rem ============================================================
cd /d "%~dp0"

rem --- backend virtual environment ---
if not exist backend\venv\Scripts\activate.bat (
    echo Creating backend venv...
    python -m venv backend\venv
)
call backend\venv\Scripts\activate.bat
python -m pip install --upgrade pip

rem --- PyTorch with Intel XPU support (NVIDIA users: use cu126 index) ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

rem --- backend dependencies ---
rem If pip fails on pyopenjtalk: install VS Build Tools (see above)
rem or remove ",ja" from misaki extras in backend\requirements.txt
pip install -r backend\requirements.txt

echo.
echo Setup complete. Double-click start_voicebox.bat to launch.
pause