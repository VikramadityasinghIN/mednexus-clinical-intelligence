@echo off
REM Clinical Intelligence — one-click Windows installer
setlocal

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org (tick "Add to PATH").
    pause
    exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist .venv python -m venv .venv

echo [3/4] Installing packages (this takes a few minutes)...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo NOTE: some optional packages failed. The app still runs on the core set.
    echo You can retry heavy ones later, e.g.:
    echo   .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo [4/4] Training models on synthetic data...
.venv\Scripts\python scripts\train_all.py

echo.
echo DONE. Start the dashboard with:  launch.bat
echo Or the API with:  .venv\Scripts\uvicorn api.server:app --port 8123
pause
