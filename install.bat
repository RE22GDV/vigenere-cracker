@echo off
REM One-shot setup for Windows: virtual environment, dependencies, language data.
chcp 65001 >nul
cd /d "%~dp0"
setlocal

echo ============================================================
echo  Vigenere Cracker - setup
echo ============================================================

where python >nul 2>&1
if errorlevel 1 (
    echo [!] Python was not found in PATH.
    echo     Install Python 3.9+ from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/4] Creating the virtual environment...
    python -m venv .venv || goto :fail
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Installing dependencies...
call .venv\Scripts\python.exe -m pip install --upgrade pip >nul
call .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail

echo.
echo [3/4] Optional: CUDA support (about 2.5 GB, ~10x faster brute force).
set /p CUDA="    Install PyTorch with CUDA? [y/N]: "
if /i "%CUDA%"=="y" (
    call .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo [4/4] Language data (word lists and corpora, about 280 MB).
set /p DATA="    Download now? [Y/n]: "
if /i not "%DATA%"=="n" (
    call .venv\Scripts\python.exe -m vigenere.download_data || goto :fail
)

echo.
echo ============================================================
echo  Done. Start the app with run_gui.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [!] Setup failed, see the messages above.
pause
exit /b 1
