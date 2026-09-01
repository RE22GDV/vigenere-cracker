@echo off
REM Launch the graphical interface, preferring the local virtual environment.
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m vigenere.gui
) else (
    start "" pythonw -m vigenere.gui
    if errorlevel 1 python -m vigenere.gui
)
