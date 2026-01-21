@echo off

if not exist "venv" (
    echo Virtual environment not found, please run setup_windows.bat first
    pause
    exit /b 1
)

echo Starting YouDub GUI...
venv\Scripts\python.exe gui.py

pause