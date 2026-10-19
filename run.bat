@echo off
title AI Second-Hand Item Analyzer (Localhost)
echo ========================================================
echo   Launching AI Second-Hand Item Analyzer on Localhost...
echo ========================================================
python run.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server exited with an error. Ensure Python 3.10+ is installed and on PATH.
    pause
)
