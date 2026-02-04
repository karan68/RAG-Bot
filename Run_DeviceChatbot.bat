@echo off
title Device Chatbot
cd /d "%~dp0"

echo ================================================
echo   Device Chatbot - Starting...
echo ================================================
echo.

:: Check Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed.
    echo Please install from https://ollama.ai
    pause
    exit /b 1
)

:: Try to start Ollama in background if not running
echo Checking Ollama server...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama server...
    start /min "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: Run the app
echo.
echo Starting Device Chatbot...
echo Browser will open automatically at http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the application.
echo.

python launcher.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed. See error above.
    pause
)
