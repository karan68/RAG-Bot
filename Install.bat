@echo off
echo ================================================
echo   Device Chatbot - Windows Installer
echo ================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or newer from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version

:: Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Ollama is not installed.
    echo Please install Ollama from https://ollama.ai
    echo The app will still install, but you need Ollama to run it.
    echo.
    pause
)

echo.
echo Installing Python dependencies...
pip install chromadb gradio ollama pydantic --quiet

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [OK] Dependencies installed

echo.
echo ================================================
echo   Installation Complete!
echo ================================================
echo.
echo To run the Device Chatbot:
echo   1. Make sure Ollama is running (ollama serve)
echo   2. Double-click "Run_DeviceChatbot.bat"
echo.
echo On first run, the app will download AI models (~1.3 GB)
echo.
pause
