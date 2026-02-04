"""
Build script for creating Device Chatbot installer.

This creates a Windows executable using PyInstaller.
The executable will include all necessary data files.
Models are downloaded on first run via Ollama.

Requirements:
    pip install pyinstaller

Usage:
    python build_installer.py
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


# Configuration
APP_NAME = "DeviceChatbot"
APP_VERSION = "1.0.0"
MAIN_SCRIPT = "launcher.py"

# Directories
BASE_DIR = Path(__file__).parent
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"

# Data files to include
DATA_FILES = [
    ("config.json", "."),
    ("devices.json", "."),
    ("data/chromadb", "data/chromadb"),
    ("data/processed_devices.json", "data"),
    ("data/device_documents.json", "data"),
    ("inference_rules.json", "."),
    ("src", "src"),
]

# Hidden imports that PyInstaller might miss
HIDDEN_IMPORTS = [
    "chromadb",
    "chromadb.api",
    "chromadb.config",
    "chromadb.db",
    "chromadb.segment",
    "chromadb.telemetry",
    "gradio",
    "ollama",
    "pydantic",
    "httpx",
    "anyio",
    "starlette",
    "fastapi",
    "uvicorn",
    "tiktoken",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]

# Modules to exclude (conflicts, unnecessary)
EXCLUDE_MODULES = [
    "PyQt5",
    "PyQt6",
    "PySide2",
    # Keep PySide6 if gradio uses it, or exclude it too if not needed
]


def check_requirements():
    """Check if required tools are installed."""
    print("Checking requirements...")
    
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  ✓ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  ✗ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Check if data directories exist
    required_dirs = ["data/chromadb"]
    required_files = ["data/processed_devices.json", "data/device_documents.json", "inference_rules.json"]
    
    for dir_path in required_dirs:
        full_path = BASE_DIR / dir_path
        if not full_path.exists():
            print(f"  ⚠ Missing {dir_path} - Run the app first to generate data")
            return False
        print(f"  ✓ {dir_path} exists")
    
    for file_path in required_files:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            print(f"  ⚠ Missing {file_path} - Run data processing first")
            return False
        print(f"  ✓ {file_path} exists")
    
    return True


def clean_build():
    """Clean previous build artifacts."""
    print("\nCleaning previous build...")
    
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed {dir_path}")


def build_executable():
    """Build the executable using PyInstaller."""
    print("\nBuilding executable...")
    
    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onedir",  # Create a folder with dependencies
        "--windowed",  # No console window
        "--noconfirm",  # Overwrite without asking
        "--clean",  # Clean cache
        # "--icon", "assets/icon.ico",  # Uncomment if you have an icon
    ]
    
    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", imp])
    
    # Add excluded modules to avoid conflicts
    for mod in EXCLUDE_MODULES:
        cmd.extend(["--exclude-module", mod])
    
    # Add data files
    for src, dst in DATA_FILES:
        src_path = BASE_DIR / src
        if src_path.exists():
            separator = ";" if sys.platform == "win32" else ":"
            cmd.extend(["--add-data", f"{src_path}{separator}{dst}"])
            print(f"  Adding: {src} -> {dst}")
    
    # Add main script
    cmd.append(str(BASE_DIR / MAIN_SCRIPT))
    
    print(f"\nRunning PyInstaller...")
    print(f"  Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def create_installer_readme():
    """Create a README for the installer."""
    readme_content = f"""# Device Chatbot v{APP_VERSION}

A local AI-powered chatbot for querying device specifications.

## Requirements

- **Ollama**: Required for running AI models locally
  - Download from: https://ollama.ai
  - Install before running Device Chatbot

## First Run Setup

1. Install Ollama from https://ollama.ai
2. Run DeviceChatbot.exe
3. The app will download required AI models (~1.3 GB total):
   - Qwen 2.5 1.5B (986 MB) - Language model
   - Nomic Embed Text (274 MB) - Embeddings
4. After download completes, the chatbot UI will open

## Usage

Simply run DeviceChatbot.exe to start the chatbot.

Ask questions like:
- "What are the specs of HP Spectre x360?"
- "Can the Dell XPS 15 handle video editing?"
- "Compare Surface Pro with MacBook Air"
- "Which laptops support Copilot+ features?"

## Troubleshooting

**"Ollama not found" error:**
- Make sure Ollama is installed and in your PATH
- Restart your computer after installing Ollama

**Slow first response:**
- First response may take 10-20 seconds as models load
- Subsequent responses will be faster

**Model download fails:**
- Check your internet connection
- Try running: ollama pull qwen2.5:1.5b
- Try running: ollama pull nomic-embed-text

## Files

- DeviceChatbot.exe - Main application
- config.json - Configuration settings
- devices.json - Device database
- data/ - Vector database and processed data

## Support

For issues, please check:
1. Ollama is running (ollama serve)
2. Models are downloaded (ollama list)
3. Port 7860 is available

---
Built with ❤️ using local AI
"""
    
    readme_path = DIST_DIR / APP_NAME / "README.txt"
    readme_path.write_text(readme_content)
    print(f"  Created {readme_path}")


def create_batch_launcher():
    """Create a batch file to launch the app with console for debugging."""
    batch_content = f"""@echo off
echo Starting Device Chatbot...
echo.
cd /d "%~dp0"
start "" "{APP_NAME}.exe"
"""
    
    batch_path = DIST_DIR / APP_NAME / f"Launch_{APP_NAME}.bat"
    batch_path.write_text(batch_content)
    print(f"  Created {batch_path}")
    
    # Also create a debug version that shows console
    debug_content = f"""@echo off
echo Starting Device Chatbot (Debug Mode)...
echo.
cd /d "%~dp0"
"{APP_NAME}.exe"
pause
"""
    
    debug_path = DIST_DIR / APP_NAME / f"Debug_{APP_NAME}.bat"
    debug_path.write_text(debug_content)
    print(f"  Created {debug_path}")


def create_zip_package():
    """Create a ZIP package for distribution."""
    print("\nCreating ZIP package...")
    
    output_name = f"{APP_NAME}_v{APP_VERSION}_Windows"
    output_path = DIST_DIR / output_name
    
    shutil.make_archive(
        str(output_path),
        'zip',
        DIST_DIR,
        APP_NAME
    )
    
    print(f"  Created {output_path}.zip")
    return f"{output_path}.zip"


def print_summary(zip_path):
    """Print build summary."""
    print("\n" + "=" * 50)
    print("BUILD COMPLETE")
    print("=" * 50)
    
    # Get folder size
    folder_path = DIST_DIR / APP_NAME
    total_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"""
Output:
  Folder: {folder_path}
  ZIP: {zip_path}
  Size: {size_mb:.1f} MB (before models)

Required Models (downloaded on first run):
  - qwen2.5:1.5b (986 MB)
  - nomic-embed-text (274 MB)
  Total: ~1.3 GB

To distribute:
  1. Share the ZIP file
  2. User extracts and runs {APP_NAME}.exe
  3. First run downloads AI models automatically

Note: Users need Ollama installed (https://ollama.ai)
""")


def main():
    """Main build process."""
    print("=" * 50)
    print(f"Building {APP_NAME} v{APP_VERSION}")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Build failed: Missing requirements")
        sys.exit(1)
    
    clean_build()
    
    if not build_executable():
        print("\n❌ Build failed: PyInstaller error")
        sys.exit(1)
    
    print("\nCreating additional files...")
    create_installer_readme()
    create_batch_launcher()
    
    zip_path = create_zip_package()
    
    print_summary(zip_path)
    
    print("✅ Build successful!")


if __name__ == "__main__":
    main()
