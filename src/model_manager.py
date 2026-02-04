"""
Model Manager - Handles model availability and downloads
Checks for required models on startup and downloads if needed.
"""

import subprocess
import sys
import time
from typing import Tuple, List


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def start_ollama_server() -> bool:
    """Start Ollama server in background."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        time.sleep(3)  # Wait for server to start
        return check_ollama_running()
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False


def check_model_available(model_name: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        # Check if model name (without tag) is in output
        model_base = model_name.split(':')[0]
        return model_base in result.stdout
    except Exception:
        return False


def get_required_models() -> List[dict]:
    """Get list of required models with their details."""
    return [
        {
            "name": "qwen2.5:1.5b",
            "display_name": "Qwen 2.5 1.5B (Language Model)",
            "size_mb": 986,
            "purpose": "Answering questions about devices"
        },
        {
            "name": "nomic-embed-text",
            "display_name": "Nomic Embed Text (Embeddings)",
            "size_mb": 274,
            "purpose": "Searching device database"
        }
    ]


def check_all_models() -> Tuple[bool, str, List[str]]:
    """
    Check if all required models are available.
    Returns (all_available, status_message, missing_models)
    """
    # Check Ollama
    if not check_ollama_installed():
        return False, "Ollama not installed", ["ollama"]
    
    missing = []
    for model in get_required_models():
        if not check_model_available(model["name"]):
            missing.append(model["name"])
    
    if missing:
        return False, f"Missing models: {', '.join(missing)}", missing
    
    return True, "All models available", []


def download_model(model_name: str, progress_callback=None) -> bool:
    """
    Download a model using Ollama.
    
    Args:
        model_name: Name of the model to download
        progress_callback: Optional callback function(status_text)
    """
    try:
        if progress_callback:
            progress_callback(f"Downloading {model_name}...")
        
        # Use subprocess to show progress
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        # Read output
        for line in process.stdout:
            line = line.strip()
            if line and progress_callback:
                # Extract percentage if present
                if "%" in line:
                    progress_callback(f"Downloading {model_name}: {line}")
        
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return False


def ensure_models_available(progress_callback=None) -> Tuple[bool, str]:
    """
    Ensure all required models are available, download if needed.
    
    Args:
        progress_callback: Optional callback function(status_text)
    
    Returns:
        (success, message)
    """
    # Check Ollama installation
    if progress_callback:
        progress_callback("Checking Ollama installation...")
    
    if not check_ollama_installed():
        return False, "Ollama is not installed. Please install from https://ollama.ai"
    
    # Check if Ollama is running, start if not
    if progress_callback:
        progress_callback("Starting Ollama server...")
    
    if not check_ollama_running():
        if not start_ollama_server():
            return False, "Could not start Ollama server. Please start it manually."
    
    # Check models
    if progress_callback:
        progress_callback("Checking installed models...")
    
    all_available, status, missing = check_all_models()
    
    if all_available:
        if progress_callback:
            progress_callback("All models ready!")
        return True, "All models available"
    
    # Download missing models
    required = get_required_models()
    total_size = sum(m["size_mb"] for m in required if m["name"] in missing)
    
    if progress_callback:
        progress_callback(f"Downloading {len(missing)} model(s) (~{total_size} MB)...")
    
    for model in required:
        if model["name"] in missing:
            if progress_callback:
                progress_callback(f"Downloading {model['display_name']} ({model['size_mb']} MB)...")
            
            if not download_model(model["name"], progress_callback):
                return False, f"Failed to download {model['display_name']}"
            
            if progress_callback:
                progress_callback(f"✓ {model['display_name']} ready")
    
    if progress_callback:
        progress_callback("All models downloaded successfully!")
    
    return True, "All models downloaded and ready"


def install_ollama_windows() -> bool:
    """Attempt to install Ollama on Windows using winget."""
    try:
        print("Installing Ollama...")
        result = subprocess.run(
            ["winget", "install", "Ollama.Ollama", "--accept-source-agreements", "--accept-package-agreements"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Could not install Ollama automatically: {e}")
        print("Please install manually from https://ollama.ai")
        return False


if __name__ == "__main__":
    # Test model availability
    print("Checking model availability...")
    
    def print_status(msg):
        print(f"  {msg}")
    
    success, message = ensure_models_available(print_status)
    print(f"\nResult: {message}")
