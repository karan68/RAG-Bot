"""
Device Chatbot Launcher
This is the main entry point for the application.
Checks for required models on first run and launches the UI.
"""

import sys
import os

# Add src to path
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    app_dir = os.path.dirname(sys.executable)
else:
    # Running as script
    app_dir = os.path.dirname(os.path.abspath(__file__))

src_dir = os.path.join(app_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

os.chdir(app_dir)


def show_setup_gui():
    """Show a GUI for first-time setup with model download progress."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    import threading
    from src.model_manager import (
        check_ollama_installed, 
        ensure_models_available,
        get_required_models
    )
    
    root = tk.Tk()
    root.title("Device Chatbot - First Time Setup")
    root.geometry("500x350")
    root.resizable(False, False)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (500 // 2)
    y = (root.winfo_screenheight() // 2) - (350 // 2)
    root.geometry(f"+{x}+{y}")
    
    # Main frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)
    
    # Title
    title_label = ttk.Label(
        main_frame, 
        text="🖥️ Device Chatbot Setup",
        font=("Segoe UI", 16, "bold")
    )
    title_label.pack(pady=(0, 10))
    
    # Info label
    info_label = ttk.Label(
        main_frame,
        text="This app requires AI models to run locally.\nFirst-time download is about 1.3 GB.",
        font=("Segoe UI", 10),
        justify="center"
    )
    info_label.pack(pady=(0, 20))
    
    # Model checklist frame
    checklist_frame = ttk.LabelFrame(main_frame, text="Required Components", padding=10)
    checklist_frame.pack(fill="x", pady=(0, 20))
    
    # Create checkmarks for each model
    model_vars = {}
    model_labels = {}
    
    # Ollama check
    ollama_var = tk.StringVar(value="⏳")
    ollama_label = ttk.Label(checklist_frame, textvariable=ollama_var, font=("Segoe UI", 10))
    ollama_label.pack(anchor="w")
    model_vars["ollama"] = ollama_var
    
    # Model checks
    for model in get_required_models():
        var = tk.StringVar(value=f"⏳ {model['display_name']} ({model['size_mb']} MB)")
        label = ttk.Label(checklist_frame, textvariable=var, font=("Segoe UI", 10))
        label.pack(anchor="w")
        model_vars[model["name"]] = var
        model_labels[model["name"]] = model
    
    # Progress bar
    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100, length=400)
    progress_bar.pack(pady=(0, 10))
    
    # Status label
    status_var = tk.StringVar(value="Click 'Install' to begin setup...")
    status_label = ttk.Label(main_frame, textvariable=status_var, font=("Segoe UI", 9))
    status_label.pack(pady=(0, 20))
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack()
    
    setup_complete = [False]
    
    def update_model_status(name, success):
        if name == "ollama":
            if success:
                model_vars["ollama"].set("✅ Ollama installed")
            else:
                model_vars["ollama"].set("❌ Ollama not found")
        elif name in model_vars:
            model_info = model_labels.get(name, {"display_name": name})
            if success:
                model_vars[name].set(f"✅ {model_info.get('display_name', name)}")
            else:
                model_vars[name].set(f"❌ {model_info.get('display_name', name)}")
    
    def run_setup():
        install_btn.config(state="disabled")
        progress_var.set(10)
        
        # Check Ollama
        status_var.set("Checking Ollama installation...")
        if check_ollama_installed():
            update_model_status("ollama", True)
            progress_var.set(20)
        else:
            update_model_status("ollama", False)
            status_var.set("Please install Ollama from https://ollama.ai")
            messagebox.showerror(
                "Ollama Required",
                "Ollama is not installed.\n\nPlease download and install from:\nhttps://ollama.ai\n\nThen restart this application."
            )
            install_btn.config(state="normal")
            return
        
        def progress_callback(msg):
            status_var.set(msg)
            # Update progress based on message
            if "qwen" in msg.lower():
                progress_var.set(40)
                if "ready" in msg.lower() or "✓" in msg:
                    update_model_status("qwen2.5:1.5b", True)
                    progress_var.set(70)
            elif "nomic" in msg.lower():
                progress_var.set(75)
                if "ready" in msg.lower() or "✓" in msg:
                    update_model_status("nomic-embed-text", True)
                    progress_var.set(95)
            elif "All models" in msg:
                progress_var.set(100)
        
        def do_setup():
            success, message = ensure_models_available(progress_callback)
            
            def finish():
                if success:
                    progress_var.set(100)
                    status_var.set("✅ Setup complete! Starting app...")
                    for name in model_vars:
                        if name != "ollama":
                            update_model_status(name, True)
                    setup_complete[0] = True
                    root.after(1500, root.destroy)
                else:
                    status_var.set(f"❌ {message}")
                    install_btn.config(state="normal")
                    messagebox.showerror("Setup Failed", message)
            
            root.after(0, finish)
        
        thread = threading.Thread(target=do_setup)
        thread.daemon = True
        thread.start()
    
    def cancel():
        root.destroy()
        sys.exit(0)
    
    install_btn = ttk.Button(button_frame, text="Install Models", command=run_setup, width=15)
    install_btn.pack(side="left", padx=5)
    
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=cancel, width=15)
    cancel_btn.pack(side="left", padx=5)
    
    root.mainloop()
    return setup_complete[0]


def check_first_run():
    """Check if models need to be downloaded."""
    from src.model_manager import check_all_models, check_ollama_installed
    
    if not check_ollama_installed():
        return True
    
    all_available, _, missing = check_all_models()
    return not all_available


def main():
    """Main entry point."""
    print("=" * 50)
    print("   Device Chatbot - Local AI Assistant")
    print("=" * 50)
    
    # Check if first run setup needed
    if check_first_run():
        print("\nFirst time setup required...")
        if not show_setup_gui():
            print("Setup cancelled or failed.")
            sys.exit(1)
    
    print("\nStarting application...")
    
    # Import and run the app
    from src.app import demo
    from src.config import config
    
    host = config.get("app.host", "127.0.0.1")
    port = config.get("app.port", 7860)
    
    print(f"\n🚀 Starting Device Chatbot at http://{host}:{port}")
    print("   Press Ctrl+C to stop\n")
    
    # Launch browser automatically
    import webbrowser
    webbrowser.open(f"http://{host}:{port}")
    
    # Start the app
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=False  # We already opened browser
    )


if __name__ == "__main__":
    main()
