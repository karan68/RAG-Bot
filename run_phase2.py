"""
Phase 2 Runner Script
Builds vector index and launches the chatbot
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import chromadb
    except ImportError:
        missing.append('chromadb')
    
    try:
        import ollama
    except ImportError:
        missing.append('ollama')
    
    try:
        import gradio
    except ImportError:
        missing.append('gradio')
    
    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_ollama_models():
    """Check if required Ollama models are available."""
    try:
        import ollama
        
        models = ollama.list()
        model_names = [m.get('name', m.get('model', '')).split(':')[0] for m in models.get('models', [])]
        
        required = ['phi3.5', 'nomic-embed-text']
        missing = [m for m in required if m not in model_names]
        
        if missing:
            print(f"Missing Ollama models: {missing}")
            print("Pull them with:")
            for m in missing:
                print(f"  ollama pull {m}")
            return False
        
        return True
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
        print("Make sure Ollama is running (ollama serve)")
        return False


def build_index():
    """Build the vector index."""
    from vector_index import VectorIndexBuilder
    
    base_path = Path(__file__).parent
    documents_path = base_path / 'data' / 'device_documents.json'
    persist_directory = base_path / 'data' / 'chromadb'
    
    if not documents_path.exists():
        print(f"Error: Documents not found at {documents_path}")
        print("Run run_phase1.py first")
        return False
    
    builder = VectorIndexBuilder(
        documents_path=str(documents_path),
        persist_directory=str(persist_directory)
    )
    
    builder.load_documents()
    builder.initialize_database()
    indexed = builder.build_index()
    
    print(f"\n✓ Indexed {indexed} documents")
    
    # Test search
    print("\nTesting search...")
    results = builder.search("gaming laptop with good graphics", n_results=3)
    print("Sample results for 'gaming laptop with good graphics':")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['metadata'].get('device_name')} (similarity: {r['similarity']:.3f})")
    
    return True


def launch_app():
    """Launch the Gradio chat application."""
    from app import DeviceChatbot
    
    chatbot = DeviceChatbot()
    chatbot.initialize()
    demo = chatbot.build_ui()
    
    print("\n" + "=" * 60)
    print("LAUNCHING DEVICE CHATBOT")
    print("=" * 60)
    print("\nOpen http://localhost:7860 in your browser")
    print("Press Ctrl+C to stop\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )


def main():
    """Run Phase 2 setup and launch."""
    print("=" * 60)
    print("PHASE 2: RAG INFRASTRUCTURE & CHATBOT")
    print("=" * 60)
    
    # Check dependencies
    print("\n[1/4] Checking dependencies...")
    if not check_dependencies():
        return 1
    print("✓ All Python dependencies installed")
    
    # Check Ollama models
    print("\n[2/4] Checking Ollama models...")
    if not check_ollama_models():
        return 1
    print("✓ Ollama models available (phi3.5, nomic-embed-text)")
    
    # Check if index exists or needs to be built
    base_path = Path(__file__).parent
    chromadb_path = base_path / 'data' / 'chromadb'
    
    print("\n[3/4] Setting up vector index...")
    if chromadb_path.exists():
        print("Vector index already exists. Rebuilding...")
    
    if not build_index():
        return 1
    
    # Launch app
    print("\n[4/4] Launching chatbot...")
    launch_app()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
