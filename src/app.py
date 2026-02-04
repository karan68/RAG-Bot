"""
Device Specification Chatbot - Gradio UI
Interactive chat interface for device queries
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    gr = None

import sys
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import create_pipeline, RAGPipeline
from validators import validate_input, InputValidator
from config import get_config


class DeviceChatbot:
    """Gradio-based chatbot interface for device specifications."""
    
    def __init__(self):
        self.pipeline: Optional[RAGPipeline] = None
        self.device_list: List[str] = []
        self.config = get_config()
        self.base_path = self.config.base_path
        self.validator = InputValidator(
            max_length=self.config.get('validation.max_query_length', 500),
            min_length=self.config.get('validation.min_query_length', 2),
            blocked_patterns=self.config.get('validation.blocked_patterns', [])
        )
        
    def initialize(self):
        """Initialize the RAG pipeline and load device list."""
        print("Initializing Device Chatbot...")
        
        # Create RAG pipeline
        self.pipeline = create_pipeline(str(self.base_path))
        
        # Load device list for dropdown
        processed_path = self.base_path / 'data' / 'processed_devices.json'
        with open(processed_path, 'r', encoding='utf-8') as f:
            devices = json.load(f)
        
        # Get unique device names
        self.device_list = sorted(set(
            d.get('device_name', '') for d in devices if d.get('device_name')
        ))
        
        # Also load test devices
        test_devices_path = self.base_path / 'test_devices'
        self.test_device_files = []
        if test_devices_path.exists():
            self.test_device_files = [
                f.name for f in test_devices_path.glob('*.json')
            ]
        
        print(f"Loaded {len(self.device_list)} devices")
        print(f"Found {len(self.test_device_files)} test device files")
        
        return self
    
    def set_device(self, device_name: str) -> str:
        """Set the current device context."""
        if not device_name:
            return "Please select a device"
        
        # Check if it's a test device file
        if device_name in self.test_device_files:
            filepath = self.base_path / 'test_devices' / device_name
            self.pipeline.load_current_device_from_file(str(filepath))
            return f"✓ Loaded test device: {device_name}"
        
        # Otherwise, look up in processed devices
        result = self.pipeline.set_current_device(device_name)
        if result:
            return f"✓ Current device set to: {result.get('device_name', device_name)}"
        else:
            return f"✗ Device not found: {device_name}"
    
    def chat(
        self,
        message: str,
        history: List[dict]
    ) -> Generator[Tuple[str, List[dict]], None, None]:
        """Process chat message with streaming and return response."""
        if not message.strip():
            yield "", history
            return
        
        # Validate input
        validation = self.validator.validate(message)
        if not validation.is_valid:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": validation.error_message})
            yield "", history
            return
        
        # Use cleaned input
        clean_message = validation.cleaned_input
        
        if not self.pipeline:
            history.append({"role": "user", "content": clean_message})
            history.append({"role": "assistant", "content": "Error: Pipeline not initialized"})
            yield "", history
            return
        
        if not self.pipeline.current_device:
            history.append({"role": "user", "content": clean_message})
            history.append({"role": "assistant", "content": "Please select a device first using the dropdown above."})
            yield "", history
            return
        
        # Add user message to history
        history.append({"role": "user", "content": clean_message})
        
        # Stream the response
        history.append({"role": "assistant", "content": "Thinking..."})
        
        try:
            for chunk in self.pipeline.chat_stream(clean_message):
                # Update the last assistant message with streamed content
                history[-1] = {"role": "assistant", "content": chunk['response']}
                yield "", history
                
                if chunk.get('done', False):
                    break
        except Exception as e:
            history[-1] = {"role": "assistant", "content": f"Error: {str(e)}"}
            yield "", history
            return
        
        yield "", history
    
    def chat_sync(
        self,
        message: str,
        history: List[dict]
    ) -> Tuple[str, List[dict]]:
        """Non-streaming chat for compatibility (used by tests)."""
        if not message.strip():
            return "", history
        
        # Validate input
        validation = self.validator.validate(message)
        if not validation.is_valid:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": validation.error_message})
            return "", history
        
        clean_message = validation.cleaned_input
        
        if not self.pipeline:
            history.append({"role": "user", "content": clean_message})
            history.append({"role": "assistant", "content": "Error: Pipeline not initialized"})
            return "", history
        
        if not self.pipeline.current_device:
            history.append({"role": "user", "content": clean_message})
            history.append({"role": "assistant", "content": "Please select a device first using the dropdown above."})
            return "", history
        
        # Process query (non-streaming)
        result = self.pipeline.chat(clean_message)
        response = result['response']
        
        # Update history
        history.append({"role": "user", "content": clean_message})
        history.append({"role": "assistant", "content": response})
        
        return "", history
    
    def get_device_info(self) -> str:
        """Get current device information."""
        if not self.pipeline or not self.pipeline.current_device:
            return "No device selected"
        
        device = self.pipeline.current_device
        name = device.get('device_name') or device.get('DeviceName', 'Unknown')
        brand = device.get('brand') or device.get('Brand', 'Unknown')
        category = device.get('category') or device.get('Category', 'Unknown')
        
        info = f"**{name}**\n"
        info += f"- Brand: {brand}\n"
        info += f"- Category: {category}\n"
        
        # Add key specs
        if device.get('ram_gb'):
            info += f"- RAM: {int(device['ram_gb'])} GB\n"
        if device.get('storage_gb'):
            storage = device['storage_gb']
            if storage >= 1000:
                info += f"- Storage: {storage/1000:.1f} TB\n"
            else:
                info += f"- Storage: {int(storage)} GB\n"
        if device.get('primary_gpu'):
            info += f"- GPU: {device['primary_gpu']}\n"
        if device.get('is_copilot_plus'):
            info += f"- ✨ Copilot+ PC\n"
        
        return info
    
    def clear_chat(self) -> Tuple[str, List]:
        """Clear chat history."""
        if self.pipeline:
            self.pipeline.reset_conversation()
        return "", []
    
    def build_ui(self) -> gr.Blocks:
        """Build the Gradio UI."""
        if gr is None:
            raise ImportError("Gradio not installed")
        
        # Custom CSS
        css = """
        .device-info {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        """
        
        self.css = css
        
        with gr.Blocks(
            title="Device Specification Assistant"
        ) as demo:
            
            # Header
            gr.Markdown(
                """
                # 💻 Device Specification Assistant
                Ask questions about your Windows PC's capabilities, compare devices, and get upgrade advice.
                """
            )
            
            with gr.Row():
                # Left column - Device selection
                with gr.Column(scale=1):
                    gr.Markdown("### Select Device")
                    
                    # Device dropdown - combine catalog and test devices
                    all_devices = self.device_list + ["---TEST DEVICES---"] + self.test_device_files
                    device_dropdown = gr.Dropdown(
                        choices=all_devices,
                        label="Choose a device",
                        value=self.test_device_files[0] if self.test_device_files else None,
                        interactive=True
                    )
                    
                    set_device_btn = gr.Button("Set Device", variant="primary")
                    device_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1
                    )
                    
                    device_info = gr.Markdown(
                        value="No device selected",
                        label="Current Device"
                    )
                
                # Right column - Chat
                with gr.Column(scale=2):
                    gr.Markdown("### Chat")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your question",
                            placeholder="Ask about specs, capabilities, comparisons, upgrades...",
                            scale=4,
                            lines=1
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
            
            # Example questions
            gr.Markdown("### Example Questions")
            gr.Examples(
                examples=[
                    "What are the specs of this device?",
                    "Can I run video editing software?",
                    "Is this good for gaming?",
                    "Can I upgrade the RAM?",
                    "Does this have Copilot features?",
                    "What is this device best for?",
                    "Can I add an external GPU?",
                    "How does this compare to a gaming laptop for video editing?"
                ],
                inputs=msg
            )
            
            # Event handlers
            def on_device_select(device_name):
                status = self.set_device(device_name)
                info = self.get_device_info()
                return status, info
            
            set_device_btn.click(
                fn=on_device_select,
                inputs=[device_dropdown],
                outputs=[device_status, device_info]
            )
            
            # Also set device on dropdown change
            device_dropdown.change(
                fn=on_device_select,
                inputs=[device_dropdown],
                outputs=[device_status, device_info]
            )
            
            submit_btn.click(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[msg, chatbot]
            )
        
        return demo


def main():
    """Launch the chatbot UI."""
    print("=" * 60)
    print("DEVICE SPECIFICATION CHATBOT")
    print("=" * 60)
    
    # Load config
    config = get_config()
    
    # Initialize chatbot
    chatbot = DeviceChatbot()
    chatbot.initialize()
    
    # Build and launch UI
    demo = chatbot.build_ui()
    
    host = config.server_host
    port = config.server_port
    
    print(f"\nLaunching Gradio interface...")
    print(f"Open http://{host}:{port} in your browser")
    print("Press Ctrl+C to stop")
    
    demo.launch(
        server_name=host,
        server_port=port,
        share=False
    )


if __name__ == '__main__':
    main()

