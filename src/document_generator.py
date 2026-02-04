"""
Document Generator for Device Specifications
Converts processed device JSON to searchable text documents for embedding
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class DocumentGenerator:
    """Generates searchable text documents from processed device data."""
    
    def __init__(self, processed_path: str, rules_path: str, output_path: str):
        self.processed_path = processed_path
        self.rules_path = rules_path
        self.output_path = output_path
        self.devices = []
        self.rules = {}
        self.documents = []
        
    def load_data(self):
        """Load processed devices and inference rules."""
        with open(self.processed_path, 'r', encoding='utf-8') as f:
            self.devices = json.load(f)
        print(f"Loaded {len(self.devices)} processed devices")
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
        print(f"Loaded inference rules with {len(self.rules.get('rules', {}))} categories")
        
    def evaluate_condition(self, device: dict, condition: dict) -> bool:
        """Evaluate a single condition against device specs."""
        for field, check in condition.items():
            device_value = device.get(field)
            
            if isinstance(check, bool):
                if device_value != check:
                    return False
            elif isinstance(check, dict):
                if 'gte' in check and (device_value is None or device_value < check['gte']):
                    return False
                if 'gt' in check and (device_value is None or device_value <= check['gt']):
                    return False
                if 'lte' in check and (device_value is None or device_value > check['lte']):
                    return False
                if 'lt' in check and (device_value is None or device_value >= check['lt']):
                    return False
                if 'in' in check and device_value not in check['in']:
                    return False
                if 'eq' in check and device_value != check['eq']:
                    return False
            elif device_value != check:
                return False
                
        return True
    
    def get_capability(self, device: dict, capability: str) -> Optional[dict]:
        """Get capability assessment for a device."""
        rules = self.rules.get('rules', {}).get(capability, [])
        
        for rule in rules:
            conditions = rule.get('conditions', {})
            if self.evaluate_condition(device, conditions):
                return {
                    'capability': capability,
                    'level': rule['result']['level'],
                    'message': rule['result']['message']
                }
        
        return None
    
    def get_all_capabilities(self, device: dict) -> dict:
        """Get all capability assessments for a device."""
        capabilities = {}
        
        for capability in self.rules.get('rules', {}).keys():
            result = self.get_capability(device, capability)
            if result:
                capabilities[capability] = result
                
        return capabilities
    
    def generate_spec_section(self, device: dict) -> str:
        """Generate hardware specifications section."""
        lines = []
        
        # Basic Info
        lines.append(f"Device: {device.get('device_name', 'Unknown')}")
        lines.append(f"Brand: {device.get('brand', 'Unknown')}")
        lines.append(f"Category: {device.get('category', 'Unknown')}")
        lines.append(f"Form Factor: {device.get('form_factor', 'Unknown')}")
        
        if device.get('title'):
            lines.append(f"Full Name: {device['title']}")
        
        # OS
        if device.get('os'):
            lines.append(f"Operating System: {device['os']}")
        
        # Processor
        if device.get('cpu_model'):
            cpu_line = f"Processor: {device['cpu_model']}"
            if device.get('cpu_tier'):
                cpu_line += f" ({device['cpu_tier']} tier)"
            lines.append(cpu_line)
        
        # Memory
        if device.get('ram_gb'):
            ram_line = f"Memory (RAM): {int(device['ram_gb'])} GB"
            if device.get('ram_type'):
                ram_line += f" {device['ram_type']}"
            lines.append(ram_line)
        
        # Storage
        if device.get('storage_gb'):
            storage_gb = device['storage_gb']
            if storage_gb >= 1000:
                storage_line = f"Storage: {storage_gb/1000:.1f} TB"
            else:
                storage_line = f"Storage: {int(storage_gb)} GB"
            if device.get('storage_type'):
                storage_line += f" {device['storage_type']}"
            lines.append(storage_line)
        
        # Display
        if device.get('screen_inches'):
            display_line = f"Display: {device['screen_inches']} inch"
            if device.get('resolution_type'):
                display_line += f" {device['resolution_type']}"
            if device.get('is_touchscreen'):
                display_line += " Touchscreen"
            lines.append(display_line)
        
        # Graphics
        if device.get('primary_gpu'):
            gpu_line = f"Graphics: {device['primary_gpu']}"
            if device.get('has_dedicated_gpu'):
                gpu_line += " (Dedicated)"
            else:
                gpu_line += " (Integrated)"
            if device.get('gpu_tier'):
                gpu_line += f" - {device['gpu_tier']} tier"
            lines.append(gpu_line)
        
        # NPU / AI
        if device.get('has_npu'):
            npu_line = "NPU: Yes"
            if device.get('npu_tops'):
                npu_line += f" - {device['npu_tops']} TOPS"
            if device.get('is_copilot_plus'):
                npu_line += " (Copilot+ PC)"
            lines.append(npu_line)
        else:
            lines.append("NPU: No")
        
        # Battery
        if device.get('battery_hours'):
            lines.append(f"Battery Life: Up to {device['battery_hours']} hours")
        
        # Connectivity
        ports = []
        if device.get('hdmi_ports'):
            ports.append(f"{device['hdmi_ports']} HDMI")
        if device.get('thunderbolt_ports'):
            ports.append(f"{device['thunderbolt_ports']} Thunderbolt")
        usb_total = (device.get('usb30_a_ports', 0) + device.get('usb30_c_ports', 0) + 
                    device.get('usb31_a_ports', 0) + device.get('usb31_c_ports', 0))
        if usb_total:
            ports.append(f"{usb_total} USB")
        if ports:
            lines.append(f"Ports: {', '.join(ports)}")
        
        # Security
        security = []
        if device.get('has_windows_hello'):
            security.append("Windows Hello")
        if device.get('has_fingerprint'):
            security.append("Fingerprint Reader")
        if security:
            lines.append(f"Security: {', '.join(security)}")
        
        # Price
        if device.get('price_min') and device.get('price_max'):
            lines.append(f"Price Range: ${int(device['price_min'])} - ${int(device['price_max'])}")
        elif device.get('price_avg'):
            lines.append(f"Price: ~${int(device['price_avg'])}")
        
        return '\n'.join(lines)
    
    def generate_features_section(self, device: dict) -> str:
        """Generate features and primary activities section."""
        lines = []
        
        # Features
        features = device.get('features', [])
        if features:
            lines.append(f"Features: {', '.join(features)}")
        
        # Primary Activities
        activities = device.get('primary_activities', [])
        if activities:
            activity_names = {
                'justtheessentials': 'Basic Computing',
                'family': 'Family Use',
                'entertainment': 'Entertainment',
                'work': 'Work/Business',
                'school': 'Education',
                'casualgaming': 'Casual Gaming',
                'advancedgaming': 'Advanced Gaming',
                'multitasking': 'Multitasking',
                'creating': 'Content Creation',
                'photovideoediting': 'Photo/Video Editing'
            }
            named_activities = [activity_names.get(a, a) for a in activities]
            lines.append(f"Best For: {', '.join(named_activities)}")
        
        # Specific feature flags
        special = []
        if device.get('has_touchscreen'):
            special.append("Touchscreen")
        if device.get('supports_pen'):
            special.append("Pen/Stylus Support")
        if device.get('is_copilot_plus'):
            special.append("Copilot+ PC")
        if device.get('has_thunderbolt'):
            special.append("Thunderbolt Support")
        if device.get('has_long_battery'):
            special.append("Long Battery Life")
        
        if special:
            lines.append(f"Special Features: {', '.join(special)}")
        
        return '\n'.join(lines)
    
    def generate_capabilities_section(self, device: dict) -> str:
        """Generate capability assessments section."""
        capabilities = self.get_all_capabilities(device)
        
        if not capabilities:
            return ""
        
        lines = ["", "CAPABILITY ASSESSMENT:"]
        
        # Group by category for readability
        capability_order = [
            'gaming', 'video_editing', 'photo_editing', '3d_modeling',
            'programming', 'ai_ml_development', 'office_productivity',
            'streaming_content_creation', 'copilot_features',
            'multitasking', 'portability', 'virtual_machines',
            'ram_upgrade', 'gpu_upgrade', 'storage_upgrade'
        ]
        
        capability_labels = {
            'gaming': 'Gaming',
            'video_editing': 'Video Editing',
            'photo_editing': 'Photo Editing',
            '3d_modeling': '3D Modeling',
            'programming': 'Programming/Development',
            'ai_ml_development': 'AI/Machine Learning',
            'office_productivity': 'Office/Productivity',
            'streaming_content_creation': 'Streaming/Content Creation',
            'copilot_features': 'Copilot Features',
            'multitasking': 'Multitasking',
            'portability': 'Portability',
            'virtual_machines': 'Virtual Machines',
            'ram_upgrade': 'RAM Upgrade',
            'gpu_upgrade': 'GPU Upgrade',
            'storage_upgrade': 'Storage Upgrade'
        }
        
        for cap_key in capability_order:
            if cap_key in capabilities:
                cap = capabilities[cap_key]
                label = capability_labels.get(cap_key, cap_key)
                level = cap['level'].upper()
                lines.append(f"- {label}: {level}")
                lines.append(f"  {cap['message']}")
        
        return '\n'.join(lines)
    
    def generate_comparison_keywords(self, device: dict) -> str:
        """Generate keywords for comparison retrieval."""
        keywords = []
        
        # Brand comparisons
        keywords.append(f"compare {device.get('brand', '').lower()}")
        keywords.append(f"{device.get('brand', '').lower()} vs")
        
        # Category comparisons
        keywords.append(f"compare {device.get('category', '').lower()}")
        
        # Device name variations
        name = device.get('device_name', '').lower()
        keywords.append(f"compare {name}")
        keywords.append(f"{name} vs")
        keywords.append(f"better than {name}")
        keywords.append(f"is {name} good")
        
        # Price tier
        if device.get('price_tier'):
            keywords.append(f"{device['price_tier']} laptop")
            keywords.append(f"{device['price_tier']} pc")
        
        return ' | '.join(keywords)
    
    def generate_document(self, device: dict) -> dict:
        """Generate a complete searchable document for a device."""
        # Build sections
        spec_section = self.generate_spec_section(device)
        features_section = self.generate_features_section(device)
        capabilities_section = self.generate_capabilities_section(device)
        
        # Combine into full text
        full_text = f"""
{spec_section}

{features_section}
{capabilities_section}
""".strip()
        
        # Create document with metadata
        doc = {
            'id': device.get('sku_id', device.get('product_id', '')),
            'device_name': device.get('device_name', ''),
            'title': device.get('title', ''),
            'brand': device.get('brand', ''),
            'category': device.get('category', ''),
            'form_factor': device.get('form_factor', ''),
            'price_tier': device.get('price_tier', ''),
            'text': full_text,
            'keywords': self.generate_comparison_keywords(device),
            # Include key specs for filtering
            'specs': {
                'ram_gb': device.get('ram_gb'),
                'storage_gb': device.get('storage_gb'),
                'screen_inches': device.get('screen_inches'),
                'has_dedicated_gpu': device.get('has_dedicated_gpu'),
                'has_npu': device.get('has_npu'),
                'is_copilot_plus': device.get('is_copilot_plus'),
                'cpu_tier': device.get('cpu_tier'),
                'gpu_tier': device.get('gpu_tier'),
                'price_min': device.get('price_min'),
                'price_max': device.get('price_max')
            },
            # Include capabilities summary
            'capabilities': {k: v['level'] for k, v in self.get_all_capabilities(device).items()}
        }
        
        return doc
    
    def generate_all_documents(self) -> list:
        """Generate documents for all devices."""
        if not self.devices:
            self.load_data()
        
        self.documents = []
        
        for device in self.devices:
            try:
                doc = self.generate_document(device)
                self.documents.append(doc)
            except Exception as e:
                print(f"Error generating document for {device.get('sku_id', 'unknown')}: {e}")
                continue
        
        print(f"Generated {len(self.documents)} documents")
        return self.documents
    
    def save_documents(self) -> str:
        """Save documents to output file."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        print(f"Saved documents to {self.output_path}")
        return self.output_path
    
    def save_text_documents(self, output_dir: str) -> str:
        """Save individual text files for each device (useful for debugging)."""
        os.makedirs(output_dir, exist_ok=True)
        
        for doc in self.documents:
            device_id = doc['id'] or doc['device_name'].replace(' ', '_')
            filename = f"{device_id}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc['text'])
        
        print(f"Saved {len(self.documents)} text files to {output_dir}")
        return output_dir


def main():
    """Main function to generate documents."""
    # Paths
    base_path = Path(__file__).parent.parent
    processed_path = base_path / 'data' / 'processed_devices.json'
    rules_path = base_path / 'inference_rules.json'
    output_path = base_path / 'data' / 'device_documents.json'
    text_output_dir = base_path / 'data' / 'text_documents'
    
    # Check if processed data exists
    if not processed_path.exists():
        print(f"Error: Processed data not found at {processed_path}")
        print("Run data_processor.py first to generate processed_devices.json")
        return
    
    # Generate documents
    generator = DocumentGenerator(str(processed_path), str(rules_path), str(output_path))
    generator.load_data()
    generator.generate_all_documents()
    generator.save_documents()
    
    # Optionally save text files
    # generator.save_text_documents(str(text_output_dir))
    
    # Print sample document
    if generator.documents:
        print("\n=== Sample Document ===")
        sample = generator.documents[0]
        print(f"ID: {sample['id']}")
        print(f"Device: {sample['device_name']}")
        print(f"\n{sample['text'][:2000]}...")
        print(f"\nCapabilities: {sample['capabilities']}")


if __name__ == '__main__':
    main()
