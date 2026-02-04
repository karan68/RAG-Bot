"""
RAG Pipeline for Device Specification Chatbot
Combines retrieval, inference rules, and LLM generation
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import ollama
except ImportError:
    print("Ollama not installed. Run: pip install ollama")
    ollama = None

from vector_index import VectorIndexBuilder


class InferenceEngine:
    """Applies rule-based inference for capability assessment."""
    
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.rules = {}
        self.load_rules()
    
    def load_rules(self):
        """Load inference rules from JSON file."""
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
        print(f"Loaded {len(self.rules.get('rules', {}))} rule categories")
    
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
    
    def assess_all_capabilities(self, device: dict) -> dict:
        """Get all capability assessments for a device."""
        capabilities = {}
        for capability in self.rules.get('rules', {}).keys():
            result = self.get_capability(device, capability)
            if result:
                capabilities[capability] = result
        return capabilities


class RAGPipeline:
    """Main RAG pipeline for device chatbot."""
    
    def __init__(
        self,
        vector_index: VectorIndexBuilder,
        inference_engine: InferenceEngine,
        processed_devices_path: str,
        llm_model: str = "qwen2.5:1.5b"
    ):
        self.vector_index = vector_index
        self.inference_engine = inference_engine
        self.llm_model = llm_model
        self.processed_devices = {}
        self.device_brands = set()  # Extracted dynamically from database
        self.current_device = None
        self.conversation_history = []
        
        # Load processed devices for detailed lookups
        self._load_processed_devices(processed_devices_path)
    
    def _load_processed_devices(self, path: str):
        """Load processed devices for detailed spec lookups."""
        with open(path, 'r', encoding='utf-8') as f:
            devices = json.load(f)
        
        # Index by multiple keys for easy lookup
        for device in devices:
            sku = device.get('sku_id', '')
            name = device.get('device_name', '').lower()
            title = device.get('title', '').lower()
            brand = (device.get('brand') or device.get('Brand', '')).lower()
            
            if sku:
                self.processed_devices[sku] = device
            if name:
                self.processed_devices[name] = device
            if title:
                self.processed_devices[title] = device
            
            # Extract brand for dynamic brand list
            if brand and len(brand) > 2 and not brand.replace('.', '').isdigit():
                self.device_brands.add(brand)
            
            # Also extract first word of device name as potential brand/line
            if name:
                first_word = name.split()[0] if name.split() else ''
                # Filter out numbers and short words
                if first_word and len(first_word) > 2 and not first_word.replace('.', '').isdigit():
                    self.device_brands.add(first_word)
                # Extract model line (e.g., "dell xps" -> "xps", "hp pavilion" -> "pavilion")
                words = name.split()
                if len(words) >= 2:
                    second_word = words[1]
                    if len(second_word) > 2 and not second_word.replace('.', '').isdigit():
                        self.device_brands.add(second_word)
        
        print(f"Indexed {len(devices)} devices for lookup")
    
    def set_current_device(self, device_identifier: str) -> Optional[dict]:
        """Set the current device context."""
        # Try exact match first
        device = self.processed_devices.get(device_identifier.lower())
        
        if not device:
            # Try partial match
            for key, dev in self.processed_devices.items():
                if device_identifier.lower() in key.lower():
                    device = dev
                    break
        
        if device:
            self.current_device = device
            print(f"Current device set to: {device.get('device_name')}")
            return device
        
        print(f"Device not found: {device_identifier}")
        return None
    
    def load_current_device_from_file(self, filepath: str) -> Optional[dict]:
        """Load current device context from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.current_device = data.get('device', data)
        print(f"Loaded current device: {self.current_device.get('DeviceName', self.current_device.get('device_name'))}")
        return self.current_device
    
    def _extract_comparison_device(self, query: str) -> Optional[str]:
        """Extract the name of the device being compared to from the query."""
        query_lower = query.lower()
        
        # Use dynamically extracted brands from database
        device_brands = self.device_brands
        
        # Patterns to extract the "other" device in comparison queries
        # Order matters - more specific patterns first
        patterns = [
            # "against DEVICE" - very common pattern
            r'against\s+(?:the\s+|a\s+)?([a-z0-9][a-z0-9\s\-]+?)(?:\s+along|\s+for|\s+in\s+terms|\s+regarding|\?|$|\.)',
            # "with DEVICE" 
            r'(?:compare|comparing).*?(?:with|to)\s+(?:the\s+|a\s+)?([a-z0-9][a-z0-9\s\-]+?)(?:\s+along|\s+for|\s+in\s+terms|\s+regarding|\?|$|\.)',
            # "vs DEVICE" or "versus DEVICE"
            r'(?:vs\.?|versus)\s+(?:the\s+|a\s+)?([a-z0-9][a-z0-9\s\-]+?)(?:\s+along|\s+for|\s+in\s+terms|\s+regarding|\?|$|\.)',
            # "difference between X and DEVICE"
            r'difference\s+between\s+.+?\s+and\s+(?:the\s+|a\s+)?([a-z0-9][a-z0-9\s\-]+?)(?:\s+along|\s+for|\s+in\s+terms|\?|$|\.)',
            # "better/worse than DEVICE"
            r'(?:better|worse)\s+than\s+(?:the\s+|a\s+)?([a-z0-9][a-z0-9\s\-]+?)(?:\s+along|\s+for|\s+in\s+terms|\?|$|\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                device_name = match.group(1).strip()
                device_name = self._clean_device_name(device_name, device_brands)
                if device_name and len(device_name) > 3:
                    return device_name
        
        # Fallback: Look for known brand names followed by model info
        for brand in device_brands:
            pattern = rf'\b({brand}\s+[a-z0-9][a-z0-9\s\-]*?)(?:\s+along|\s+for|\s+in\s+terms|\s+regarding|\?|$|\.)'
            match = re.search(pattern, query_lower)
            if match:
                device_name = self._clean_device_name(match.group(1).strip(), device_brands)
                # Make sure it's not the "this device" being compared
                if device_name and len(device_name) > 3 and 'this' not in device_name:
                    return device_name
        
        return None
    
    def _clean_device_name(self, device_name: str, device_brands: List[str]) -> Optional[str]:
        """Clean up extracted device name by removing trailing noise."""
        if not device_name:
            return None
        
        # Words that indicate end of device name
        stop_phrases = [
            'along with', 'for price', 'for gaming', 'for video', 'for photo',
            'in terms of', 'regarding', 'about', 'including', 'with price',
            'along', 'price', 'specs', 'specification', 'features', 'pros',
            'cons', 'comparison', 'compare', 'performance'
        ]
        
        # Remove stop phrases from the end
        cleaned = device_name.lower().strip()
        for phrase in stop_phrases:
            if cleaned.endswith(phrase):
                cleaned = cleaned[:-len(phrase)].strip()
            # Also check with space prefix
            if f' {phrase}' in cleaned:
                cleaned = cleaned.split(f' {phrase}')[0].strip()
        
        # Remove trailing punctuation and whitespace
        cleaned = re.sub(r'[\s\?\.\,\!]+$', '', cleaned)
        
        # Validate: should contain at least one brand or look like a device name
        has_brand = any(brand in cleaned for brand in device_brands)
        has_model_number = bool(re.search(r'\d', cleaned))
        
        if has_brand or has_model_number or len(cleaned) > 5:
            return cleaned if cleaned else None
        
        return None
    
    def _find_device_by_name(self, device_name: str) -> Optional[dict]:
        """Find a device by name with fuzzy matching."""
        if not device_name:
            return None
        
        name_lower = device_name.lower().strip()
        
        # Try exact match first
        if name_lower in self.processed_devices:
            return self.processed_devices[name_lower]
        
        # Try partial match - device name contains the search term
        best_match = None
        best_score = 0
        
        for key, device in self.processed_devices.items():
            # Check if search term is in device key
            if name_lower in key:
                # Score based on how close the match is
                score = len(name_lower) / len(key)
                if score > best_score:
                    best_score = score
                    best_match = device
            # Check if device key is in search term
            elif key in name_lower:
                score = len(key) / len(name_lower)
                if score > best_score:
                    best_score = score
                    best_match = device
        
        # Also check DeviceName and device_name fields directly
        for key, device in self.processed_devices.items():
            dev_name = (device.get('device_name') or device.get('DeviceName', '')).lower()
            if name_lower in dev_name or dev_name in name_lower:
                # Prefer more specific matches
                score = min(len(name_lower), len(dev_name)) / max(len(name_lower), len(dev_name))
                if score > best_score:
                    best_score = score
                    best_match = device
        
        # Only return if we have a reasonable match (>50% similarity)
        if best_score > 0.3:
            return best_match
        
        return None
    
    def classify_query(self, query: str) -> dict:
        """Classify the type of query and extract intent."""
        query_lower = query.lower()
        
        classification = {
            'type': 'general',
            'intent': None,
            'entities': [],
            'requires_comparison': False,
            'requires_current_device': True,
            'comparison_device_name': None  # Name of device to compare to
        }
        
        # Check for comparison queries
        comparison_patterns = [
            r'compare|vs|versus|better than|worse than|difference between',
            r'which (one|is|has)|should i (get|buy|choose)'
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, query_lower):
                classification['type'] = 'comparison'
                classification['requires_comparison'] = True
                # Extract the comparison device name
                classification['comparison_device_name'] = self._extract_comparison_device(query)
                break
        
        # Check for capability queries
        capability_keywords = {
            'gaming': ['game', 'gaming', 'play games', 'fortnite', 'minecraft', 'aaa'],
            'video_editing': ['video edit', 'premiere', 'davinci', 'final cut', 'edit video'],
            'photo_editing': ['photo edit', 'photoshop', 'lightroom', 'edit photo'],
            '3d_modeling': ['3d', 'blender', 'maya', 'cad', 'modeling', 'render'],
            'programming': ['program', 'coding', 'development', 'ide', 'visual studio'],
            'ai_ml_development': ['machine learning', 'ai ', 'ml ', 'deep learning', 'neural'],
            'copilot_features': ['copilot', 'recall', 'npu', 'ai feature', 'studio effect'],
            'ram_upgrade': ['upgrade ram', 'add ram', 'more memory', 'upgrade memory'],
            'gpu_upgrade': ['upgrade gpu', 'add gpu', 'graphics card', 'upgrade graphics', 'external gpu', 'egpu'],
            'storage_upgrade': ['upgrade storage', 'add storage', 'more storage', 'add ssd'],
            'portability': ['portable', 'travel', 'carry', 'lightweight', 'battery life'],
            'multitasking': ['multitask', 'multiple apps', 'many programs'],
            'virtual_machines': ['virtual machine', 'vm', 'vmware', 'hyper-v', 'virtualbox']
        }
        
        for capability, keywords in capability_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    classification['type'] = 'capability'
                    classification['intent'] = capability
                    break
            if classification['intent']:
                break
        
        # Check for spec lookups
        spec_keywords = ['ram', 'memory', 'storage', 'processor', 'cpu', 'gpu', 'graphics',
                        'screen', 'display', 'battery', 'port', 'price', 'weight']
        for keyword in spec_keywords:
            if keyword in query_lower:
                if classification['type'] == 'general':
                    classification['type'] = 'spec_lookup'
                break
        
        # Check for out-of-scope
        out_of_scope_keywords = ['mac', 'macbook', 'iphone', 'android', 'samsung galaxy',
                                 'weather', 'stock', 'recipe', 'news']
        for keyword in out_of_scope_keywords:
            if keyword in query_lower:
                classification['type'] = 'out_of_scope'
                break
        
        return classification
    
    def retrieve_context(
        self,
        query: str,
        classification: dict,
        n_results: int = 3
    ) -> Tuple[str, List[dict]]:
        """Retrieve relevant context for the query."""
        contexts = []
        retrieved_devices = []
        
        # Always include current device if set
        if self.current_device and classification.get('requires_current_device', True):
            current_doc = self._format_device_context(self.current_device, is_current=True)
            contexts.append(current_doc)
            retrieved_devices.append(self.current_device)
        
        # For comparisons, find the specific device mentioned
        if classification['requires_comparison']:
            comparison_device_name = classification.get('comparison_device_name')
            comparison_device = None
            
            if comparison_device_name:
                # Try to find the specific device mentioned
                comparison_device = self._find_device_by_name(comparison_device_name)
            
            if comparison_device:
                # Check it's not the same as current device
                current_name = (self.current_device.get('device_name') or 
                               self.current_device.get('DeviceName', '')).lower()
                comparison_name = (comparison_device.get('device_name') or 
                                  comparison_device.get('DeviceName', '')).lower()
                
                if current_name != comparison_name:
                    contexts.append(self._format_device_context(comparison_device, is_current=False))
                    retrieved_devices.append(comparison_device)
            else:
                # Mark that comparison device was not found
                classification['comparison_device_not_found'] = True
                classification['comparison_device_requested'] = comparison_device_name
        
        # Add capability assessment if relevant
        if classification['type'] == 'capability' and classification['intent'] and self.current_device:
            capability = self.inference_engine.get_capability(
                self.current_device,
                classification['intent']
            )
            if capability:
                contexts.append(f"\n[CAPABILITY ASSESSMENT - {classification['intent'].upper()}]\n{capability['message']}\nLevel: {capability['level'].upper()}")
        
        combined_context = "\n\n".join(contexts)
        return combined_context, retrieved_devices
    
    def _format_device_context(self, device: dict, is_current: bool = False) -> str:
        """Format device data into context string."""
        prefix = "[CURRENT DEVICE]" if is_current else "[COMPARISON DEVICE]"
        
        # Handle both processed format and raw format
        name = device.get('device_name') or device.get('DeviceName', 'Unknown Device')
        brand = device.get('brand') or device.get('Brand', 'Unknown')
        category = device.get('category') or device.get('Category', '')
        
        lines = [f"{prefix}", f"Device: {name}", f"Brand: {brand}"]
        if category:
            lines.append(f"Category: {category}")
        
        # Add specs - handle both formats
        if 'Specification' in device:
            # Raw format from test devices
            spec = device['Specification']
            if spec.get('CPU'):
                lines.append(f"Processor: {spec['CPU']}")
            if spec.get('Memory'):
                lines.append(f"RAM: {spec['Memory']}")
            if spec.get('Storage'):
                storage = spec['Storage']
                lines.append(f"Storage: {storage} SSD")
            if spec.get('GPU'):
                lines.append(f"Integrated GPU: {spec['GPU']}")
            if spec.get('dGPU') and spec.get('dGPU') != '':
                lines.append(f"Dedicated GPU: {spec['dGPU']} (discrete graphics)")
            elif spec.get('HasdGPU') == 'No':
                lines.append(f"Dedicated GPU: None (integrated graphics only)")
            if spec.get('ScreenSize'):
                lines.append(f"Display: {spec['ScreenSize']}")
            if spec.get('BatteryLife'):
                lines.append(f"Battery: {spec['BatteryLife']}")
            # Thunderbolt for test devices
            if spec.get('ThunderboltPorts') and int(spec.get('ThunderboltPorts', 0)) > 0:
                lines.append(f"Thunderbolt: Yes ({spec['ThunderboltPorts']} ports) - supports eGPU")
            elif spec.get('ThunderboltPorts') == '0':
                lines.append(f"Thunderbolt: No - cannot connect external GPU (eGPU)")
            # NPU for test devices
            if spec.get('NGAINPU') and spec.get('NGAINPU') != '':
                lines.append(f"NPU: {spec['NGAINPU']} - enables AI features like Recall, Live Captions, Cocreator")
            
            # Is it a Copilot+ PC?
            if device.get('IsNPU') == 'Y' or 'Copilot+' in str(device.get('Feature', [])) or category == 'Copilot+ PC':
                lines.append(f"Copilot+ PC: Yes - full Windows Copilot+ features available")
            
            # Price for test devices
            if device.get('PriceRange'):
                pr = device['PriceRange']
                lines.append(f"Price: ${pr.get('MinPrice', '')} - ${pr.get('MaxPrice', '')}")
            elif device.get('HardCodedPrice'):
                lines.append(f"Price: {device['HardCodedPrice']}")
            
            # Upgrade info for test devices
            if spec.get('InternalMemoryType'):
                mem_type = spec['InternalMemoryType']
                if 'LPDDR' in mem_type:
                    lines.append(f"RAM Upgrade: Not possible - {mem_type} is soldered to motherboard")
                else:
                    lines.append(f"RAM Type: {mem_type} - check manufacturer specs for upgrade options")
            
            # Ports for test devices
            ports = []
            if spec.get('HDMIPorts') and int(spec.get('HDMIPorts', 0)) > 0:
                ports.append(f"HDMI ({spec['HDMIPorts']})")
            if spec.get('USBPorts') and int(spec.get('USBPorts', 0)) > 0:
                ports.append(f"USB ({spec['USBPorts']})")
            if spec.get('USB30CPorts') and int(spec.get('USB30CPorts', 0)) > 0:
                ports.append(f"USB-C ({spec['USB30CPorts']})")
            if ports:
                lines.append(f"Ports: {', '.join(ports)}")
            
            # Fingerprint reader for test devices
            if spec.get('FingerprintReader') == 'Yes':
                lines.append(f"Fingerprint Reader: Yes - biometric security available")
            elif spec.get('FingerprintReader') == 'No':
                lines.append(f"Fingerprint Reader: No")
            
            # Copilot+ PC status - be explicit about NO as well
            if device.get('IsNPU') == 'N' or (not spec.get('NGAINPU') or spec.get('NGAINPU') == ''):
                if 'Copilot+ PC: Yes' not in '\n'.join(lines):
                    lines.append(f"Copilot+ PC: No - does not have NPU for Copilot+ features")
        else:
            # Processed format
            if device.get('cpu_model'):
                lines.append(f"Processor: {device['cpu_model']} ({device.get('cpu_tier', '')} tier)")
            if device.get('ram_gb'):
                lines.append(f"RAM: {int(device['ram_gb'])} GB {device.get('ram_type', '')}")
            if device.get('storage_gb'):
                storage = device['storage_gb']
                if storage >= 1000:
                    lines.append(f"Storage: {storage/1000:.1f} TB {device.get('storage_type', '')}")
                else:
                    lines.append(f"Storage: {int(storage)} GB {device.get('storage_type', '')}")
            if device.get('primary_gpu'):
                gpu_type = "Dedicated" if device.get('has_dedicated_gpu') else "Integrated"
                lines.append(f"GPU: {device['primary_gpu']} ({gpu_type}, {device.get('gpu_tier', '')} tier)")
            if device.get('screen_inches'):
                lines.append(f"Display: {device['screen_inches']}\" {device.get('resolution_type', '')}")
            if device.get('battery_hours'):
                lines.append(f"Battery: Up to {device['battery_hours']} hours")
            if device.get('has_npu'):
                npu_info = "Yes"
                if device.get('npu_tops'):
                    npu_info += f" - {device['npu_tops']} TOPS"
                if device.get('is_copilot_plus'):
                    npu_info += " (Copilot+ PC)"
                lines.append(f"NPU: {npu_info}")
            if device.get('has_thunderbolt'):
                lines.append(f"Thunderbolt: Yes ({device.get('thunderbolt_ports', 1)} ports)")
            if device.get('price_min') and device.get('price_max'):
                lines.append(f"Price: ${int(device['price_min'])} - ${int(device['price_max'])}")
        
        # Add features
        features = device.get('features') or device.get('Feature', [])
        if features:
            lines.append(f"Features: {', '.join(features)}")
        
        activities = device.get('primary_activities') or device.get('PrimaryActivities', [])
        if activities:
            lines.append(f"Best For: {', '.join(activities)}")
        
        return "\n".join(lines)
    
    def build_prompt(
        self,
        query: str,
        context: str,
        classification: dict
    ) -> str:
        """Build the complete prompt for the LLM."""
        
        system_prompt = """You are a Windows PC specification assistant. Answer the user's question about their device.

CRITICAL RULES - FOLLOW EXACTLY:
1. Answer ONLY the single question asked - nothing more
2. DO NOT generate additional "User Question:" or "Answer:" sections
3. DO NOT create follow-up questions or continue the conversation
4. Give ONE direct answer then STOP IMMEDIATELY
5. Keep answers concise: 2-4 sentences for simple questions
6. ONLY use the device data provided below
7. If information is not provided, say "I don't have that specification"
8. NEVER mention Mac, iPhone, or devices not in the data
9. NEVER make up specifications

STOP after answering. Do not write anything else."""

        # Add type-specific instructions
        if classification['type'] == 'out_of_scope':
            system_prompt += "\n10. This is out-of-scope. Briefly explain you only help with Windows PC specs."
        
        if classification['type'] == 'comparison':
            system_prompt += "\n10. Compare devices briefly using specific numbers."
        
        prompt = f"""{system_prompt}

DEVICE DATA:
{context}

QUESTION: {query}

ANSWER (be brief, then stop):"""
        
        return prompt
    
    def _get_token_limit(self, classification: dict) -> int:
        """Get appropriate token limit based on query type."""
        query_type = classification.get('type', 'general')
        
        # Comparisons and detailed capability queries need more tokens
        if query_type == 'comparison':
            return 500  # Comparisons need ~350-500 tokens for pros/cons
        elif query_type == 'capability' and classification.get('intent') in [
            'gaming', 'video_editing', '3d_modeling', 'ai_ml_development'
        ]:
            return 350  # Capability assessments need more detail
        else:
            return 250  # Simple queries stay concise
    
    def generate_response(self, prompt: str, classification: dict = None) -> str:
        """Generate response using Ollama LLM."""
        if ollama is None:
            return "Error: Ollama not installed. Run: pip install ollama"
        
        # Determine token limit based on query type
        token_limit = self._get_token_limit(classification) if classification else 250
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.2,  # Low temperature for focused responses
                    'num_predict': token_limit,  # Dynamic limit based on query type
                    'top_p': 0.9,
                    'stop': ['User Question:', 'User:', 'Question:', 'Q:', '\n\nUser', '---', 'Follow-up:', 'Next question:']
                }
            )
            raw_response = response['message']['content']
            # Clean response to remove any leaked Q&A patterns
            return self._clean_response(raw_response)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_response_stream(self, prompt: str, classification: dict = None):
        """
        Generate response using Ollama LLM with streaming.
        Yields partial responses as they are generated.
        """
        if ollama is None:
            yield "Error: Ollama not installed. Run: pip install ollama"
            return
        
        # Determine token limit based on query type
        token_limit = self._get_token_limit(classification) if classification else 250
        
        try:
            stream = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.2,
                    'num_predict': token_limit,  # Dynamic limit based on query type
                    'top_p': 0.9,
                    'stop': ['User Question:', 'User:', 'Question:', 'Q:', '\n\nUser', '---']
                },
                stream=True  # Enable streaming
            )
            
            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    
                    # Check for patterns that indicate we should stop
                    should_stop = False
                    for pattern in ['User Question:', 'User:', 'Question:', '\n\nUser']:
                        if pattern in full_response:
                            # Trim at the pattern
                            full_response = full_response.split(pattern)[0].strip()
                            should_stop = True
                            break
                    
                    yield full_response
                    
                    if should_stop:
                        break
            
            # Final cleanup
            yield self._clean_response(full_response)
            
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def chat_stream(self, query: str):
        """
        Streaming version of chat - yields partial responses.
        """
        # Classify query
        classification = self.classify_query(query)
        
        # Handle out-of-scope
        if classification['type'] == 'out_of_scope':
            yield {
                'response': "I'm sorry, I can only help with Windows PC specifications and capabilities.",
                'classification': classification,
                'done': True
            }
            return
        
        # Check if current device is needed but not set
        if classification.get('requires_current_device', True) and not self.current_device:
            yield {
                'response': "Please select a device first. I need to know which device you're asking about.",
                'classification': classification,
                'done': True
            }
            return
        
        # Retrieve context
        context, retrieved_devices = self.retrieve_context(query, classification)
        
        # Handle comparison device not found
        if classification.get('comparison_device_not_found'):
            requested_device = classification.get('comparison_device_requested', 'that device')
            current_name = self.current_device.get('device_name') or self.current_device.get('DeviceName', 'your current device')
            yield {
                'response': f"I don't have '{requested_device}' in my database, so I can't compare it with {current_name}. Try comparing with a device from the dropdown list, or ask me about {current_name}'s specifications instead.",
                'classification': classification,
                'done': True
            }
            return
        
        # Build prompt
        prompt = self.build_prompt(query, context, classification)
        
        # Stream response with dynamic token limit
        for partial_response in self.generate_response_stream(prompt, classification):
            yield {
                'response': partial_response,
                'classification': classification,
                'done': False
            }
        
        # Final message
        yield {
            'response': partial_response,
            'classification': classification,
            'devices_retrieved': [d.get('device_name', d.get('DeviceName', 'Unknown')) for d in retrieved_devices],
            'done': True
        }
    
    def _clean_response(self, response: str) -> str:
        """Remove any extra Q&A patterns from the response."""
        import re
        
        # Patterns that indicate the model started generating extra Q&A
        cutoff_patterns = [
            r'\n\s*User Question:',
            r'\n\s*User:',
            r'\n\s*Question:',
            r'\n\s*Q:',
            r'\n\s*\d+\.\s*User',
            r'\n\s*---',
            r'\n\s*Next question',
            r'\n\s*Another question',
            r'\n\s*Follow-up',
            r'\n\s*Additionally,\s*User',
        ]
        
        for pattern in cutoff_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                response = response[:match.start()].strip()
        
        # Remove "Answer:" prefix if model echoed it
        if response.lower().startswith('answer:'):
            response = response[7:].strip()
        
        return response.strip()
    
    def chat(self, query: str) -> dict:
        """Main chat interface - process query and return response."""
        # Classify query
        classification = self.classify_query(query)
        
        # Handle out-of-scope
        if classification['type'] == 'out_of_scope':
            return {
                'response': "I'm sorry, I can only help with Windows PC specifications and capabilities. I don't have information about that topic.",
                'classification': classification,
                'context_used': None,
                'devices_retrieved': []
            }
        
        # Check if current device is needed but not set
        if classification.get('requires_current_device', True) and not self.current_device:
            return {
                'response': "Please set a current device first. I need to know which device you're asking about.",
                'classification': classification,
                'context_used': None,
                'devices_retrieved': []
            }
        
        # Retrieve context
        context, retrieved_devices = self.retrieve_context(query, classification)
        
        # Handle comparison device not found
        if classification.get('comparison_device_not_found'):
            requested_device = classification.get('comparison_device_requested', 'that device')
            current_name = self.current_device.get('device_name') or self.current_device.get('DeviceName', 'your current device')
            return {
                'response': f"I don't have '{requested_device}' in my database, so I can't compare it with {current_name}. Try comparing with a device from the dropdown list, or ask me about {current_name}'s specifications instead.",
                'classification': classification,
                'context_used': None,
                'devices_retrieved': [d.get('device_name', d.get('DeviceName', 'Unknown')) for d in retrieved_devices]
            }
        
        # Build prompt
        prompt = self.build_prompt(query, context, classification)
        
        # Generate response with dynamic token limit
        response = self.generate_response(prompt, classification)
        
        # Store in conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'classification': classification
        })
        
        return {
            'response': response,
            'classification': classification,
            'context_used': context,
            'devices_retrieved': [d.get('device_name', d.get('DeviceName', 'Unknown')) for d in retrieved_devices]
        }
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


def create_pipeline(base_path: Optional[str] = None) -> RAGPipeline:
    """Factory function to create a fully initialized RAG pipeline."""
    if base_path is None:
        base_path = Path(__file__).parent.parent
    else:
        base_path = Path(base_path)
    
    # Paths
    documents_path = base_path / 'data' / 'device_documents.json'
    processed_path = base_path / 'data' / 'processed_devices.json'
    rules_path = base_path / 'inference_rules.json'
    persist_directory = base_path / 'data' / 'chromadb'
    
    # Check required files
    for path in [documents_path, processed_path, rules_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Initialize vector index
    vector_index = VectorIndexBuilder(
        documents_path=str(documents_path),
        persist_directory=str(persist_directory)
    )
    
    # Load or build index
    if persist_directory.exists():
        try:
            vector_index.load_collection()
        except:
            print("Building new vector index...")
            vector_index.load_documents()
            vector_index.initialize_database()
            vector_index.build_index()
    else:
        print("Building vector index...")
        vector_index.load_documents()
        vector_index.initialize_database()
        vector_index.build_index()
    
    # Initialize inference engine
    inference_engine = InferenceEngine(str(rules_path))
    
    # Create pipeline
    pipeline = RAGPipeline(
        vector_index=vector_index,
        inference_engine=inference_engine,
        processed_devices_path=str(processed_path)
    )
    
    return pipeline


def main():
    """Interactive test of RAG pipeline."""
    print("=" * 60)
    print("DEVICE SPECIFICATION CHATBOT - RAG Pipeline Test")
    print("=" * 60)
    
    # Create pipeline
    print("\nInitializing pipeline...")
    pipeline = create_pipeline()
    
    # Set a test device
    test_device_path = Path(__file__).parent.parent / 'test_devices' / 'current_device_gaming_laptop.json'
    if test_device_path.exists():
        pipeline.load_current_device_from_file(str(test_device_path))
    else:
        # Try to set by name
        pipeline.set_current_device("HP Pavilion x360 14")
    
    print("\n" + "=" * 60)
    print("Ready! Type 'quit' to exit, 'device <name>' to change device")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            
            if query.lower().startswith('device '):
                device_name = query[7:].strip()
                pipeline.set_current_device(device_name)
                continue
            
            # Process query
            result = pipeline.chat(query)
            
            print(f"\nAssistant: {result['response']}")
            print(f"\n[Debug: Query type: {result['classification']['type']}, "
                  f"Intent: {result['classification'].get('intent', 'N/A')}]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == '__main__':
    main()
