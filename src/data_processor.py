"""
Data Processor for Device Specifications
Cleans, normalizes, and extracts structured data from devices.json
"""

import json
import re
import os
from typing import Any, Optional
from pathlib import Path


class DeviceDataProcessor:
    """Processes and normalizes device specification data."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.devices = []
        self.processed_devices = []
        
    def load_data(self) -> list:
        """Load the raw devices.json file."""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.devices = json.load(f)
        print(f"Loaded {len(self.devices)} devices from {self.input_path}")
        return self.devices
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Remove HTML tags like <sup>7</sup>, <sup>**</sup>, etc.
        cleaned = re.sub(r'<[^>]+>', '', str(text))
        return cleaned.strip()
    
    def extract_number(self, text: str, default: Optional[float] = None) -> Optional[float]:
        """Extract numeric value from text like 'Memory: Up to 16 GB' -> 16."""
        if not text:
            return default
        # Handle "Up to X" patterns - extract the number
        text = str(text)
        # Find all numbers (including decimals)
        numbers = re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            # Return the first significant number (usually the main value)
            return float(numbers[0])
        return default
    
    def extract_label_value(self, text: str) -> str:
        """Extract value from 'Label: Value' format."""
        if not text:
            return ""
        text = self.clean_html(str(text))
        # Remove common prefixes like "Memory:", "Storage:", etc.
        patterns = [
            r'^(OS|Memory|Storage|Screen size|CPU|GPU|HDMI ports|USB ports|'
            r'CPU family|Storage media|Fingerprint reader|Product type|'
            r'HDtype|Has GPU|Internal memory type):\s*',
            r'^USB \d+\.\d+ \([\d\.]+ Gen \d+\) Type-[AC] ports:\s*'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def parse_ram(self, spec: dict) -> dict:
        """Parse RAM-related fields."""
        memory_text = spec.get('Memory', '')
        memory_type = spec.get('InternalMemoryType', '')
        
        return {
            'ram_gb': self.extract_number(memory_text),
            'ram_text': self.extract_label_value(memory_text),
            'ram_type': self.extract_label_value(memory_type),
            'ram_is_ddr5': 'DDR5' in str(memory_type).upper(),
            'ram_is_lpddr': 'LPDDR' in str(memory_type).upper()
        }
    
    def parse_storage(self, spec: dict) -> dict:
        """Parse storage-related fields."""
        storage_text = spec.get('Storage', '')
        storage_media = spec.get('StorageMedia', '')
        
        # Extract storage value
        storage_value = self.extract_number(storage_text)
        storage_text_clean = self.extract_label_value(storage_text)
        
        # Convert TB to GB for consistency
        storage_gb = storage_value
        if storage_value and 'TB' in str(storage_text).upper():
            storage_gb = storage_value * 1000
        
        return {
            'storage_gb': storage_gb,
            'storage_text': storage_text_clean,
            'storage_type': self.extract_label_value(storage_media),
            'is_ssd': 'SSD' in str(storage_text).upper() or 'SSD' in str(storage_media).upper(),
            'is_hdd': 'HDD' in str(storage_text).upper() or 'HDD' in str(storage_media).upper()
        }
    
    def parse_display(self, spec: dict, device: dict) -> dict:
        """Parse display-related fields."""
        screen_size = spec.get('ScreenSize', '')
        hd_type = spec.get('HDtype', '')
        has_4k = spec.get('Has4K', '')
        touchscreen = spec.get('Touchscreen', '')
        
        # Check features for touch
        features = device.get('Feature', []) or []
        has_touch = touchscreen == 'Y' or 'Touch' in features
        
        return {
            'screen_inches': self.extract_number(screen_size),
            'screen_text': self.extract_label_value(screen_size),
            'resolution_type': self.extract_label_value(hd_type),
            'is_4k': has_4k == 'Y' or '4K' in str(hd_type).upper() or 'UHD' in str(hd_type).upper(),
            'is_fhd': 'Full HD' in str(hd_type) or 'FHD' in str(hd_type).upper(),
            'is_touchscreen': has_touch,
            'supports_pen': 'Ink' in features
        }
    
    def parse_cpu(self, spec: dict) -> dict:
        """Parse CPU-related fields."""
        cpu_text = spec.get('CPU', '')
        cpu_family = spec.get('CPUFamily', '')
        
        cpu_clean = self.extract_label_value(cpu_text)
        family_clean = self.extract_label_value(cpu_family)
        
        # Determine CPU tier
        cpu_tier = 'unknown'
        cpu_upper = cpu_clean.upper()
        if 'I9' in cpu_upper or 'RYZEN 9' in cpu_upper or 'ULTRA 9' in cpu_upper:
            cpu_tier = 'high-end'
        elif 'I7' in cpu_upper or 'RYZEN 7' in cpu_upper or 'ULTRA 7' in cpu_upper:
            cpu_tier = 'performance'
        elif 'I5' in cpu_upper or 'RYZEN 5' in cpu_upper or 'ULTRA 5' in cpu_upper:
            cpu_tier = 'mainstream'
        elif 'I3' in cpu_upper or 'RYZEN 3' in cpu_upper:
            cpu_tier = 'entry'
        elif 'CELERON' in cpu_upper or 'PENTIUM' in cpu_upper or 'ATHLON' in cpu_upper:
            cpu_tier = 'budget'
        elif 'SNAPDRAGON' in cpu_upper:
            if 'ELITE' in cpu_upper or 'X PLUS' in cpu_upper:
                cpu_tier = 'performance'
            else:
                cpu_tier = 'mainstream'
        
        # Determine brand
        cpu_brand = 'unknown'
        if 'INTEL' in cpu_upper or 'CORE' in cpu_upper:
            cpu_brand = 'Intel'
        elif 'AMD' in cpu_upper or 'RYZEN' in cpu_upper:
            cpu_brand = 'AMD'
        elif 'SNAPDRAGON' in cpu_upper or 'QUALCOMM' in cpu_upper:
            cpu_brand = 'Qualcomm'
        elif 'APPLE' in cpu_upper:
            cpu_brand = 'Apple'
            
        return {
            'cpu_model': cpu_clean,
            'cpu_family': family_clean,
            'cpu_brand': cpu_brand,
            'cpu_tier': cpu_tier
        }
    
    def parse_gpu(self, spec: dict) -> dict:
        """Parse GPU-related fields."""
        gpu_text = spec.get('GPU', '')
        dgpu_text = spec.get('dGPU', '')
        has_gpu = spec.get('HasGPU', '')
        has_dgpu = spec.get('HasdGPU', '')
        
        gpu_clean = self.extract_label_value(gpu_text)
        dgpu_clean = self.extract_label_value(dgpu_text)
        
        # Determine if has dedicated GPU
        has_dedicated = bool(dgpu_clean) or has_dgpu == 'Y'
        
        # Determine GPU tier
        gpu_tier = 'integrated'
        primary_gpu = dgpu_clean if dgpu_clean else gpu_clean
        gpu_upper = primary_gpu.upper()
        
        if has_dedicated:
            if 'RTX 40' in gpu_upper or 'RTX 50' in gpu_upper:
                if '90' in gpu_upper or '80' in gpu_upper:
                    gpu_tier = 'enthusiast'
                elif '70' in gpu_upper:
                    gpu_tier = 'high-end'
                elif '60' in gpu_upper:
                    gpu_tier = 'performance'
                else:
                    gpu_tier = 'entry-dedicated'
            elif 'RTX 30' in gpu_upper:
                if '90' in gpu_upper or '80' in gpu_upper:
                    gpu_tier = 'high-end'
                elif '70' in gpu_upper or '60' in gpu_upper:
                    gpu_tier = 'performance'
                else:
                    gpu_tier = 'entry-dedicated'
            elif 'GTX' in gpu_upper:
                gpu_tier = 'entry-dedicated'
            elif 'RADEON RX' in gpu_upper:
                gpu_tier = 'performance'
            elif 'QUADRO' in gpu_upper or 'RTX A' in gpu_upper:
                gpu_tier = 'workstation'
            else:
                gpu_tier = 'entry-dedicated'
        else:
            if 'IRIS XE' in gpu_upper or 'IRIS PLUS' in gpu_upper:
                gpu_tier = 'integrated-performance'
            elif 'ADRENO' in gpu_upper:
                gpu_tier = 'integrated-mobile'
            else:
                gpu_tier = 'integrated-basic'
        
        return {
            'gpu_model': gpu_clean,
            'dgpu_model': dgpu_clean,
            'has_dedicated_gpu': has_dedicated,
            'gpu_tier': gpu_tier,
            'primary_gpu': primary_gpu
        }
    
    def parse_npu(self, spec: dict, device: dict) -> dict:
        """Parse NPU/AI-related fields."""
        npu_text = spec.get('NGAINPU', '')
        is_npu = device.get('IsNPU', 'N')
        category = device.get('Category', '')
        
        # Extract TOPS if available
        tops = self.extract_number(npu_text)
        
        return {
            'has_npu': is_npu == 'Y' or 'Copilot+ PC' in category,
            'npu_tops': tops,
            'npu_text': self.clean_html(npu_text),
            'is_copilot_plus': is_npu == 'Y' or 'Copilot+ PC' in category or (tops and tops >= 40)
        }
    
    def parse_connectivity(self, spec: dict) -> dict:
        """Parse port and connectivity fields."""
        return {
            'hdmi_ports': int(self.extract_number(spec.get('HDMIPorts', ''), 0) or 0),
            'usb_ports_total': int(self.extract_number(spec.get('USBPorts', ''), 0) or 0),
            'usb30_a_ports': int(self.extract_number(spec.get('USB30APorts', ''), 0) or 0),
            'usb30_c_ports': int(self.extract_number(spec.get('USB30CPorts', ''), 0) or 0),
            'usb31_a_ports': int(self.extract_number(spec.get('USB31APorts', ''), 0) or 0),
            'usb31_c_ports': int(self.extract_number(spec.get('USB31CPorts', ''), 0) or 0),
            'thunderbolt_ports': int(self.extract_number(spec.get('ThunderboltPorts', ''), 0) or 0),
            'has_thunderbolt': bool(spec.get('ThunderboltPorts', '')),
            'has_fingerprint': spec.get('FingerprintReader', '') == 'Y' or 'fingerprint' in str(spec.get('FingerprintReader', '')).lower(),
            'has_sim': bool(spec.get('SIM', '')),
            'has_esim': bool(spec.get('eSIM', ''))
        }
    
    def parse_battery(self, spec: dict) -> dict:
        """Parse battery-related fields."""
        battery_text = spec.get('BatteryLife', '')
        battery_clean = self.clean_html(battery_text)
        hours = self.extract_number(battery_clean)
        
        return {
            'battery_hours': hours,
            'battery_text': battery_clean
        }
    
    def parse_price(self, device: dict) -> dict:
        """Parse price-related fields."""
        price_range = device.get('PriceRange', {}) or {}
        hardcoded = device.get('HardCodedPrice', '')
        
        min_price = self.extract_number(str(price_range.get('Min', '')))
        max_price = self.extract_number(str(price_range.get('Max', '')))
        
        # Determine price tier
        price_tier = 'unknown'
        avg_price = None
        if min_price and max_price:
            avg_price = (min_price + max_price) / 2
        elif min_price:
            avg_price = min_price
        elif max_price:
            avg_price = max_price
            
        if avg_price:
            if avg_price < 500:
                price_tier = 'budget'
            elif avg_price < 800:
                price_tier = 'entry'
            elif avg_price < 1200:
                price_tier = 'mainstream'
            elif avg_price < 1800:
                price_tier = 'premium'
            else:
                price_tier = 'luxury'
        
        return {
            'price_min': min_price,
            'price_max': max_price,
            'price_avg': avg_price,
            'price_tier': price_tier,
            'price_text': hardcoded
        }
    
    def parse_features(self, device: dict) -> dict:
        """Parse feature flags and primary activities."""
        features = device.get('Feature', []) or []
        activities = device.get('PrimaryActivities', []) or []
        
        return {
            'features': features,
            'primary_activities': activities,
            'has_windows_hello': 'Hello' in features,
            'has_touchscreen': 'Touch' in features,
            'has_pen_support': 'Ink' in features,
            'is_portable': 'Portability' in features,
            'has_long_battery': 'Longbatterylife' in features,
            'has_large_display': 'LargerDisplay' in features,
            'has_lots_storage': 'LotsOfStorage' in features,
            # Activity flags
            'good_for_gaming': 'casualgaming' in activities or 'advancedgaming' in activities,
            'good_for_advanced_gaming': 'advancedgaming' in activities,
            'good_for_work': 'work' in activities,
            'good_for_school': 'school' in activities,
            'good_for_creating': 'creating' in activities or 'photovideoediting' in activities,
            'good_for_video_editing': 'photovideoediting' in activities,
            'good_for_family': 'family' in activities,
            'good_for_entertainment': 'entertainment' in activities,
            'good_for_multitasking': 'multitasking' in activities
        }
    
    def parse_metadata(self, device: dict) -> dict:
        """Parse device metadata."""
        spec = device.get('Specification', {}) or {}
        
        # Determine form factor
        category = device.get('Category', '')
        product_type = spec.get('ProductType', '')
        
        form_factor = 'laptop'
        if '2-in-1' in category or 'Hybrid' in product_type:
            form_factor = '2-in-1'
        elif 'Desktop' in category or 'All-in-One' in category:
            if 'All-in-One' in category or 'AIO' in product_type:
                form_factor = 'all-in-one'
            else:
                form_factor = 'desktop'
        elif 'Copilot' in category:
            form_factor = 'laptop'  # Most Copilot+ PCs are laptops
            
        return {
            'sku_id': device.get('SKUID', ''),
            'device_name': device.get('DeviceName', ''),
            'brand': device.get('Brand', ''),
            'category': category,
            'form_factor': form_factor,
            'title': device.get('Title', ''),
            'series': device.get('Series', ''),
            'description': self.clean_html(device.get('Description', '')),
            'os': self.extract_label_value(spec.get('OS', '')),
            'colors': device.get('Colors', ''),
            'on_market_date': device.get('OnMarket', ''),
            'product_id': device.get('ProductID', ''),
            'show': device.get('Show', 'True') == 'True'
        }
    
    def process_device(self, device: dict) -> dict:
        """Process a single device and return normalized structure."""
        spec = device.get('Specification', {}) or {}
        
        processed = {
            # Metadata
            **self.parse_metadata(device),
            # Hardware specs
            **self.parse_ram(spec),
            **self.parse_storage(spec),
            **self.parse_display(spec, device),
            **self.parse_cpu(spec),
            **self.parse_gpu(spec),
            **self.parse_npu(spec, device),
            **self.parse_connectivity(spec),
            **self.parse_battery(spec),
            # Price
            **self.parse_price(device),
            # Features
            **self.parse_features(device),
            # Keep original for reference
            'original_specification': spec
        }
        
        return processed
    
    def process_all(self) -> list:
        """Process all devices."""
        if not self.devices:
            self.load_data()
            
        self.processed_devices = []
        for device in self.devices:
            try:
                processed = self.process_device(device)
                self.processed_devices.append(processed)
            except Exception as e:
                print(f"Error processing device {device.get('SKUID', 'unknown')}: {e}")
                continue
                
        print(f"Processed {len(self.processed_devices)} devices")
        return self.processed_devices
    
    def save_processed(self) -> str:
        """Save processed devices to output file."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_devices, f, indent=2, ensure_ascii=False)
            
        print(f"Saved processed data to {self.output_path}")
        return self.output_path
    
    def get_stats(self) -> dict:
        """Get statistics about processed data."""
        if not self.processed_devices:
            return {}
            
        stats = {
            'total_devices': len(self.processed_devices),
            'brands': {},
            'categories': {},
            'form_factors': {},
            'cpu_tiers': {},
            'gpu_tiers': {},
            'price_tiers': {},
            'with_dedicated_gpu': 0,
            'with_npu': 0,
            'with_touchscreen': 0,
            'with_thunderbolt': 0
        }
        
        for device in self.processed_devices:
            # Count brands
            brand = device.get('brand', 'Unknown')
            stats['brands'][brand] = stats['brands'].get(brand, 0) + 1
            
            # Count categories
            category = device.get('category', 'Unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Count form factors
            ff = device.get('form_factor', 'Unknown')
            stats['form_factors'][ff] = stats['form_factors'].get(ff, 0) + 1
            
            # Count CPU tiers
            cpu_tier = device.get('cpu_tier', 'Unknown')
            stats['cpu_tiers'][cpu_tier] = stats['cpu_tiers'].get(cpu_tier, 0) + 1
            
            # Count GPU tiers
            gpu_tier = device.get('gpu_tier', 'Unknown')
            stats['gpu_tiers'][gpu_tier] = stats['gpu_tiers'].get(gpu_tier, 0) + 1
            
            # Count price tiers
            price_tier = device.get('price_tier', 'Unknown')
            stats['price_tiers'][price_tier] = stats['price_tiers'].get(price_tier, 0) + 1
            
            # Count features
            if device.get('has_dedicated_gpu'):
                stats['with_dedicated_gpu'] += 1
            if device.get('has_npu'):
                stats['with_npu'] += 1
            if device.get('is_touchscreen'):
                stats['with_touchscreen'] += 1
            if device.get('has_thunderbolt'):
                stats['with_thunderbolt'] += 1
                
        return stats


def main():
    """Main function to process device data."""
    # Paths
    base_path = Path(__file__).parent.parent
    input_path = base_path / 'devices.json'
    output_path = base_path / 'data' / 'processed_devices.json'
    
    # Process
    processor = DeviceDataProcessor(str(input_path), str(output_path))
    processor.load_data()
    processor.process_all()
    processor.save_processed()
    
    # Print stats
    stats = processor.get_stats()
    print("\n=== Processing Statistics ===")
    print(f"Total devices: {stats['total_devices']}")
    print(f"\nBrands: {stats['brands']}")
    print(f"\nCategories: {stats['categories']}")
    print(f"\nForm Factors: {stats['form_factors']}")
    print(f"\nCPU Tiers: {stats['cpu_tiers']}")
    print(f"\nGPU Tiers: {stats['gpu_tiers']}")
    print(f"\nPrice Tiers: {stats['price_tiers']}")
    print(f"\nDevices with dedicated GPU: {stats['with_dedicated_gpu']}")
    print(f"Devices with NPU: {stats['with_npu']}")
    print(f"Devices with touchscreen: {stats['with_touchscreen']}")
    print(f"Devices with Thunderbolt: {stats['with_thunderbolt']}")


if __name__ == '__main__':
    main()
