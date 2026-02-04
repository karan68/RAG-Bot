"""
Phase 1 Runner Script
Executes all data preparation steps in sequence
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processor import DeviceDataProcessor
from document_generator import DocumentGenerator


def main():
    """Run Phase 1: Data Preparation pipeline."""
    print("=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    
    # Step 1: Process raw device data
    print("\n[1/3] Processing raw device data...")
    print("-" * 40)
    
    input_path = base_path / 'devices.json'
    processed_path = base_path / 'data' / 'processed_devices.json'
    
    if not input_path.exists():
        print(f"ERROR: devices.json not found at {input_path}")
        return 1
    
    processor = DeviceDataProcessor(str(input_path), str(processed_path))
    processor.load_data()
    processor.process_all()
    processor.save_processed()
    
    # Print stats
    stats = processor.get_stats()
    print(f"\n✓ Processed {stats['total_devices']} devices")
    print(f"  - Brands: {len(stats['brands'])}")
    print(f"  - Categories: {len(stats['categories'])}")
    print(f"  - With dedicated GPU: {stats['with_dedicated_gpu']}")
    print(f"  - With NPU: {stats['with_npu']}")
    
    # Step 2: Generate searchable documents
    print("\n[2/3] Generating searchable documents...")
    print("-" * 40)
    
    rules_path = base_path / 'inference_rules.json'
    documents_path = base_path / 'data' / 'device_documents.json'
    
    if not rules_path.exists():
        print(f"ERROR: inference_rules.json not found at {rules_path}")
        return 1
    
    generator = DocumentGenerator(str(processed_path), str(rules_path), str(documents_path))
    generator.load_data()
    generator.generate_all_documents()
    generator.save_documents()
    
    print(f"\n✓ Generated {len(generator.documents)} documents")
    
    # Step 3: Validate test data
    print("\n[3/3] Validating test dataset...")
    print("-" * 40)
    
    test_path = base_path / 'tests' / 'test_queries.json'
    
    if test_path.exists():
        import json
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_cases = test_data.get('test_cases', [])
        categories = test_data.get('category_counts', {})
        
        print(f"✓ Loaded {len(test_cases)} test cases")
        print(f"  Categories: {categories}")
    else:
        print(f"WARNING: Test dataset not found at {test_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"""
Output files created:
  1. {processed_path}
     - Cleaned and normalized device specifications
     - Extracted numeric values (ram_gb, storage_gb, etc.)
     - Added computed fields (cpu_tier, gpu_tier, etc.)
  
  2. {documents_path}
     - Searchable text documents for each device
     - Capability assessments included
     - Ready for embedding in Phase 2
  
  3. {test_path}
     - 65 test Q&A pairs for evaluation
     - Covers: specs, capabilities, comparisons, upgrades, Copilot, edge cases

Next Steps (Phase 2):
  1. Install Ollama: winget install Ollama.Ollama
  2. Pull models: ollama pull phi3.5 && ollama pull nomic-embed-text
  3. Run: python src/rag_pipeline.py (to be created in Phase 2)
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
