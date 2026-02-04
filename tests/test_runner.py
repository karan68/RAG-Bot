"""
Test Runner for Device Chatbot
Evaluates accuracy against test dataset
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rag_pipeline import create_pipeline


class TestRunner:
    """Runs test cases and evaluates chatbot accuracy."""
    
    def __init__(self, pipeline, test_data_path: str):
        self.pipeline = pipeline
        self.test_data_path = test_data_path
        self.test_cases = []
        self.results = []
        
    def load_tests(self):
        """Load test cases from JSON file."""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.test_cases = data.get('test_cases', [])
        print(f"Loaded {len(self.test_cases)} test cases")
        return self.test_cases
    
    def set_test_device(self, device_name: str) -> bool:
        """Set device for testing, mapping test device names to actual devices."""
        # Map test device names to actual devices or test files
        device_mapping = {
            'HP Pavilion x360 14': 'HP Pavilion x360 14',
            'HP 15 Laptop': 'current_device_budget_laptop.json',
            'ASUS ROG Strix G16': 'current_device_gaming_laptop.json',
            'Microsoft Surface Laptop 7': 'current_device_copilot_pc.json',
            'Lenovo Yoga 7i 2-in-1': 'current_device_2in1.json',
            'HP All-in-One 27': 'current_device_desktop_aio.json',
            'Dell Precision 5680': 'current_device_workstation.json'
        }
        
        mapped_name = device_mapping.get(device_name, device_name)
        
        # Check if it's a test file
        if mapped_name.endswith('.json'):
            test_path = Path(__file__).parent.parent / 'test_devices' / mapped_name
            if test_path.exists():
                self.pipeline.load_current_device_from_file(str(test_path))
                return True
        
        # Otherwise try direct lookup
        result = self.pipeline.set_current_device(mapped_name)
        return result is not None
    
    def check_answer(self, response: str, expected_contains: List[str]) -> Dict:
        """Check if response contains expected content."""
        response_lower = response.lower()
        
        matches = []
        misses = []
        
        for expected in expected_contains:
            if expected.lower() in response_lower:
                matches.append(expected)
            else:
                misses.append(expected)
        
        accuracy = len(matches) / len(expected_contains) if expected_contains else 1.0
        
        return {
            'matches': matches,
            'misses': misses,
            'accuracy': accuracy,
            'passed': accuracy >= 0.5  # Pass if at least 50% of expected terms found
        }
    
    def run_test(self, test_case: Dict) -> Dict:
        """Run a single test case."""
        test_id = test_case.get('id', 'unknown')
        question = test_case.get('question', '')
        context_device = test_case.get('context_device', '')
        expected_contains = test_case.get('expected_answer_contains', [])
        category = test_case.get('category', 'unknown')
        
        # Set device
        if context_device:
            device_set = self.set_test_device(context_device)
            if not device_set:
                return {
                    'id': test_id,
                    'category': category,
                    'passed': False,
                    'error': f"Could not set device: {context_device}",
                    'response': None
                }
        
        # Run query
        try:
            start_time = time.time()
            result = self.pipeline.chat(question)
            elapsed = time.time() - start_time
            
            response = result['response']
            
            # Check answer
            check = self.check_answer(response, expected_contains)
            
            return {
                'id': test_id,
                'category': category,
                'question': question,
                'context_device': context_device,
                'response': response,
                'expected_contains': expected_contains,
                'matches': check['matches'],
                'misses': check['misses'],
                'accuracy': check['accuracy'],
                'passed': check['passed'],
                'classification': result['classification'],
                'elapsed_seconds': elapsed
            }
            
        except Exception as e:
            return {
                'id': test_id,
                'category': category,
                'passed': False,
                'error': str(e),
                'response': None
            }
    
    def run_all_tests(self, categories: List[str] = None) -> Dict:
        """Run all test cases and return summary."""
        if not self.test_cases:
            self.load_tests()
        
        self.results = []
        
        # Filter by category if specified
        tests_to_run = self.test_cases
        if categories:
            tests_to_run = [t for t in self.test_cases if t.get('category') in categories]
        
        print(f"\nRunning {len(tests_to_run)} tests...")
        print("-" * 60)
        
        for i, test_case in enumerate(tests_to_run):
            test_id = test_case.get('id', f'test_{i}')
            print(f"[{i+1}/{len(tests_to_run)}] {test_id}...", end=" ")
            
            result = self.run_test(test_case)
            self.results.append(result)
            
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{status} ({result.get('accuracy', 0):.0%})")
            
            if not result['passed'] and result.get('misses'):
                print(f"         Missing: {result['misses'][:3]}")
        
        # Calculate summary
        summary = self.calculate_summary()
        
        return summary
    
    def calculate_summary(self) -> Dict:
        """Calculate test summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        # By category
        by_category = {}
        for result in self.results:
            cat = result.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = {'total': 0, 'passed': 0}
            by_category[cat]['total'] += 1
            if result['passed']:
                by_category[cat]['passed'] += 1
        
        # Calculate category accuracies
        for cat, stats in by_category.items():
            stats['accuracy'] = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        
        # Average response time
        times = [r.get('elapsed_seconds', 0) for r in self.results if r.get('elapsed_seconds')]
        avg_time = sum(times) / len(times) if times else 0
        
        # Overall accuracy (average of per-test accuracies)
        accuracies = [r.get('accuracy', 0) for r in self.results if 'accuracy' in r]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        summary = {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'average_accuracy': avg_accuracy,
            'average_response_time': avg_time,
            'by_category': by_category
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print formatted test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"\nOverall Results:")
        print(f"  Total Tests:     {summary['total_tests']}")
        print(f"  Passed:          {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"  Failed:          {summary['failed']}")
        print(f"  Avg Accuracy:    {summary['average_accuracy']:.1%}")
        print(f"  Avg Response:    {summary['average_response_time']:.2f}s")
        
        print(f"\nBy Category:")
        for cat, stats in sorted(summary['by_category'].items()):
            print(f"  {cat:20s}: {stats['passed']}/{stats['total']} ({stats['accuracy']:.0%})")
        
        # List failed tests
        failed_tests = [r for r in self.results if not r['passed']]
        if failed_tests:
            print(f"\nFailed Tests ({len(failed_tests)}):")
            for result in failed_tests[:10]:  # Show first 10
                print(f"  - {result['id']}: {result.get('error', 'Missing: ' + str(result.get('misses', [])[:2]))}")
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON file."""
        output = {
            'summary': self.calculate_summary(),
            'results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_path}")


def main():
    """Run test suite."""
    print("=" * 60)
    print("DEVICE CHATBOT TEST RUNNER")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    test_data_path = base_path / 'tests' / 'test_queries.json'
    results_path = base_path / 'tests' / 'test_results.json'
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = create_pipeline(str(base_path))
    
    # Create test runner
    runner = TestRunner(pipeline, str(test_data_path))
    runner.load_tests()
    
    # Run tests (optionally filter by category)
    # categories = ['spec_lookup', 'capability']  # Run specific categories
    categories = None  # Run all
    
    summary = runner.run_all_tests(categories)
    
    # Print summary
    runner.print_summary(summary)
    
    # Save results
    runner.save_results(str(results_path))
    
    # Return exit code based on pass rate
    target_accuracy = 0.70  # 70% pass rate target
    if summary['pass_rate'] >= target_accuracy:
        print(f"\n✓ Tests PASSED (>= {target_accuracy:.0%} pass rate)")
        return 0
    else:
        print(f"\n✗ Tests FAILED (< {target_accuracy:.0%} pass rate)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
