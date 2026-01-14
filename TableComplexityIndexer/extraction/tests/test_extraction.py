import sys
import os
import json
import pytest

# Add project root to path (3 levels up: extraction/tests/ -> TableComplexityIndexer/extraction/ -> TableComplexityIndexer/ -> ROOT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from TableComplexityIndexer.extraction.extract_tables import run_extraction, OUTPUT_JSON

# Test output directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TEST_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "test_results.json")

def test_run_small_batch():
    """Test that extraction runs on a small batch and produces valid JSON."""
    print("\n=== Testing Extraction Run (Limit=2) ===")
    
    # 1. Clean previous output
    if os.path.exists(TEST_OUTPUT):
        os.remove(TEST_OUTPUT)
        
    # 2. Run Extraction (Limit 2)
    run_extraction(limit=2, output_path=TEST_OUTPUT)
    
    # 3. Verify Output
    assert os.path.exists(TEST_OUTPUT), "Output JSON should exist"
    
    with open(TEST_OUTPUT, 'r') as f:
        data = json.load(f)
        
    assert len(data) == 2, f"Should have processed exactly 2 items, found {len(data)}"
    
    first_item = data[0]
    assert 'teds_score' in first_item, "Result should include TEDS score"
    assert 'pred_html' in first_item, "Result should include predicted HTML"
    assert 'label' in first_item, "Result should include PASS/FAIL label"
    
    print("Success: Extraction ran and produced valid JSON for 2 items.")
