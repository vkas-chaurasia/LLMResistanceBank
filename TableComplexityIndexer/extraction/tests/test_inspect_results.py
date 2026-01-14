import json
import os
import sys
import pytest

# Path to the shared JSON output in output/ (2 levels up from extraction/tests/)
JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/processing_results.json"))

def test_inspect_processing_results():
    """Verify and inspect the production processing_results.json file."""
    print(f"\nInspecting extraction results from: {JSON_PATH}")
    
    assert os.path.exists(JSON_PATH), f"Output file not found at {JSON_PATH}. Run extraction first."

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    assert isinstance(data, list), "Output should be a list of records"
    assert len(data) > 0, "Output list is empty"
    
    print(f"Total Records: {len(data)}")
    
    # Inspect first 2 items
    for i, item in enumerate(data[:2]):
        print(f"\n--- Item {i+1} ---")
        item_keys = ['filename', 'imgid', 'teds_score', 'label', 'pred_html']
        for k in item_keys:
            assert k in item, f"Item {i} matches schema: missing {k}"
            
        print(f"Filename:   {item.get('filename')}")
        print(f"ImgID:      {item.get('imgid')}")
        print(f"TEDS Score: {item.get('teds_score')}")
        print(f"Label:      {item.get('label')}")
        print(f"HTML len:   {len(item.get('pred_html', ''))} chars")
        
    print("\nSuccess: Output file is valid and follows schema.")
