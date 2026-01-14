import os
import json
import pytest
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from TableComplexityIndexer.config import PUBTABNET_DIR

def test_pubtabnet_validation_data_exists():
    """Verify that the validation annotations file and images directory exist."""
    annotations_path = os.path.join(PUBTABNET_DIR, "validation_annotations.json")
    images_dir = os.path.join(PUBTABNET_DIR, "images")
    
    assert os.path.exists(annotations_path), f"Annotations file not found at {annotations_path}"
    assert os.path.exists(images_dir), f"Images directory not found at {images_dir}"

def test_pubtabnet_validation_content():
    """Verify content of the validation set: count, image existence, and HTML structure."""
    annotations_path = os.path.join(PUBTABNET_DIR, "validation_annotations.json")
    images_dir = os.path.join(PUBTABNET_DIR, "images")
    
    if not os.path.exists(annotations_path):
        pytest.skip("Data not downloaded yet.")
        
    valid_samples = 0
    
    # Load FULL JSON (List of records)
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse JSON: {e}")
        
    assert isinstance(data, list), "Annotations JSON should be a list of records"
    assert len(data) > 1000, f"Expected > 1000 validation samples, found {len(data)}"
    
    # Check first 20 items to be safe (avoid slow I/O on 9000 items)
    for i, record in enumerate(data[:20]):
        # Check fields
        assert 'filename' in record, f"Item {i} missing 'filename'"
        assert 'html' in record, f"Item {i} missing 'html'"
        
        # Check HTML not empty
        assert record['html'] and len(record['html']) > 0, f"Item {i} has empty HTML"
        
        # Check Image Exists
        image_path = os.path.join(images_dir, record['filename'])
        assert os.path.exists(image_path), f"Image {record['filename']} not found for item {i}"
        
        valid_samples += 1
            
    print(f"\n Verified first {valid_samples} validation samples.")
