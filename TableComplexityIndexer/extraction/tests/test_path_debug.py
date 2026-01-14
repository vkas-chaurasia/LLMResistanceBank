import os
import sys
import pytest

# Add project root to path (3 levels up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from TableComplexityIndexer.config import PUBTABNET_DIR

def test_pubtabnet_path_configuration():
    """Verify that PUBTABNET_DIR is correctly configured and points to an existing directory."""
    print(f"Config PUBTABNET_DIR: {PUBTABNET_DIR}")
    
    # Verify Directory Exists
    assert os.path.exists(PUBTABNET_DIR), f"PUBTABNET_DIR does not exist: {PUBTABNET_DIR}"
    assert os.path.isdir(PUBTABNET_DIR), f"PUBTABNET_DIR is not a directory: {PUBTABNET_DIR}"

    # Verify Validation File Exists
    target_file = os.path.join(PUBTABNET_DIR, "validation_annotations.json")
    print(f"Checking target file: {target_file}")
    
    assert os.path.exists(target_file), f"Validation annotations check failed. Expected at: {target_file}"
