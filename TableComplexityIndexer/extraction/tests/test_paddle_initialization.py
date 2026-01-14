import os
import sys
import pytest
from paddleocr import PaddleOCRVL

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from TableComplexityIndexer.config import PUBTABNET_DIR

def test_paddle_ocr_initialization_and_inference():
    """Test PaddleOCRVL initialization with various configs and run inference on one sample."""
    
    image_path = os.path.join(PUBTABNET_DIR, "images", "val_0.png")
    
    if not os.path.exists(image_path):
        pytest.skip(f"Sample image not found at {image_path}")

    # Configs to test (reduced set from debug_vl.py)
    configs = [
        ("No Format", {"use_chart_recognition": False}),
        ("With Format", {"use_chart_recognition": False, "format_block_content": True}),
    ]
    
    for name, kwargs in configs:
        print(f"\nTesting Config: {name}")
        try:
            # Init Model
            engine = PaddleOCRVL(**kwargs)
            
            # Predict
            res = engine.predict(image_path)
            assert res is not None, f"Prediction returned None for {name}"
            print(f"Success for {name}")
            
        except Exception as e:
            pytest.fail(f"Config {name} failed: {e}")
