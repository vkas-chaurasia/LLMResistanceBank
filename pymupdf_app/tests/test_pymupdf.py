import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pymupdf_app.extract_tables_pymupdf import PyMuPDFGenerator

def test_generation():
    generator = PyMuPDFGenerator()
    input_pdf = Path("../data/pdf/Xie2019.pdf").resolve()
    output_dir = Path("pymupdf_app/tests/outputs/Xie2019").resolve()
    
    print(f"Testing generation for {input_pdf}")
    summary = generator.generate(input_pdf, output_dir)
    
    print("Generation complete.")
    print(f"Total pages: {summary['total_pages']}")
    print(f"Combined MD: {summary['combined_md']}")
    
    # Verify files exist
    assert Path(summary['combined_md']).exists()
    assert Path(summary['combined_custom_md']).exists()
    for page in summary['pages']:
        assert Path(page['md']).exists()
        assert Path(page['json']).exists()
        assert Path(page['img']).exists()
        
    print("All artifacts verified successfully.")

if __name__ == "__main__":
    test_generation()
