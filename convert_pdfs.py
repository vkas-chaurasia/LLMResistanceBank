import os
import logging
from pathlib import Path
from pymupdf_app.extract_tables_pymupdf import PyMuPDFGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    pdf_dir = base_dir / "data" / "pdf"
    output_dir = base_dir / "pymupdf_app" / "tests" / "outputs"
    
    # Initialize generator
    generator = PyMuPDFGenerator()
    
    # Get all PDF files
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    logger.info(f"Found {len(pdf_files)} PDF files.")
    
    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file.name}...")
        try:
            # We want the output specifically in data/markdown/{pdf_name_folder} or just flat?
            # The generator creates a folder per page in the output_dir.
            # If we want data/markdown/filename.md directly, we might need to adjust.
            # But the generator signature is generate(input, output_dir_root).
            # It creates output_dir_root/input_stem.md
            
            # Let's point it to data/markdown/outputs/filename_stem
            # so we keep things organized.
            file_output_dir = output_dir / pdf_file.stem
            
            summary = generator.generate(pdf_file, file_output_dir)
            logger.info(f"Generated {summary['combined_md']}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
