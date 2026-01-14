import os
import sys
import pytest
from pathlib import Path

# Add the parent directory (paddle_app) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the project root directory to sys.path for shared utils and tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from extract_tables_opensource import PaddleOCRGenerator
from utils.pdf_utils import save_pdf_pages_as_png
from utils.clean_md import clean_markdown

# Adapter function to match user's signature request
def run_paddle_ocr_generate(images, output_path, pdf_file=None):
    generator = PaddleOCRGenerator(use_gpu=True)
    return generator.generate(input_file=pdf_file, output_dir=output_path)


@pytest.fixture(scope="session")
def pdf_folder():
    # Adjusted path to match where we put data
    return Path(__file__).parent.parent.parent / "data" / "pdf"

@pytest.fixture(scope="session")
def output_folder():
    out = Path(__file__).parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out

@pytest.mark.parametrize("pdf_file_name", ["Xie2019.pdf"])
def test_paddle_ocr(pdf_file_name, pdf_folder, output_folder):
    model_type = "PaddlePaddle/PaddleOCR-VL"
    
    safe_model_name = model_type.replace("/", "_").replace("-", "_")
    model_output_folder = output_folder / safe_model_name
    model_output_folder.mkdir(parents=True, exist_ok=True)

    pdf_file = pdf_folder / pdf_file_name

    print(f"Processing PDF: {pdf_file}")
    if not pdf_file.exists():
        pytest.skip(f"PDF {pdf_file} not found")

    # Save PNGs for verification/debugging as per user request
    saved_pngs = save_pdf_pages_as_png(pdf_file, out_dir=model_output_folder)
    assert len(saved_pngs) > 0
    for path in saved_pngs:
        assert path.exists()

    # Run generation
    # Logic: output_path=str(model_output_folder / pdf_file.stem)
    # Our generate function uses output_dir as base.
    target_output = model_output_folder / pdf_file.stem
    
    run_paddle_ocr_generate(saved_pngs, output_path=target_output, pdf_file=pdf_file)
    
    # Check artifacts
    # Combined MD expected at specific location
    md_path = target_output / f"{pdf_file.stem}.md"
    
    assert md_path.exists(), f"Markdown file missing at {md_path}"
    
    clean_markdown(str(md_path))
    
    clean_path = md_path.with_name(f"{md_path.stem}_clean.md")
    assert clean_path.exists()
    
    # Check custom markdown (from JSON reconstruction)
    custom_md_path = target_output / f"{pdf_file.stem}_custom.md"
    assert custom_md_path.exists(), f"Custom markdown file missing at {custom_md_path}"
    
    print(f"Test complete. Artifacts in {target_output}")
