import pymupdf
from pathlib import Path
from typing import List

def save_pdf_pages_as_png(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """
    Save each PDF page as PNGs for testing/debugging.
    Creates a subdirectory under out_dir named after the PDF filename (without extension).

    Args:
        pdf_path: Path to the PDF file
        out_dir: Base output folder for test PNGs
        dpi: Rendering resolution

    Returns:
        List of Path objects pointing to saved PNGs
    """
    pdf_name = pdf_path.stem
    pdf_out_dir = out_dir / pdf_name
    pdf_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    saved_paths: List[Path] = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        # Using 0-indexed + 1 for page number
        path = pdf_out_dir / f"{pdf_name}_{page.number + 1:03d}.png"
        pix.save(str(path))
        saved_paths.append(path)

    doc.close()
    return saved_paths
