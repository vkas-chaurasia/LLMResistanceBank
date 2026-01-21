import os
import json
import fitz  # pymupdf

# Configuration
PDF_DIR = '/cluster/home/chaurvik/LLMResistanceBank/data/pdf/'
SEARCH_DIR = '/cluster/home/chaurvik/LLMResistanceBank/paddle_app/tests/outputs/PaddlePaddle_PaddleOCR_VL/'
OUTPUT_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_images_clean/'

def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def extract_from_pdf():
    setup_output_dir()
    
    if not os.path.exists(SEARCH_DIR):
        print(f"Search directory not found: {SEARCH_DIR}")
        return

    doc_dirs = sorted([d for d in os.listdir(SEARCH_DIR) if os.path.isdir(os.path.join(SEARCH_DIR, d))])
    
    global_table_count = 0
    
    for doc_name in doc_dirs:
        # Determine PDF path
        # Assuming PDF name matches doc_name.pdf
        # doc_name might be "Adelowo2014" -> "Adelowo2014.pdf"
        pdf_path = os.path.join(PDF_DIR, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
            print(f"PDF not found for {doc_name}: {pdf_path}")
            continue
            
        print(f"Processing PDF: {doc_name}...")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open PDF {pdf_path}: {e}")
            continue

        doc_path = os.path.join(SEARCH_DIR, doc_name)
        
        # Determine page directories (page_1, page_2, ...)
        page_dirs = []
        for d in os.listdir(doc_path):
            if d.startswith('page_') and os.path.isdir(os.path.join(doc_path, d)):
                try:
                    page_num = int(d.split('_')[1])
                    page_dirs.append((page_num, os.path.join(doc_path, d)))
                except ValueError:
                    pass
        
        page_dirs.sort(key=lambda x: x[0])
        
        doc_table_count = 0
        
        for page_num, page_path in page_dirs:
            structure_file = os.path.join(page_path, f'structure_{page_num}.json')
            if not os.path.exists(structure_file):
                continue
            
            try:
                with open(structure_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {structure_file}: {e}")
                continue
            
            # Find tables
            tables = []
            if 'parsing_res_list' in data:
                for block in data['parsing_res_list']:
                    if block.get('block_label') == 'table' or block.get('label') == 'table':
                        tables.append(block)
            
            if not tables and 'layout_det_res' in data and 'boxes' in data['layout_det_res']:
                 for box in data['layout_det_res']['boxes']:
                     if box.get('label') == 'table':
                         tables.append(box)
            
            if not tables:
                continue
                
            # Get dimensions from JSON to calculate scale
            # "width": 1224, "height": 1584 for example. 
            # Note: OCR often runs on images rendered at e.g. 200 DPI or similar.
            
            json_w = data.get('width')
            json_h = data.get('height')
            
            # Get PDF page
            # page_num is 1-based usually in folder structure, fitz is 0-based
            try:
                page = doc.load_page(page_num - 1)
            except Exception as e:
                print(f"  Error loading page {page_num}: {e}")
                continue
            
            pdf_w = page.rect.width
            pdf_h = page.rect.height
            
            scale_w = pdf_w / json_w if json_w else 1.0
            scale_h = pdf_h / json_h if json_h else 1.0

            # Render page to Pixmap
            # We want high quality
            zoom = 2.0  # 2x resolution (144 DPI)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            for table in tables:
                doc_table_count += 1
                
                # Get OCR coordinates
                bbox = table.get('block_bbox') or table.get('coordinate') or table.get('bbox')
                if not bbox:
                    continue
                
                # Scale coordinates to PDF points
                x1 = bbox[0] * scale_w
                y1 = bbox[1] * scale_h
                x2 = bbox[2] * scale_w
                y2 = bbox[3] * scale_h
                
                # Further scale to Pixmap dimensions (zoom factor)
                x1 *= zoom
                y1 *= zoom
                x2 *= zoom
                y2 *= zoom
                
                # Padding
                padding = 10 * zoom
                x1 = max(0, int(x1 - padding))
                y1 = max(0, int(y1 - padding))
                x2 = min(pix.width, int(x2 + padding))
                y2 = min(pix.height, int(y2 + padding))
                
                # Crop using Pixmap
                # pymupdf 1.18+ supports pixmap cropping but easier might be just using PIL on the pixmap bytes
                # Actually newer pymupdf supports 'pix.subpixmap' or similar?
                # Actually, pix includes the whole page.
                
                try:
                    # Create sub-pixmap
                    rect = fitz.Rect(x1, y1, x2, y2)
                    # Use 'irect' (integer rect) for safety
                    irect = fitz.IRect(x1, y1, x2, y2)
                    
                    # Ensure within bounds
                    irect = irect & pix.irect
                    
                    if irect.is_empty:
                         continue

                    # Extract the area
                    # Note: fitz doesn't have a direct crop that returns a new pixmap easily in older versions, 
                    # but we can try just saving it via PIL or simply using the clip parameter if rendering again (slower).
                    # Actually, the most efficient way for multiple tables per page:
                    # Render full page once (done), then slice the buffer?
                    # Or simple: render only the clip for each table!
                    pass 
                except:
                    pass

            # Better approach loop:
            for table in tables:
                # Calculate PDF Rect for the table
                bbox = table.get('block_bbox') or table.get('coordinate') or table.get('bbox')
                if not bbox: continue
                
                x1 = bbox[0] * scale_w
                y1 = bbox[1] * scale_h
                x2 = bbox[2] * scale_w
                y2 = bbox[3] * scale_h
                
                # Add padding in Points
                p = 10
                pdf_rect = fitz.Rect(x1 - p, y1 - p, x2 + p, y2 + p)
                
                # Render ONLY this clip
                try:
                    # Clip must be intersected with page rect
                    pdf_rect = pdf_rect & page.rect
                    
                    pix_crop = page.get_pixmap(matrix=mat, clip=pdf_rect)
                    
                    out_filename = f"{doc_name}_clean_Table_{doc_table_count}.png"
                    out_path = os.path.join(OUTPUT_DIR, out_filename)
                    
                    pix_crop.save(out_path)
                    global_table_count += 1
                except Exception as e:
                    print(f"Error extracting table: {e}")
                    
    print(f"Done. Extracted {global_table_count} clean table images from PDFs.")

if __name__ == "__main__":
    extract_from_pdf()
