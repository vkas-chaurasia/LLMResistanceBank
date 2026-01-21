import os
import json
import fitz  # pymupdf

# Configuration
PDF_DIR = '/cluster/home/chaurvik/LLMResistanceBank/data/pdf/'
SEARCH_DIR = '/cluster/home/chaurvik/LLMResistanceBank/paddle_app/tests/outputs/PaddlePaddle_PaddleOCR_VL/'
OUTPUT_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_pages/'

def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def extract_table_pages():
    setup_output_dir()
    
    if not os.path.exists(SEARCH_DIR):
        print(f"Search directory not found: {SEARCH_DIR}")
        return

    doc_dirs = sorted([d for d in os.listdir(SEARCH_DIR) if os.path.isdir(os.path.join(SEARCH_DIR, d))])
    
    global_page_count = 0
    
    for doc_name in doc_dirs:
        pdf_path = os.path.join(PDF_DIR, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
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
        
        # We want to export the whole page if it contains a table.
        # But wait, one page might have multiple tables.
        # We should export the page ONCE per table? Or just unique pages?
        # The user said "extract page with table and pass that page".
        # If we are doing per-table analysis, we probably want to associate "Table X" with "Page Y".
        # If a page possesses 2 tables, we can just save the page image and reference it.
        # But to be safe and simple: Save {doc_name}_Table_{i}_Page_{p}.png
        # This duplicates storage but ensures 1:1 mapping for the LLM script later.
        
        for page_num, page_path in page_dirs:
            structure_file = os.path.join(page_path, f'structure_{page_num}.json')
            if not os.path.exists(structure_file):
                continue
            
            try:
                with open(structure_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                continue
            
            # Check for tables
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
            
            try:
                # Load page (0-indexed)
                page = doc.load_page(page_num - 1)
                
                # Render high quality
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Save one image for each table found on this page
                # This might seem redundant if multiple tables are on one page, 
                # but it keeps the file count consistent with the extracted markdown tables.
                
                for t in tables:
                    doc_table_count += 1
                    out_filename = f"{doc_name}_clean_Table_{doc_table_count}.png"
                    out_path = os.path.join(OUTPUT_DIR, out_filename)
                    pix.save(out_path)
                    global_page_count += 1
                    
            except Exception as e:
                print(f"Error rendering page {page_num}: {e}")

    print(f"Done. Extracted {global_page_count} page images containing tables.")

if __name__ == "__main__":
    extract_table_pages()
