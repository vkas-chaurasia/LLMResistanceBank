import json
import os
import sys
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.html_processing import canonicalize_html
from config import PUBTABNET_DIR

def extract_features(html_str, filename):
    """
    Extracts comprehensive structural features from a table HTML string.
    """
    if not html_str:
        return None

    # 1. Raw Parsing (for Spans, Headers, Nesting)
    soup = BeautifulSoup(html_str, 'html.parser')
    
    # --- Spans & Merged Cells ---
    all_cells = soup.find_all(['td', 'th'])
    
    rowspans = []
    colspans = []
    merged_cells_count = 0
    
    for cell in all_cells:
        r = int(cell.get('rowspan', 1))
        c = int(cell.get('colspan', 1))
        rowspans.append(r)
        colspans.append(c)
        if r > 1 or c > 1:
            merged_cells_count += 1
            
    max_rowspan = max(rowspans) if rowspans else 1
    max_colspan = max(colspans) if colspans else 1
    
    # --- Headers ---
    thead = soup.find('thead')
    if thead:
        header_rows = thead.find_all('tr')
        header_depth = len(header_rows)
        num_header_cells = len(thead.find_all(['td', 'th']))
    else:
        # Fallback: Count rows with majority <th>? Or just 0 if strict.
        # PubTabNet annotations usually explicit.
        header_depth = 0
        num_header_cells = len(soup.find_all('th'))
        
    # --- Nesting ---
    tables = soup.find_all('table')
    num_nested_tables = max(0, len(tables) - 1) # Subtract main table
    
    def get_max_depth(tag, current_depth):
        nested = tag.find('table')
        if not nested:
            return current_depth
        return get_max_depth(nested, current_depth + 1)
        
    max_nested_depth = 0
    if num_nested_tables > 0:
        max_nested_depth = get_max_depth(soup, 0)

    # 2. Canonicalize (for Dimensions & Text Density)
    try:
        canonical_html = canonicalize_html(html_str)
    except Exception:
        return None
    
    can_soup = BeautifulSoup(canonical_html, 'html.parser')
    rows = can_soup.find_all('tr')
    
    if not rows:
        return None
    
    num_rows = len(rows)
    num_cols = 0
    for r in rows:
        num_cols = max(num_cols, len(r.find_all(['td', 'th'])))
        
    total_grid_cells = num_rows * num_cols
    
    # Content Analysis (on Canonical Grid)
    empty_cells = 0
    total_text_len = 0
    
    for r in rows:
        cells = r.find_all(['td', 'th'])
        # Account for sparse grid in canonical rep if any
        current_row_len = len(cells)
        empty_cells += (num_cols - current_row_len)
        
        for c in cells:
            text = c.get_text(strip=True)
            if not text:
                empty_cells += 1
            total_text_len += len(text)
            
    empty_cell_ratio = empty_cells / total_grid_cells if total_grid_cells > 0 else 0
    merged_cell_density = merged_cells_count / total_grid_cells if total_grid_cells > 0 else 0
    
    return {
        "filename": filename,
        # Dimensions
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_cells": total_grid_cells,
        # Spans
        "max_rowspan": max_rowspan,
        "max_colspan": max_colspan,
        "total_merged_cells": merged_cells_count,
        "merged_cell_density": merged_cell_density,
        # Headers
        "header_depth": header_depth,
        "num_header_cells": num_header_cells,
        # Nesting
        "num_nested_tables": num_nested_tables,
        "max_nested_depth": max_nested_depth,
        # Content
        "empty_cell_ratio": empty_cell_ratio,
        "total_text_len": total_text_len,
        "avg_text_len": total_text_len / total_grid_cells if total_grid_cells > 0 else 0,
        "canonical_html_len": len(canonical_html)
    }

def main():
    print("=== Extracting Features from Ground Truth (Canonicalized) ===")
    
    dataset_path = os.path.join(PUBTABNET_DIR, "validation_annotations.json")
    output_path = os.path.join(os.path.dirname(__file__), "gt_features.json")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} items...")
    
    results = []
    
    # For now, process all. 
    for item in tqdm(data):
        filename = item.get('filename', 'unknown')
        html_str = item.get('html', '')
        
        feats = extract_features(html_str, filename)
        if feats:
            # Also keep 'imgid' if available to link with other data
            feats['imgid'] = item.get('imgid')
            results.append(feats)
            
    # Save Results
    print(f"Extracted features for {len(results)} tables.")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Features saved to: {output_path}")

    # Preliminary Analysis
    if len(results) > 0:
        rows = [r['num_rows'] for r in results]
        cols = [r['num_cols'] for r in results]
        sparsities = [r['empty_cell_ratio'] for r in results]
        
        print("\n--- Summary Statistics ---")
        print(f"Avg Rows: {np.mean(rows):.2f} (Max: {np.max(rows)})")
        print(f"Avg Cols: {np.mean(cols):.2f} (Max: {np.max(cols)})")
        print(f"Avg Sparsity: {np.mean(sparsities):.2f}")

if __name__ == "__main__":
    main()
