import re
from bs4 import BeautifulSoup, Tag
from pylatexenc.latex2text import LatexNodes2Text

def clean_cell_content(tag):
    """Removes styling and keeps only text/structural content.
       Converts LaTeX math to text equivalents using pylatexenc.
    """
    # Remove all attributes (style, align, etc)
    tag.attrs = {}
    
    # 1. Image tags
    for img in tag.find_all('img'):
        img.replace_with("[IMAGE]")
        
    # 2. Extract Text
    text = tag.get_text(" ", strip=True) 
    
    # 3. LaTeX Normalization
    try:
        text = LatexNodes2Text().latex_to_text(text)
    except:
        pass 
        
    tag.string = text.strip()
    return tag

def canonicalize_html(html_str):
    """
    Converts a table HTML to a dense grid format.
    - Resolves rowspans/colspans by duplicating cells.
    - Flattens nested structures.
    - Removes all styling.
    """
    if not html_str: return "<table><tbody></tbody></table>"
    
    soup = BeautifulSoup(html_str, 'html.parser')
    orig_table = soup.find('table')
    if not orig_table: return "<table><tbody></tbody></table>"
    
    # Grid map: (r, c) -> content
    grid = {} 
    max_row = 0
    max_col = 0
    
    # Try to find rows in thead/tbody/tfoot or direct children
    rows = orig_table.find_all('tr')
    
    for r_idx, row in enumerate(rows):
        c_idx = 0
        cells = row.find_all(['td', 'th'])
        
        for cell in cells:
            # Skip occupied cells (from previous rowspans)
            while (r_idx, c_idx) in grid:
                c_idx += 1
            
            # Get dimensions
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            
            # Clean content
            clean_cell_content(cell)
            content = cell.string or ""
            
            # Fill Grid
            for r in range(rowspan):
                for c in range(colspan):
                    target_r = r_idx + r
                    target_c = c_idx + c
                    grid[(target_r, target_c)] = {'content': content}
                    max_row = max(max_row, target_r)
                    max_col = max(max_col, target_c)
            
            # Advance cursor
            c_idx += colspan
    
    # Reconstruct Table
    new_table = Tag(name='table')
    tbody = Tag(name='tbody')
    new_table.append(tbody)
    
    for r in range(max_row + 1):
        tr = Tag(name='tr')
        c = 0
        while c <= max_col:
            cell_data = grid.get((r, c), {'content': ''})
            new_cell = Tag(name='td')
            new_cell.string = cell_data['content']
            tr.append(new_cell)
            c += 1
        tbody.append(tr)
            
    return str(new_table)
