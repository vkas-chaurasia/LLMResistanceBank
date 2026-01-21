import os
import re
from bs4 import BeautifulSoup

SEARCH_DIR = '/cluster/home/chaurvik/LLMResistanceBank/paddle_app/tests/outputs/PaddlePaddle_PaddleOCR_VL/'
OUTPUT_FILE = 'all_tables.md'

def collect_tables():
    # 1. Collect all matching files first
    all_files = []
    for root, dirs, files in os.walk(SEARCH_DIR):
        for file in files:
            if file.endswith('_clean.md'):
                all_files.append((file, os.path.join(root, file)))
    
    # 2. Sort by filename
    all_files.sort(key=lambda x: x[0])
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # 3. Process sorted files
        for file, file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # We will start by finding tables, then look around them for context
                # Find all tables with their span
                table_matches = list(re.finditer(r'<table.*?>.*?</table>', content, re.DOTALL | re.IGNORECASE))
                
                if table_matches:
                    outfile.write(f"\n\n# File: {file}\n\n")
                    
                    for match in table_matches:
                        start, end = match.span()
                        raw_html = match.group(0)
                        
                        # Prettify HTML with BeautifulSoup
                        try:
                            soup = BeautifulSoup(raw_html, 'html.parser')
                            table_html = soup.prettify()
                        except Exception as parse_error:
                            print(f"Warning: Failed to prettify table in {file}: {parse_error}")
                            table_html = raw_html
                        
                        # Look backwards for Caption/Title (e.g. "Table 1. ...")
                        # Scan up to 1000 chars back (increased from 500)
                        pre_context = content[max(0, start-1000):start]
                        caption = ""
                        # Regex for caption in div
                        div_caption = re.search(r'<div[^>]*>(Table\s*\d+[.]?.*?)</div>\s*$', pre_context, re.DOTALL | re.IGNORECASE)
                        if div_caption:
                            caption = f"**Captured Caption**: {div_caption.group(1).strip()}\n\n"
                        else:
                            # Regex for plain text caption on a line by itself
                            plain_caption = re.search(r'(Table\s*\d+[.]?.*?)\n\s*$', pre_context, re.MULTILINE)
                            if plain_caption:
                                caption = f"**Captured Caption**: {plain_caption.group(1).strip()}\n\n"

                        outfile.write(caption)
                        outfile.write(table_html)
                        outfile.write("\n\n")
                        
                        # Look forward for Footnotes/Abbreviations
                        post_context = content[end:min(len(content), end+1000)]
                        note_match = re.search(r'^\s*(Footnote|Note|Abbreviation|Legend|Key).*?(?=\n\n)', post_context, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                        if note_match:
                            outfile.write(f"**Captured Notes**: {note_match.group(0).strip()}\n\n")
                        else:
                            next_paras = post_context.strip().split('\n\n')
                            if next_paras:
                                next_para = next_paras[0]
                                if len(next_para) < 500 and not next_para.strip().startswith('#') and \
                                   ("values" in next_para or "indicates" in next_para or "mean" in next_para or "mg/L" in next_para or "resistance" in next_para):
                                    outfile.write(f"**Captured Notes**: {next_para.strip()}\n\n")
                        
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

if __name__ == "__main__":
    collect_tables()
