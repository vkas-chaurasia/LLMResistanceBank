import os
import re
from bs4 import BeautifulSoup

# Configuration
SEARCH_DIR = '/cluster/home/chaurvik/LLMResistanceBank/paddle_app/tests/outputs/PaddlePaddle_PaddleOCR_VL/'
OUTPUT_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/single_tables/'

def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def collect_and_split_tables():
    setup_output_dir()
    
    # 1. Collect all matching files first
    all_files = []
    if not os.path.exists(SEARCH_DIR):
        print(f"Search directory does not exist: {SEARCH_DIR}")
        return

    for root, dirs, files in os.walk(SEARCH_DIR):
        for file in files:
            if file.endswith('_clean.md'):
                all_files.append((file, os.path.join(root, file)))
    
    # 2. Sort by filename
    all_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(all_files)} files to process.")
    
    table_global_count = 0
    
    # 3. Process sorted files
    for file_name, file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all tables with their span
            table_matches = list(re.finditer(r'<table.*?>.*?</table>', content, re.DOTALL | re.IGNORECASE))
            
            if not table_matches:
                continue
                
            print(f"Processing {file_name}: {len(table_matches)} tables found.")
            
            for i, match in enumerate(table_matches):
                start, end = match.span()
                raw_html = match.group(0)
                
                # Prettify HTML with BeautifulSoup
                try:
                    soup = BeautifulSoup(raw_html, 'html.parser')
                    table_html = soup.prettify()
                except Exception as parse_error:
                    print(f"Warning: Failed to prettify table in {file_name}: {parse_error}")
                    table_html = raw_html
                
                # Look backwards for Caption (up to 1000 chars)
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

                # Look forward for Footnotes/Abbreviations (up to 1000 chars)
                post_context = content[end:min(len(content), end+1000)]
                notes = ""
                note_match = re.search(r'^\s*(Footnote|Note|Abbreviation|Legend|Key).*?(?=\n\n)', post_context, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                if note_match:
                    notes = f"**Captured Notes**: {note_match.group(0).strip()}\n\n"
                else:
                    next_paras = post_context.strip().split('\n\n')
                    if next_paras:
                        next_para = next_paras[0]
                        if len(next_para) < 500 and not next_para.strip().startswith('#') and \
                           ("values" in next_para or "indicates" in next_para or "mean" in next_para or "mg/L" in next_para or "resistance" in next_para):
                            notes = f"**Captured Notes**: {next_para.strip()}\n\n"
                
                # Construct the file content
                file_content = f"# Source: {file_name}\n\n"
                file_content += caption
                file_content += table_html
                file_content += "\n\n"
                file_content += notes
                
                # Create a unique filename
                # clean_file_name removing extension
                base_name = os.path.splitext(file_name)[0]
                output_filename = f"{base_name}_Table_{i+1}.md"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(file_content)
                
                table_global_count += 1
                
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    print(f"Total tables extracted: {table_global_count}")

if __name__ == "__main__":
    collect_and_split_tables()
