import os
import re
import logging
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Union, Any, Dict
from paddleocr import PaddleOCRVL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaddleOCRGenerator:
    """
    Wrapper for PaddleOCR-VL to handle PDF processing.
    
    Generates markdown, structured JSON, and validation images per page,
    plus a combined markdown file for the full document.
    """
    
    def __init__(self, use_gpu: bool = True):
        # Initialize PaddleOCRVL with configuration for layout formatting
        # Using specific parameters as recommended for document parsing
        try:
            self.pipeline = PaddleOCRVL(
                use_chart_recognition=False, 
                format_block_content=True
            )
            logger.info("PaddleOCRVL initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCRVL: {e}")
            raise

    def generate(
        self,
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process a PDF or Image using PaddleOCR-VL and generate artifacts.
        
        Args:
            input_file: Path to PDF or Image file.
            output_dir: Root directory for outputs.
            
        Returns:
            Dictionary containing metadata about generated files.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing document: {input_path}")
        
        try:
            # Predict method accepts str path
            results = self.pipeline.predict(str(input_path))
        except Exception as e:
            logger.error(f"Prediction execution failed: {e}")
            raise
            
        summary = {
            "total_pages": len(results),
            "pages": [],
            "combined_md": None,
            "combined_custom_md": None
        }
        
        markdown_list = []
        custom_markdown_list = []
        markdown_images_collection = [] 
        references_found = False
        
        for i, res in enumerate(results):
            page_num = i + 1
            page_folder_name = f"page_{page_num}"
            page_dir = output_base / page_folder_name
            page_dir.mkdir(exist_ok=True)
            
            logger.info(f"Generating artifacts for Page {page_num} in {page_dir}")
            
            # 1. Save OCR Visualization (Image)
            img_path = page_dir / f"ocr_{page_num}.png"
            res.save_to_img(save_path=str(img_path))
            
            # 2. Save JSON Structure
            json_path = page_dir / f"structure_{page_num}.json"
            res.save_to_json(save_path=str(json_path))
            
            # 3. Save Page Markdown (Paddle Original)
            md_path = page_dir / f"content_{page_num}.md"
            try:
                res.save_to_markdown(save_path=str(md_path), pretty=False)
            except AttributeError:
                # Fallback to content property if method missing
                md_content = res.markdown
                if isinstance(md_content, dict):
                    md_content = md_content.get('content', str(md_content))
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(str(md_content))

            # Store for combined generation
            markdown_list.append(res.markdown)
            markdown_images_collection.append(res.markdown.get("markdown_images", {}))
            
            # 4. Generate Custom Markdown from JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                
            custom_md_content, refs_found_in_page = self._reconstruct_md_from_json(page_data, references_found)
            
            if refs_found_in_page:
                references_found = True
                
            custom_md_path = page_dir / f"content_{page_num}_custom.md"
            with open(custom_md_path, 'w', encoding='utf-8') as f:
                f.write(custom_md_content)
                
            custom_markdown_list.append(custom_md_content)
            
            summary["pages"].append({
                "page": page_num,
                "md": str(md_path),
                "custom_md": str(custom_md_path),
                "json": str(json_path),
                "img": str(img_path)
            })

        # 5. Generate Combined Markdown Files
        logger.info("Generating combined Markdown files...")
        
        # Original Combined
        combined_md_filename = f"{input_path.stem}.md"
        combined_md_path = output_base / combined_md_filename
        
        # Custom Combined
        combined_custom_filename = f"{input_path.stem}_custom.md"
        combined_custom_path = output_base / combined_custom_filename
        
        try:
            # Save Original Combined
            merged_content = self.pipeline.concatenate_markdown_pages(markdown_list)
            with open(combined_md_path, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            summary["combined_md"] = str(combined_md_path)
            
            # Save Custom Combined
            custom_merged_content = "\n\n".join(custom_markdown_list)
            with open(combined_custom_path, 'w', encoding='utf-8') as f:
                f.write(custom_merged_content)
            summary["combined_custom_md"] = str(combined_custom_path)
            
            # Save images referenced in the combined markdown
            for page_imgs in markdown_images_collection:
                if page_imgs:
                    for img_rel_path, img_obj in page_imgs.items():
                        full_img_path = output_base / img_rel_path
                        full_img_path.parent.mkdir(parents=True, exist_ok=True)
                        img_obj.save(full_img_path)
                        
        except Exception as e:
            logger.error(f"Failed to generate combined markdown: {e}")
            
        return summary

    def _reconstruct_md_from_json(self, json_data: Dict[str, Any], global_ref_found: bool = False) -> tuple[str, bool]:
        """
        Reconstructs Markdown from JSON layout data by filtering headers/footers
        and removing references.
        """
        # The 'parsing_res_list' returned by PaddleOCR is already sorted based on 
        # layout analysis (reading order). We iterate sequentially to preserve this.
        parsing_res = json_data.get('parsing_res_list', [])
        content_lines = []
        found_ref = global_ref_found
        
        # Labels to ignore for cleaner output
        ignore_labels = {'header_image', 'footer_image', 'page_number'}
        
        for block in parsing_res:
            if found_ref:
                break
                
            label = block.get('block_label', '').lower()
            text = block.get('block_content', '')
            
            # Skip ignored layout elements
            if label in ignore_labels:
                continue
                
            # Stop if we hit a Reference content block
            if 'reference_content' in label:
                found_ref = True
                break
                
            # Check for References section start (Markdown header)
            # Matches "## References", "# References", etc.
            if re.match(r'^#+\s*References\b', text, re.IGNORECASE):
                found_ref = True
                break
            
            # Also check generic reference label if likely a section
            if 'reference' in label and label != 'reference_content':
                 # Use caution, but usually implies reference section
                 found_ref = True
                 break
                
            content_lines.append(text)
            
        return "\n\n".join(content_lines), found_ref
