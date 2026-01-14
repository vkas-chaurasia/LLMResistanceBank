import sys
from paddleocr import PaddleOCRVL
from .base_engine import BaseEngine

class PaddleEngine(BaseEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None

    def load_model(self):
        print("Loading PaddleOCR-VL...")
        # use_chart_recognition=False, format_block_content=True from original script
        self.model = PaddleOCRVL(use_chart_recognition=False, format_block_content=True)

    def predict(self, image_path: str) -> str:
        if not self.model:
            self.load_model()
            
        try:
            ocr_results = self.model.predict(image_path)
        except Exception as e:
            print(f"Paddle Inference Error: {e}")
            return ""
            
        contents = []
        if ocr_results:
            for res in ocr_results:
                structure = None
                # Handle inconsistent Paddle Output formats (JSON vs Object)
                if hasattr(res, 'json') and res.json:
                    structure = res.json.get('res', {}).get('parsing_res_list', [])
                elif hasattr(res, 'parsing_res_list'):
                    structure = res.parsing_res_list
                    
                if structure:
                    for block in structure:
                        # Only extract blocks labeled 'table'
                        if 'table' in block.get('block_label', '').lower():
                            html = block.get('html', block.get('block_content', ''))
                            if html: contents.append(html)
                            
        return "".join(contents)
