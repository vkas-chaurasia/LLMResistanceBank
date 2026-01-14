import torch
from transformers import AutoModel, AutoTokenizer
from .base_engine import BaseEngine
import re

class GotEngine(BaseEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print("Loading GOT-OCR-2.0...")
        model_path = "stepfun-ai/GOT-OCR2_0"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cuda", # Force cuda if available, usually required for GOT
                use_safetensors=True
            ).eval()
            if self.device == "cuda":
                self.model = self.model.cuda()
        except Exception as e:
            print(f"GOT Load Error: {e}")
            raise e

    def predict(self, image_path: str) -> str:
        if not self.model:
            self.load_model()
            
        try:
            # GOT API usually has a .chat() method or specific forward
            # Based on standard usage:
            # res = model.chat(tokenizer, image_file, ocr_type='format')
            # ocr_type='format' requests formatted result (often Latex/Markdown).
            # We want HTML if possible, but Latex is default for tables.
            
            # Note: GOT chat API might vary, assuming standard 'chat' interface from Stepfun repo
            res = self.model.chat(self.tokenizer, image_path, ocr_type='format')
            
            # Result is likely Latex/Markdown.
            # Example: \begin{tabular}...
            # We need HTML for TEDS. 
            # Strategy: Convert Latex -> HTML using pylatexenc (simple) or just return raw if we can't reliably convert.
            # Actually, standard TEDS metric works best on HTML.
            
            # If the user requested deepseek/GOT specifically, they might accept the raw text if TEDS fails, 
            # BUT our pipeline needs HTML for TEDS.
            
            # Simple Latex Table to HTML (heuristic)
            # Or ask the model to output HTML?
            # GOT supports 'format' (latex/markdown). Not explicit HTML.
            
            # For this benchmark, let's try prompting for HTML if chat supports custom prompts
            # res = model.chat(tokenizer, image_path, ocr_type='ocr', user_prompt='Output HTML')
            
            # If standard chat:
            return res
            
        except Exception as e:
            print(f"GOT Inference Error: {e}")
            return ""
