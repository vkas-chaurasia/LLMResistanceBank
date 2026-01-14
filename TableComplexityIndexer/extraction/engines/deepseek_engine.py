import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from .base_engine import BaseEngine
import re

class DeepSeekEngine(BaseEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print("Loading DeepSeek-VL-7B-Chat...")
        model_path = "deepseek-ai/deepseek-vl-7b-chat" 
        
        # Load VLM
        # Note: Depending on GPU VRAM, might need 4-bit loading.
        # Assuming reasonable VRAM or CPU offload for now.
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        except Exception as e:
            print(f"DeepSeek Load Error: {e}")
            raise e

    def predict(self, image_path: str) -> str:
        if not self.model:
            self.load_model()
            
        try:
            # 1. Prepare Conversation
            # DeepSeek-VL expects specific prompt format
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Extract the table from this image as HTML code. Return ONLY the HTML `<table>...</table>`. Do not include explanation.",
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            # 2. Process Input
            from PIL import Image
            pil_image = Image.open(image_path).convert("RGB")
            
            # Use processor (handling image + text)
            inputs = self.processor(
                conversations=conversation,
                images=[pil_image],
                force_batching=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 3. Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048, # Tables can be long
                    do_sample=False,     # Deterministic
                    use_cache=True
                )
                
            # 4. Decode
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 5. Extract HTML
            # Response likely contains the conversation history or just the assistant part depending on decoder.
            # Usually need to split or parse.
            # Heuristic: Find first <table> and last </table>
            
            match = re.search(r'(<table>.*?</table>)', response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                # Fallback: Look for markdown code blocks
                match_md = re.search(r'```html(.*?)```', response, re.DOTALL | re.IGNORECASE)
                if match_md:
                    return match_md.group(1).strip()
                
                # If no tags, return raw (might be failure)
                return response.strip()
                
        except Exception as e:
            print(f"DeepSeek Inference Error: {e}")
            return ""
