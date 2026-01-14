import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt
from .base_engine import BaseEngine
import json
import re

class DotsEngine(BaseEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print("Loading DotsOCR...")
        # Use HF Hub path or local if downloaded
        model_path = "rednote-hilab/DotsOCR" 
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"DotsOCR Load Error: {e}")
            raise e

    def predict(self, image_path: str) -> str:
        if not self.model:
            self.load_model()

        try:
            prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Preparation
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096) # Adjust token limit as needed
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Extract Table HTML from JSON output
            try:
                # The output should be a JSON object, but might be wrapped or have thinking tokens if it was deepseek (it's qwen though).
                # Sometimes models output markdown code blocks.
                json_str = output_text
                match_md = re.search(r'```json(.*?)```', output_text, re.DOTALL | re.IGNORECASE)
                if match_md:
                    json_str = match_md.group(1)
                
                data = json.loads(json_str)
                
                # Assume structure is a list of elements or a dict with a list
                elements = []
                if isinstance(data, list):
                    elements = data
                elif isinstance(data, dict):
                    # Try to find the list. The prompt says "A single JSON object", probably containing keys.
                    # Qwen usually outputs a list of dicts if trained that way, or a dict with "layout_elements" key.
                    # Let's simple search values.
                    for v in data.values():
                        if isinstance(v, list):
                            elements = v
                            break
                
                html_tables = []
                for el in elements:
                    if isinstance(el, dict) and el.get('category') == 'Table':
                        html = el.get('text_content') or el.get('text')
                        if html:
                            html_tables.append(html)
                
                return "\n\n".join(html_tables) if html_tables else ""

            except json.JSONDecodeError:
                print(f"DotsOCR JSON Parse Error. Raw: {output_text[:100]}...")
                return "" # Failed to parse
                
        except Exception as e:
            print(f"DotsOCR Inference Error: {e}")
            return ""
