import os
import json
import glob
import re
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from tqdm import tqdm

# Configuration
MD_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/single_tables/'
IMG_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_pages/'
OUTPUT_FILE = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_classifications.json'

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

SCHEMA_PROMPT = """
You are an expert in classifying table complexity in scientific documents.
Analyze the provided table image to classify it according to the following schema.

L1_Structure: (Flat, Grouped, Hierarchical, Matrix/Mega)
    - Flat: Simple rows/cols.
    - Grouped: Merged headers or sections.
    - Hierarchical: Nested headers, tree-like structure.
    - Matrix/Mega: Complex multi-dimensional cross-tabs or very large/dense tables.

L2_Depth: (L2-1, L2-2, L2-3)
    - Analytical Depth. 1=Descriptive, 2=Comparative, 3=Statistical/Inference.

L3_Semantic: (L3-a, L3-b, L3-c)
    - Semantic Load. a=Low (Simple Labels), b=Medium (Domain Terms), c=High (Complex Relationships/formulas).

L4_Density: (L4-0, L4-1, L4-2)
    - Representational Density. 0=Sparse, 1=Normal, 2=High/Cluttered.

L5_Domain: (L5-1, L5-2, L5-3)
    - Domain Integration. 1=General, 2=Field-Specific, 3=Highly Specialized.

Final_Class: (Simple, Low, Medium, High, Extreme)

Reasoning: Brief explanation.

Return ONLY a JSON object with these keys. Do not output markdown code blocks.
"""

def setup_model():
    print(f"Loading model: {MODEL_ID}")
    # Always use AutoModel with trust_remote_code=True for experimental models
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID, 
            dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

def classify_table(model, processor, md_path, img_path):
    # Only using Image now as per request.
    
    if not os.path.exists(img_path):
        print(f"Warning: Image not found for {md_path}")
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text", 
                    "text": SCHEMA_PROMPT
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

from qwen_vl_utils import process_vision_info

def main():
    model, processor = setup_model()
    results = {}
    
    md_files = sorted(glob.glob(os.path.join(MD_DIR, "*.md")))
    
    # Check if we can resume?
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
        except:
            pass
            
    for md_file in tqdm(md_files):
        filename = os.path.basename(md_file)
        if filename in results:
             continue # skip done
             
        img_filename = filename.replace('.md', '.png')
        img_path = os.path.join(IMG_DIR, img_filename)
        
        print(f"Processing {filename}...")
        try:
            output = classify_table(model, processor, md_file, img_path)
            
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    results[filename] = data
                except:
                    results[filename] = {"raw_output": output, "error": "JSON parse failed"}
            else:
                 results[filename] = {"raw_output": output, "error": "No JSON found"}
                 
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results[filename] = {"error": str(e)}

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print(f"Finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
