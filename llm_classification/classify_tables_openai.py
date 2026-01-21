import os
import json
import base64
import glob
import re
import sys
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()

# Configuration
MD_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/single_tables/'
IMG_DIR = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_pages/'
OUTPUT_FILE = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_classifications_openai.json'

# Verify API Key
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment or .env file.")
    # In a real scenario, we might want to ask the user, but here we just prepare the code.
    # sys.exit(1)

PROMPT = """You are a hierarchical table complexity classifier.

You are given a scientific TABLE as an IMAGE.
You must analyze ONLY what is visible inside the table borders.
Ignore captions, footnotes outside the table, and surrounding text.

Your task is to classify the table by following ALL steps below IN ORDER.
You must make exactly one decision at each step.
Do not skip steps.
Do not invent new categories.
Choose ONLY from the allowed options.

================================================
STEP 1 — Structural class
Question:
“How is the table visually and structurally organized?”

Choose ONE:

- Flat
  The table has a single header row and a simple grid.
  No grouped headers, no nested categories.

- Grouped
  The table has one level of grouping using sub-headers,
  column groups, or row groups (e.g., colspan or rowspan),
  but no deep nesting.

- Hierarchical
  The table has multi-level (nested) headers OR nested row categories,
  where a higher-level category spans multiple detailed rows
  (e.g., location → isolate → resistance profile).

- Matrix
  The table is very wide or dense (≈15 or more columns),
  showing distributions, profiles, or many repeated attributes
  (e.g., MIC distributions, resistance gene panels).

================================================
STEP 2 — Analytical depth
Question:
“How many different types of variables are jointly represented?”

Choose ONE:

- 1
  A single variable or attribute only.

- 2
  Two variables are linked
  (e.g., antibiotic × prevalence).

- 3
  Three or more variables are linked
  (e.g., isolate × source × antibiotic × gene).

================================================
STEP 3 — Semantic load
Question:
“What level of scientific reasoning is required to interpret the table?”

Choose ONE:

- Descriptive
  Values can be read directly.
  No comparison or reasoning is required.

- Comparative
  Understanding requires comparing patterns across rows or columns
  (e.g., resistance profiles across sources).

- Inferential
  Statistical inference is explicitly present,
  such as p-values, confidence intervals, or hypothesis tests.

================================================
STEP 4 — Representational density
Question:
“What kind of notation does the table use?”

Choose ONE:

- Plain
  Text and integers only.
  No domain-specific scientific notation.

- Percentages
  Percentages (%), ratios, ranges, or proportions.

- Scientific
  Domain-specific scientific notation, including:
  gene symbols (e.g., blaTEM, tet(A), int1),
  MIC values and units (mg/L),
  symbolic encodings (+++, −),
  Greek letters,
  subscripts or superscripts,
  molecular or microbiological identifiers.

================================================
STEP 5 — Domain integration
Question:
“How many distinct scientific domains are combined in the table?”

Examples of domains:
Microbiology, Genetics, Epidemiology, Pharmacology, Statistics.

Choose ONE:

- 1 domain
- 2 domains
- 3 or more domains

================================================
FINAL COMPLEXITY ASSIGNMENT RULES
Apply these rules EXACTLY and do not override them.

- Low
  Flat AND (analytical depth = 1 OR analytical depth = 2)

- Medium
  Flat AND analytical depth = 3
  OR
  Grouped AND semantic load = Descriptive

- High
  Grouped AND (Comparative OR Inferential)
  AND NOT Hierarchical
  AND NOT Matrix

- Extreme
  Any Hierarchical structure
  OR any Matrix structure
  OR Grouped AND Scientific representation
  OR Grouped AND Comparative AND 3 or more domains

================================================
IMPORTANT RULES

- Tables labeled “continued” are part of the SAME table.
- Use ONLY the information visible inside the table image.
- Provide justifications ONLY within the "decision_trace" object.
- Return ONLY the JSON object below.

================================================
OUTPUT FORMAT (STRICT JSON ONLY)

{
  "step1_structure": "",
  "step2_analytical_depth": "",
  "step3_semantic_load": "",
  "step4_representation": "",
  "step5_domains": "",
  "final_complexity": "",
  "decision_trace": {
    "structure": "",
    "depth": "",
    "semantics": "",
    "representation": "",
    "domains": "",
    "final_rule": ""
  }
}
For each decision_trace field, provide a short factual justification 
based only on visible table features (max 1 sentence each)."""

# Initialize client only if key is available
client = None
if os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify_table(img_path, table_id):
    if not client:
        return None
        
    if not os.path.exists(img_path):
        print(f"Warning: Image not found {img_path}")
        return None
    
    base64_image = encode_image(img_path)
    
    table_specific_prompt = f"### TARGET TABLE ###\nFocus ONLY on {table_id} on the provided page.\n\n{PROMPT}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": table_specific_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call for {table_id}: {e}")
        return None

def main():
    if not client:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    results = {}
    
    md_files = sorted(glob.glob(os.path.join(MD_DIR, "*.md")))
    
    # Resume if exists
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} existing entries in {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            pass
            
    for md_file in tqdm(md_files):
        filename = os.path.basename(md_file)
        if filename in results:
            continue
              
        table_id_match = re.search(r'Table_(\d+)', filename)
        table_id = f"Table {table_id_match.group(1)}" if table_id_match else "the table"

        img_filename = filename.replace('.md', '.png')
        img_path = os.path.join(IMG_DIR, img_filename)
        
        # tqdm's write avoids overlapping with progress bar
        tqdm.write(f"Processing {filename} (Target: {table_id})...")
        output = classify_table(img_path, table_id)
        
        if output:
            try:
                results[filename] = json.loads(output)
            except json.JSONDecodeError:
                results[filename] = {"raw_output": output, "error": "JSON parsing failed"}
        else:
            results[filename] = {"error": "API call failed"}

        # Save incrementally
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print(f"Finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
