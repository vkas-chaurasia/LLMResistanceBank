import json
import os

INPUT_FILE = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_classifications_openai.json'
OUTPUT_FILE = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_complexity_list.csv'

def generate_list():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Sort by filename
    sorted_items = sorted(data.items())

    with open(OUTPUT_FILE, 'w') as f:
        # Excel compatible header
        f.write("File Name;Final Complexity\n")
        
        for filename, results in sorted_items:
            # Clean up the name for the display list
            display_name = filename.replace('_clean', '').replace('.md', '')
            complexity = results.get('final_complexity', 'Unknown')
            f.write(f"{display_name};{complexity}\n")

    print(f"List saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_list()
