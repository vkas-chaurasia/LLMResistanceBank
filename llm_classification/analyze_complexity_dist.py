import json
import os

INPUT_FILE = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/table_classifications_openai.json'
OUTPUT_FILE = '/cluster/home/chaurvik/LLMResistanceBank/llm_classification/complexity_distribution_openai.md'

def analyze_distribution():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    dist = {}
    total = len(data)
    
    for entry in data.values():
        complexity = entry.get('final_complexity', 'Unknown')
        dist[complexity] = dist.get(complexity, 0) + 1

    # Sort categories in logical order
    categories = ['Low', 'Medium', 'High', 'Extreme', 'Unknown']
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Table Complexity Distribution (OpenAI GPT-4o)\n\n")
        f.write(f"Total Tables Processed: **{total}**\n\n")
        f.write("| Complexity Level | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        for cat in categories:
            count = dist.get(cat, 0)
            if count > 0 or cat != 'Unknown':
                percent = (count / total) * 100 if total > 0 else 0
                f.write(f"| **{cat}** | {count} | {percent:.1f}% |\n")

    print(f"Analysis saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_distribution()
