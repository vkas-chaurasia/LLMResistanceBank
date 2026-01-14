import json
import os
import sys
import statistics
from extract_gt_features import extract_features  # Reuse our feature extractor

def main():
    print("=== Analyzing Predicted HTML Features vs TEDS ===")
    
    # Load extraction results
    results_path = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")
    if not os.path.exists(results_path):
        print("No results found.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    perfect_preds = []
    fail_preds = []
    
    print(f"Processing {len(data)} predictions...")
    
    for item in data:
        teds = item.get('teds_score', 0)
        pred_html = item.get('pred_html', '')
        filename = item.get('filename')
        
        # Extract features from the PREDICTED HTML
        feats = extract_features(pred_html, filename)
        
        if not feats:
            continue
            
        group = perfect_preds if teds >= 0.999 else fail_preds
        group.append(feats)
        
    print(f"\nGroup Sizes based on TEDS:")
    print(f"  Success (1.0): {len(perfect_preds)}")
    print(f"  Failure (<1.0): {len(fail_preds)}")
    
    # Compare Metrics
    metrics = ['num_rows', 'num_cols', 'empty_cell_ratio', 'avg_text_len', 'max_rowspan', 'max_colspan']
    
    print("\n--- Features of PREDICTED HTML (Signal Analysis) ---")
    for m in metrics:
        p_vals = [x[m] for x in perfect_preds]
        f_vals = [x[m] for x in fail_preds]
        
        p_avg = statistics.mean(p_vals) if p_vals else 0
        f_avg = statistics.mean(f_vals) if f_vals else 0
        
        diff = f_avg - p_avg
        print(f"\nMetric: {m}")
        print(f"  Success Avg: {p_avg:.2f}")
        print(f"  Failure Avg: {f_avg:.2f}")
        print(f"  Delta: {diff:+.2f}")

if __name__ == "__main__":
    main()
