import json
import os
import sys
import statistics
from extract_gt_features import extract_features 

def main():
    print("=== Analyzing Features Characteristic of TEDS=1 (Success) ===")
    
    results_path = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")
    if not os.path.exists(results_path):
        print("No results found.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    print(f"Total Samples: {len(data)}")
    
    # We want to find a rule that PREDICTS Success (TEDS >= 0.99)
    # usually "Low Complexity" => Success.
    
    samples = []
    for item in data:
        teds = item.get('teds_score', 0)
        is_success = 1 if teds >= 0.99 else 0
        
        pred_html = item.get('pred_html', '')
        feats = extract_features(pred_html, item.get('filename'))
        
        if feats:
            feats['is_success'] = is_success
            samples.append(feats)
            
    metrics = ['avg_text_len', 'total_text_len', 'num_cols', 'num_rows', 'empty_cell_ratio']
    
    print(f"\nScanning for 'Safe Zones' (Where Precision for Success is high)...")
    
    for m in metrics:
        # We assume "Lower is Safer/Better" for these complexity metrics
        values = sorted([s[m] for s in samples])
        
        best_rule = None
        best_prec = 0
        best_rec = 0
        best_coverage = 0
        
        # Test thresholds
        for i in range(len(values) - 1):
            t = (values[i] + values[i+1]) / 2
            
            # Rule: Metric < T => SUCCESS
            tp = fp = total_predicted_safe = 0
            
            for s in samples:
                pred_safe = 1 if s[m] < t else 0
                actual_success = s['is_success']
                
                if pred_safe:
                    total_predicted_safe += 1
                    if actual_success: tp += 1
                    else: fp += 1
            
            # Precision = How many predicted safe are actually safe?
            prec = tp / total_predicted_safe if total_predicted_safe > 0 else 0
            # Recall = How many of the total successes did we catch?
            total_successes = sum(s['is_success'] for s in samples)
            rec = tp / total_successes if total_successes > 0 else 0
            
            if prec >= 0.95 and rec > best_rec: # Find max recall for 95% safety
                best_prec = prec
                best_rec = rec
                best_rule = t
                best_coverage = total_predicted_safe / len(samples)
        
        if best_rule is not None:
            print(f"\nFeature: {m}")
            print(f"  Safe Rule: < {best_rule:.2f}")
            print(f"  Reliability (Precision): {best_prec*100:.1f}%")
            print(f"  Coverage (Recall):       {best_rec*100:.1f}% of success cases found")
            print(f"  (This rule safely auto-approves {best_coverage*100:.1f}% of the dataset)")
        else:
            print(f"\nFeature: {m} - No safe threshold found (>95% precision).")

if __name__ == "__main__":
    main()
