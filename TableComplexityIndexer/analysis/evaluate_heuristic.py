import json
import os
import sys
from extract_gt_features import extract_features 

def main():
    print("=== Heuristic Router Evaluation ===")
    
    results_path = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")
    if not os.path.exists(results_path):
        print("No results found.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    # Ground Truth: FAIL if TEDS <= 0.95
    # Hypothesis: PREDICT FAIL if AvgTextLen > Threshold
    
    true_labels = [] # 1 if Fail, 0 if Pass
    scores = []      # Avg Text Len
    
    print(f"Evaluating {len(data)} samples...")
    
    valid_data = []
    
    for item in data:
        teds = item.get('teds_score', 0)
        is_fail = 1 if teds < 0.99 else 0  # Strict failure definition
        
        pred_html = item.get('pred_html', '')
        feats = extract_features(pred_html, item.get('filename'))
        
        if feats:
            avg_len = feats['avg_text_len']
            true_labels.append(is_fail)
            scores.append(avg_len)
            valid_data.append({'is_fail': is_fail, 'score': avg_len})
            
    # Sweep Thresholds
    thresholds = [10, 12, 15, 18, 20, 25, 30]
    
    print(f"\n{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Flagged%':<10}")
    print("-" * 55)
    
    total_fails = sum(true_labels)
    
    for t in thresholds:
        tp = 0 # True Positive (Correctly Flagged as Fail)
        fp = 0 # False Positive (Flagged as Fail but was Pass)
        fn = 0 # False Negative (Missed Fail)
        flagged = 0
        
        for item in valid_data:
            pred_fail = 1 if item['score'] > t else 0
            actual_fail = item['is_fail']
            
            if pred_fail: flagged += 1
            
            if pred_fail and actual_fail: tp += 1
            elif pred_fail and not actual_fail: fp += 1
            elif not pred_fail and actual_fail: fn += 1
            
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        flag_rate = flagged / len(valid_data) * 100
        
        print(f"{t:<10} {prec:.2f}       {rec:.2f}       {f1:.2f}       {flag_rate:.1f}%")

    print(f"\nTotal Real Failures: {total_fails}")

if __name__ == "__main__":
    main()
