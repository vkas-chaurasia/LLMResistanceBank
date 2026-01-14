import json
import os
import itertools
from extract_gt_features import extract_features 

def main():
    print("=== Searching for Perfect Separation Rule ===")
    
    results_path = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Prepare Data
    samples = []
    for item in data:
        teds = item.get('teds_score', 0)
        is_fail = 1 if teds < 0.99 else 0
        
        # Extract features from PREDICTED HTML (since that's what router sees)
        pred_html = item.get('pred_html', '')
        feats = extract_features(pred_html, item.get('filename'))
        
        if feats:
            feats['is_fail'] = is_fail
            samples.append(feats)
            
    # Metrics to sweep
    metrics = ['avg_text_len', 'total_text_len', 'num_cols', 'num_rows', 'empty_cell_ratio', 'max_colspan']
    
    best_rule = None
    best_score = (0, 0) # (Recall, Precision)
    
    # Brute Force Single Thresholds
    print("Checking Single Thresholds...")
    for m in metrics:
        values = sorted([s[m] for s in samples])
        # Check midpoints
        for i in range(len(values) - 1):
            t = (values[i] + values[i+1]) / 2
            
            # Rule: Metric > T => FAIL
            tp = fp = fn = tn = 0
            for s in samples:
                pred = 1 if s[m] > t else 0
                actual = s['is_fail']
                if pred and actual: tp += 1
                if pred and not actual: fp += 1
                if not pred and actual: fn += 1
                if not pred and not actual: tn += 1
            
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            if rec == 1.0 and prec == 1.0:
                print(f"FOUND PERFECT RULE: {m} > {t:.2f}")
                return
            
            if rec + prec > sum(best_score):
                best_score = (rec, prec)
                best_rule = f"{m} > {t:.2f}"
                
    print(f"Best Single Rule: {best_rule} (Rec={best_score[0]:.2f}, Prec={best_score[1]:.2f})")
    
    # Brute Force 2-Metric OR Logic (Metric1 > T1 OR Metric2 > T2)
    print("\nChecking Combined Rules (OR logic)...")
    import numpy as np
    
    for m1, m2 in itertools.combinations(metrics, 2):
        # Discretize search space to avoid explosion
        vals1 = np.percentile([s[m1] for s in samples], [10, 30, 50, 70, 90])
        vals2 = np.percentile([s[m2] for s in samples], [10, 30, 50, 70, 90])
        
        for t1 in vals1:
            for t2 in vals2:
                tp = fp = fn = tn = 0
                for s in samples:
                    pred = 1 if (s[m1] > t1 or s[m2] > t2) else 0
                    actual = s['is_fail']
                    if pred and actual: tp += 1
                    elif pred and not actual: fp += 1
                    elif not pred and actual: fn += 1
                    
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                if rec == 1.0 and prec == 1.0:
                     print(f"FOUND PERFECT RULE: {m1} > {t1:.2f} OR {m2} > {t2:.2f}")
                     return
                     
                if rec + prec > sum(best_score):
                    best_score = (rec, prec)
                    best_rule = f"{m1} > {t1:.2f} OR {m2} > {t2:.2f}"

    print(f"Best Combined Rule: {best_rule} (Rec={best_score[0]:.2f}, Prec={best_score[1]:.2f})")
    print("No 100% rule found in simple search space.")

if __name__ == "__main__":
    main()
