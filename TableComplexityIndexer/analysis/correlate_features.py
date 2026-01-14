import json
import os
import sys
import statistics

def load_json(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def print_stats(name, values):
    if not values:
        print(f"{name}: No Data")
        return
    print(f"{name}: Avg={statistics.mean(values):.2f}, Median={statistics.median(values):.2f}, Max={max(values)}")

def main():
    print("=== Correlating Structural Features with TEDS Scores ===")
    
    # Paths
    base_dir = os.path.dirname(__file__) # analysis/
    gt_features_path = os.path.join(base_dir, "gt_features.json")
    results_path = os.path.join(base_dir, "../output/processing_results.json")
    
    # Load Data
    gt_features_list = load_json(gt_features_path)
    extraction_results = load_json(results_path)
    
    if not gt_features_list or not extraction_results:
        print("Missing input files. Ensure both 'extract_gt_features.py' and 'extract_tables.py' have run.")
        return

    # Index GT Features by filename for fast lookup
    gt_map = {item['filename']: item for item in gt_features_list}
    
    # Groups
    perfect_group = [] # TEDS >= 0.99
    fail_group = []    # TEDS < 0.95 (or configurable)
    
    print(f"Analyzing {len(extraction_results)} extraction results...")
    
    for res in extraction_results:
        fname = res.get('filename')
        teds = res.get('teds_score', 0)
        
        if fname not in gt_map:
            continue
            
        feats = gt_map[fname]
        
        # We focus on specific metrics
        metrics = {
            'num_rows': feats['num_rows'],
            'num_cols': feats['num_cols'],
            'empty_cell_ratio': feats['empty_cell_ratio'],
            'text_len': feats['total_text_len'],
            'avg_text_len': feats['avg_text_len'],
            'html_len': feats['canonical_html_len'],
            # New Structural
            'max_rowspan': feats['max_rowspan'],
            'max_colspan': feats['max_colspan'],
            'merged_density': feats['merged_cell_density'],
            'header_depth': feats['header_depth'],
            'nested_depth': feats['max_nested_depth']
        }
        
        if teds >= 0.999: # Allowing extremely small float error, basically 1.0
            perfect_group.append(metrics)
        else:
            fail_group.append(metrics)
            
    # Report
    print(f"\nGroup Sizes:")
    print(f"  Perfect (TEDS=1.0): {len(perfect_group)}")
    print(f"  Fail/Partial (<1.0): {len(fail_group)}")
    
    if not perfect_group or not fail_group:
        print("\nNot enough data in both groups to compare.")
        return
        
    metrics_keys = perfect_group[0].keys()
    
    print("\n--- Feature Comparison ---")
    for key in metrics_keys:
        print(f"\nMetric: {key}")
        p_vals = [x[key] for x in perfect_group]
        f_vals = [x[key] for x in fail_group]
        
        print("  [Perfect Group]")
        print_stats("    Stats", p_vals)
        print("  [Fail Group]")
        print_stats("    Stats", f_vals)
        
        diff = statistics.mean(f_vals) - statistics.mean(p_vals)
        print(f"  -> Fail group avg is {'higher' if diff>0 else 'lower'} by {abs(diff):.2f}")

if __name__ == "__main__":
    main()
