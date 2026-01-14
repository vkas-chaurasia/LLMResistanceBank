import json
import os
import sys
import argparse
import numpy as np

def load_results(engine_name):
    # output/results_{engine}.json
    path = os.path.join(os.path.dirname(__file__), f"../output/results_{engine_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def calculate_metrics(results):
    if not results: return {}
    
    scores = [r['teds_score'] for r in results]
    perfect = sum(1 for s in scores if s >= 0.9999)
    fail = sum(1 for s in scores if s < 0.95)
    
    return {
        'count': len(scores),
        'avg_teds': np.mean(scores),
        'median_teds': np.median(scores),
        'perfect_rate': perfect / len(scores) * 100,
        'fail_rate': fail / len(scores) * 100,
        'scores': scores
    }

def main():
    print("=== Model Benchmark Comparison ===")
    
    engines = ['paddle', 'deepseek', 'got']
    stats = {}
    
    # Load Data
    for eng in engines:
        res = load_results(eng)
        if res:
            stats[eng] = calculate_metrics(res)
            print(f"Loaded {eng}: {stats[eng]['count']} samples")
        else:
            print(f"Skipping {eng}: No results found.")
            
    if not stats:
        print("No results to compare.")
        return

    # Print Table
    print("\n" + "="*80)
    print(f"{'Model':<12} | {'Samples':<8} | {'Avg TEDS':<10} | {'Perf %':<8} | {'Fail %':<8} | {'Median':<8}")
    print("-" * 80)
    
    for eng, m in stats.items():
        print(f"{eng:<12} | {m['count']:<8} | {m['avg_teds']:.4f}     | {m['perfect_rate']:.1f}%     | {m['fail_rate']:.1f}%     | {m['median_teds']:.4f}")
        
    print("="*80)
    
    # Compare Overlap (Common Samples)
    # If we have multiple models, compare on *intersection* of filenames
    if len(stats) > 1:
        print("\n=== Intersection Analysis ===")
        # Get intersection of filenames
        # (Assuming first result loaded has ample data for keys)
        # TODO: Implement deeper intersection analysis if needed
        pass

if __name__ == "__main__":
    main()
