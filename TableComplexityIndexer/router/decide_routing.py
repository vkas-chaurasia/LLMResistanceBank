import os
import sys
import json
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.extract_gt_features import extract_features
from extraction.extract_tables import run_extraction # To generate prediction if needed logic exists there?
# Actually extract_features takes HTML string. We need to get the Prediction from Paddle first.
# For this script, let's assume we maintain the pipeline logic:
# Image -> Paddle -> Pred HTML -> Feature Extractor -> Decision.

def decide_route(pred_html, filename):
    """
    Decides whether to route to 'PADDLE_OCR' (Accept Result) or 'LLM_REVIEW' (Reject Result).
    
    Rule (Safe/High-Recall):
    - Avg Cell Text Length > 10.64  OR
    - Total Text Length > 485
    
    Returns:
        decision (str): 'PADDLE_OCR' or 'LLM_REVIEW'
        reason (str): Explanation of the metric triggered.
    """
    feats = extract_features(pred_html, filename)
    if not feats:
        # If we can't extract features (empty?), it's likely a failure.
        return 'LLM_REVIEW', "Extraction Failed / Empty"
        
    avg_len = feats.get('avg_text_len', 0)
    total_len = feats.get('total_text_len', 0)
    
    # Use Central Config
    from config import ROUTER_AVG_TEXT_LEN_THRESHOLD, ROUTER_TOTAL_TEXT_LEN_THRESHOLD
    
    if avg_len > ROUTER_AVG_TEXT_LEN_THRESHOLD:
        return 'LLM_REVIEW', f"High Avg Text Length ({avg_len:.1f} > {ROUTER_AVG_TEXT_LEN_THRESHOLD})"
        
    if total_len > ROUTER_TOTAL_TEXT_LEN_THRESHOLD:
        return 'LLM_REVIEW', f"High Total Text Length ({total_len} > {ROUTER_TOTAL_TEXT_LEN_THRESHOLD})"
        
    return 'PADDLE_OCR', "Low Complexity"

def main():
    print("=== Heuristic Router Decision ===")
    
    # For demonstration, we load the PRE-COMPUTED predictions from processing_results.json
    # In a real pipeline, this would run *after* Paddle but *before* final acceptance.
    
    results_path = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")
    if not os.path.exists(results_path):
        print("Input path not found.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    decisions = {
        'PADDLE_OCR': 0,
        'LLM_REVIEW': 0
    }
    
    missed_failures = 0
    caught_failures = 0
    false_alarms = 0
    
    print(f"{'Filename':<20} {'TEDS':<6} {'Decision':<12} {'Reason'}")
    print("-" * 60)
    
    for item in data:
        teds = item.get('teds_score', 0)
        is_actual_fail = teds < 0.99
        
        decision, reason = decide_route(item.get('pred_html', ''), item.get('filename'))
        
        decisions[decision] += 1
        
        # Eval
        if is_actual_fail:
            if decision == 'LLM_REVIEW':
                caught_failures += 1
            else:
                missed_failures += 1
                print(f"[MISS] {item['filename']} (TEDS={teds:.2f}) -> {decision}")
        else:
            if decision == 'LLM_REVIEW':
                false_alarms += 1
                
        # Print a few samples
        if decisions[decision] <= 5: # Print first 5 of each type?
             print(f"{item['filename']:<20} {teds:.3f}  {decision:<12} {reason}")

    print("\n=== Summary ===")
    print(f"Routed to PADDLE (Auto): {decisions['PADDLE_OCR']}")
    print(f"Routed to LLM (Human):   {decisions['LLM_REVIEW']}")
    print("-" * 20)
    print(f"Failures Caught (Recall): {caught_failures} / {caught_failures + missed_failures}")
    print(f"False Alarms (Cost):      {false_alarms} (Good tables sent to LLM)")
    
    if missed_failures == 0:
        print("\nSUCCESS: 100% Recall Achieved. No failures slipped through.")
    else:
        print(f"\nWARNING: {missed_failures} failures were processed by Paddle incorrectly!")

if __name__ == "__main__":
    main()
