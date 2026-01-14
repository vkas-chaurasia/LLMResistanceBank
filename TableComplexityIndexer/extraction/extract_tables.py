import os
import json
import sys
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag

# Add shared directory (2 levels up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from TableComplexityIndexer.utils.teds_metric import TEDS
from TableComplexityIndexer.utils.html_processing import canonicalize_html
from TableComplexityIndexer.config import PUBTABNET_DIR, TEDS_THRESHOLD

# Engine Imports - Moved to lazy loading inside get_engine to support split venvs

def get_engine(engine_name):
    if engine_name == 'paddle':
        from TableComplexityIndexer.extraction.engines.paddle_engine import PaddleEngine
        return PaddleEngine()
    elif engine_name == 'deepseek':
        from TableComplexityIndexer.extraction.engines.deepseek_engine import DeepSeekEngine
        return DeepSeekEngine()
    elif engine_name == 'got':
        from TableComplexityIndexer.extraction.engines.got_engine import GotEngine
        return GotEngine()
    elif engine_name == 'dots':
        from TableComplexityIndexer.extraction.engines.dots_engine import DotsEngine
        return DotsEngine()
    else:
        raise ValueError(f"Unknown engine: {engine_name}")

def run_extraction(limit=None, engine_name='paddle', output_json=None):
    if output_json is None:
        output_json = os.path.join(os.path.dirname(__file__), f"../output/results_{engine_name}.json")
        
    print(f"=== Phase 1: Table Extraction Benchmark (Engine={engine_name}, Limit={limit}) ===")
    print(f"Output Path: {output_json}")
    
    # Init Engine
    try:
        engine = get_engine(engine_name)
        engine.load_model()
    except Exception as e:
        print(f"Failed to initialize engine {engine_name}: {e}")
        return

    teds_calculator = TEDS()
    annotations_path = os.path.join(PUBTABNET_DIR, "validation_annotations.json")
    images_dir = os.path.join(PUBTABNET_DIR, "images")
    
    with open(annotations_path, 'r') as f:
        data_records = json.load(f)
        
    results = []
    print(f"Processing {len(data_records)} items...")
    
    for record in tqdm(data_records):
        if limit and len(results) >= limit:
            break
            
        try:
            filename = record['filename']
            gt_html_raw = record['html'] # Typically tokens in PubTabNet, but assuming string here per legacy
            imgid = record.get('imgid')
            image_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(image_path): continue
                
            # Inference via Engine
            try:
                pred_html_raw = engine.predict(image_path)
            except Exception as e:
                print(f"Inference fail {filename}: {e}")
                pred_html_raw = ""


            # Canonicalize + Score
            # Note: TEDS/Canonical logic is shared
            gt_canon = canonicalize_html(gt_html_raw)
            pred_canon = canonicalize_html(pred_html_raw)
            try:
                score = teds_calculator.evaluate(pred_canon, gt_canon)
            except:
                score = 0.0
                
            label = "PASS" if score >= TEDS_THRESHOLD else "FAIL"
            
            results.append({
                "filename": filename,
                "imgid": imgid,
                "teds_score": float(score),
                "label": label,
                "gt_html": gt_canon,
                "pred_html": pred_canon,
                "engine": engine_name
            })
            
        except Exception as e:
            print(f"Error {filename}: {e}")
            continue

    # --- Statistics Report ---
    print("\n" + "="*40)
    print(f"       STATISTICS REPORT ({engine_name})       ")
    print("="*40)
    
    total = len(results)
    perfect = sum(1 for r in results if r['teds_score'] >= 0.9999)
    fail = sum(1 for r in results if r['label'] == 'FAIL')
    
    if total > 0:
        pct_perfect = (perfect / total) * 100
        pct_fail = (fail / total) * 100
        print(f"Total Tables Processed: {total}")
        print(f"Perfect Scores (1.0):   {perfect} ({pct_perfect:.1f}%)")
        print(f"Failed Cases (<{TEDS_THRESHOLD}):    {fail} ({pct_fail:.1f}%)")
    else:
        print("No data processed.")
        
    print("="*40)
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to process (default 50)")
    parser.add_argument("--engine", type=str, default="paddle", choices=["paddle", "deepseek", "got"], help="OCR Engine to use")
    
    args = parser.parse_args()
    
    run_extraction(limit=args.limit, engine_name=args.engine)
