import os
import json
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add shared directory (2 levels up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from shared.embedding_model import EmbeddingModel
from shared.vector_db import VectorDB
from TableComplexityIndexer.config import (
    OUTPUT_DIR, 
    EMBEDDING_MODEL_NAME, 
    INDEX_FILE, 
    METADATA_FILE
)

# Input JSON is in output directory
INPUT_JSON = os.path.join(os.path.dirname(__file__), "../output/processing_results.json")

def main():
    print("=== Phase 2: Embedding & Indexing (Torch/FAISS) ===")
    
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file {INPUT_JSON} not found. Run step1_extract.py first.")
        return

    # 1. Initialize Models
    print(f"Loading Embedding Model ({EMBEDDING_MODEL_NAME})...")
    try:
        embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
        # Force CPU if CUDA issues persist, or use GPU if available in this env
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Initializing Vector DB...")
    vector_db = VectorDB(dimension=768, index_path=INDEX_FILE, metadata_path=METADATA_FILE)

    # 2. Load Data
    with open(INPUT_JSON, 'r') as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records for indexing...")
    
    # 3. Process
    success_count = 0
    
    for record in tqdm(records):
        try:
            image_path = record.get('image_path')
            if not image_path or not os.path.exists(image_path):
                print(f"Image missing: {image_path}")
                continue
                
            # Generate Embedding
            vector = embedder.generate_embedding(image_path)
            
            if vector is not None:
                # Add to DB
                meta = {
                    "filename": record['filename'],
                    "imgid": record['imgid'],
                    "teds_score": record['teds_score'],
                    "label": record['label'],
                    "gt_html": record['gt_html'],
                    "pred_html": record['pred_html']
                }
                vector_db.add(vector, meta)
                success_count += 1
                
        except Exception as e:
            print(f"Error indexing {record.get('filename')}: {e}")
            continue
            
    # 4. Save
    vector_db.save_index()
    print("\n" + "="*40)
    print("       INDEXING COMPLETE       ")
    print("="*40)
    print(f"Total Vectors Stored: {vector_db.get_total_items()}")
    print(f"Index Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
