import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PUBTABNET_DIR = os.path.join(DATA_DIR, "pubtabnet")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TEDS
TEDS_THRESHOLD = 0.95

# Model
EMBEDDING_MODEL_NAME = "microsoft/layoutlmv3-base"

# Vector DB
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "index_meta.pkl")

# Dataset (HuggingFace)
HF_DATASET_ID = "apoidea/pubtabnet-html" 
# Alternative: "leoxong/pubtabnet" if apoidea fails validation

# Router Thresholds (Derived from Pilot Analysis)
# Tables exceeding EITHER of these values in the *Predicted* HTML are sent to LLM.
ROUTER_AVG_TEXT_LEN_THRESHOLD = 10.64
ROUTER_TOTAL_TEXT_LEN_THRESHOLD = 485
