import faiss
import numpy as np
import pickle
import os

class VectorDB:
    def __init__(self, dimension=768, index_path="index.faiss", metadata_path="index_meta.pkl"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata = [] # List of dicts, index corresponds to FAISS ID
        
        # Initialize IndexFlatIP for Cosine Similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
    
    def add(self, vector: np.ndarray, meta: dict):
        """
        Adds a single vector and its metadata to the index.
        vector: Normalized numpy array of shape (dimension,)
        meta: Dictionary containing labels (e.g., {'label': 'PASS', 'image_id': '...'})
        """
        if vector is None:
            return

        vector = vector.reshape(1, -1).astype(np.float32)
        self.index.add(vector)
        self.metadata.append(meta)
    
    def search(self, vector: np.ndarray, k=5):
        """
        Searches for the k nearest neighbors.
        Returns distances and metadata of neighbors.
        """
        vector = vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "distance": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })
        return results

    def save_index(self):
        """Persist index and metadata to disk."""
        print(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        
        print(f"Saving metadata to {self.metadata_path}...")
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
    def load_index(self):
        """Load index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            
            print(f"Loading metadata from {self.metadata_path}...")
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        else:
            print("Index or metadata file not found. Starting with empty index.")
            return False

    def get_total_items(self):
        return self.index.ntotal
