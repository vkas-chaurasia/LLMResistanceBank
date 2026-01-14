import torch
from transformers import LayoutLMv3Model, LayoutLMv3Processor
from PIL import Image
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model: {model_name} on {self.device}...")
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """
        Generates a visual embedding for the given image path.
        Returns a normalized numpy array of the embedding.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            # We only provide the image to the processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the CLS token embedding (first token) or average pooling
                # LayoutLMv3's visual backbone output. 
                # last_hidden_state shape: (batch_size, seq_len, hidden_size)
                # We typically take the embedding of the [CLS] token if available, or mean pooling.
                # LayoutLMv3 uses a CLS token.
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize the embedding for Cosine Similarity
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return None
