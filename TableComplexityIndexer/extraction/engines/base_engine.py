from abc import ABC, abstractmethod

class BaseEngine(ABC):
    """
    Abstract Base Class for Table Extraction Engines.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
    @abstractmethod
    def load_model(self):
        """
        Load weights and initialize the model.
        """
        pass
        
    @abstractmethod
    def predict(self, image_path: str) -> str:
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to the input image.
            
        Returns:
            str: Predicted HTML string of the table.
        """
        pass
