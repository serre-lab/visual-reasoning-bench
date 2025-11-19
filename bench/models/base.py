"""Base model class for VLM benchmarking."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Base class for all VLM models.
    
    All models must implement the predict method that takes an image path
    and a question, and returns a string prediction.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def predict(
        self,
        image_path: Optional[str],
        question: str,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """Generate a prediction for the given image and question.
        
        Args:
            image_path: Path to the image file (if available)
            question: Question to ask about the image
            image_bytes: Raw image bytes when no local path exists
            
        Returns:
            String prediction/answer from the model
        """
        raise NotImplementedError
    
    def batch_predict(self, samples: list) -> list:
        """Generate predictions for multiple samples.
        
        Args:
            samples: List of (image_path, question) tuples
            
        Returns:
            List of string predictions
        """
        predictions = []
        for image_path, question in samples:
            pred = self.predict(image_path, question)
            predictions.append(pred)
        return predictions
    
    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            'model_name': self.model_name,
            'config': self.config
        }
