"""Base dataset class for VLM benchmarking."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Any


class BaseDataset(ABC):
    """Base class for all VLM benchmark datasets.
    
    Datasets should yield samples with the following structure:
    {
        'id': str,           # Unique identifier for the sample
        'image_path': str,   # Path to the image file
        'question': str,     # Question to ask about the image
        'answer': str        # Ground truth answer
    }
    """
    
    def __init__(self, data_dir: str, **kwargs):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing dataset files
            **kwargs: Additional dataset-specific arguments
        """
        self.data_dir = data_dir
        self.samples = []
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset samples. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        return self.samples[idx]
