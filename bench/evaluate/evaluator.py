"""Evaluator for running VLM benchmarks."""

from typing import Dict, Any, List
from tqdm import tqdm

from ..datasets.base import BaseDataset
from ..models.base import BaseModel
from .metrics import compute_metrics


class Evaluator:
    """Evaluator for running VLM models on benchmark datasets."""
    
    def __init__(self, model: BaseModel, dataset: BaseDataset):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
        """
        self.model = model
        self.dataset = dataset
    
    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """Run evaluation on the dataset.
        
        Args:
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        predictions = []
        ground_truth = []
        sample_results = []
        
        iterator = tqdm(self.dataset, desc="Evaluating") if verbose else self.dataset
        
        for sample in iterator:
            # Get prediction from model
            prediction = self.model.predict(
                image_path=sample['image_path'],
                question=sample['question']
            )
            
            predictions.append(prediction)
            ground_truth.append(sample['answer'])
            
            sample_results.append({
                'id': sample['id'],
                'question': sample['question'],
                'prediction': prediction,
                'ground_truth': sample['answer'],
                'correct': prediction.lower().strip() == sample['answer'].lower().strip()
            })
        
        # Compute metrics
        metrics = compute_metrics(predictions, ground_truth)
        
        return {
            'model': self.model.get_info(),
            'dataset_size': len(self.dataset),
            'metrics': metrics,
            'sample_results': sample_results
        }
