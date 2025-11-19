"""Evaluator for running VLM benchmarks."""

from typing import Any, Dict, List, Tuple

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
        predictions: List[str] = []
        ground_truth: List[str] = []
        sample_results: List[Dict[str, Any]] = []

        iterator = tqdm(self.dataset, desc="Evaluating") if verbose else self.dataset

        for sample in iterator:
            raw_output, prediction = self._run_model(sample)

            predictions.append(prediction)
            ground_truth.append(sample['answer'])

            sample_results.append({
                'id': sample['id'],
                'question': sample['question'],
                'prediction': prediction,
                'raw_output': raw_output,
                'ground_truth': sample['answer'],
                'correct': prediction.lower().strip() == sample['answer'].lower().strip()
            })

        metrics = compute_metrics(predictions, ground_truth)

        return {
            'model': self.model.get_info(),
            'dataset_size': len(self.dataset),
            'metrics': metrics,
            'sample_results': sample_results
        }

    def _run_model(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Execute the model and normalize return value."""
        result = self.model.predict(
            image_path=sample.get('image_path'),
            question=sample['question'],
            image_bytes=sample.get('image_bytes')
        )

        if isinstance(result, dict):
            prediction = result.get('prediction')
            raw_output = result.get('raw_output', prediction)
        elif isinstance(result, (list, tuple)):
            prediction = result[0]
            raw_output = result[1] if len(result) > 1 else prediction
        else:
            prediction = result
            raw_output = result

        if not isinstance(prediction, str):
            prediction = str(prediction)
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)

        return raw_output, prediction
