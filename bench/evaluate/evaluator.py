"""Evaluator for running VLM benchmarks."""

import asyncio
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

    def evaluate(self, verbose: bool = True, concurrency: int = 1) -> Dict[str, Any]:
        """Run evaluation on the dataset.

        Args:
            verbose: Whether to show progress bar.
            concurrency: Number of concurrent in-flight predictions. When >1, the
                evaluation runs asynchronously using a semaphore to bound concurrency.

        Returns:
            Dictionary containing evaluation results and metrics.
        """
        if concurrency <= 1:
            return self._evaluate_sync(verbose=verbose)
        return asyncio.run(self._evaluate_async(verbose=verbose, concurrency=concurrency))

    def _evaluate_sync(self, verbose: bool) -> Dict[str, Any]:
        predictions: List[str] = []
        ground_truth: List[str] = []
        sample_results: List[Dict[str, Any]] = []

        iterator = tqdm(self.dataset, desc="Evaluating") if verbose else self.dataset

        for sample in iterator:
            raw_output, prediction, extras = self._run_model(sample)
            predictions.append(prediction)
            ground_truth.append(sample['answer'])

            sample_results.append(self._build_sample_result(sample, prediction, raw_output, extras))

        metrics = compute_metrics(predictions, ground_truth)

        return {
            'model': self.model.get_info(),
            'dataset_size': len(self.dataset),
            'metrics': metrics,
            'sample_results': sample_results
        }

    async def _evaluate_async(self, verbose: bool, concurrency: int) -> Dict[str, Any]:
        predictions: List[str] = []
        ground_truth: List[str] = []
        sample_results: List[Dict[str, Any]] = []

        samples = list(self.dataset)
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(idx: int, sample: Dict[str, Any]):
            async with semaphore:
                raw_output, prediction, extras = await asyncio.to_thread(self._run_model, sample)
                return idx, sample, raw_output, prediction, extras

        tasks = [asyncio.create_task(run_one(idx, sample)) for idx, sample in enumerate(samples)]

        if verbose:
            try:
                from tqdm.asyncio import tqdm as async_tqdm
                ordered: List[Tuple[int, Dict[str, Any], str, str]] = [None] * len(tasks)  # type: ignore
                async for finished in async_tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Evaluating"
                ):
                    idx, sample, raw_output, prediction, extras = await finished
                    ordered[idx] = (idx, sample, raw_output, prediction, extras)
                results = ordered
            except Exception:
                # Fallback: gather without async tqdm
                results = await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)

        # Restore dataset order
        results = sorted(results, key=lambda x: x[0])

        for _, sample, raw_output, prediction, extras in results:
            predictions.append(prediction)
            ground_truth.append(sample['answer'])
            sample_results.append(self._build_sample_result(sample, prediction, raw_output, extras))

        metrics = compute_metrics(predictions, ground_truth)

        return {
            'model': self.model.get_info(),
            'dataset_size': len(self.dataset),
            'metrics': metrics,
            'sample_results': sample_results
        }

    @staticmethod
    def _build_sample_result(
        sample: Dict[str, Any],
        prediction: str,
        raw_output: str,
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = {
            'id': sample['id'],
            'question': sample['question'],
            'prediction': prediction,
            'raw_output': raw_output,
            'ground_truth': sample['answer'],
            'correct': prediction.lower().strip() == sample['answer'].lower().strip()
        }
        if extras:
            result.update(extras)
        return result

    def _run_model(self, sample: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Execute the model and normalize return value."""
        extras: Dict[str, Any] = {}
        result = self.model.predict(
            image_path=sample.get('image_path'),
            question=sample['question'],
            image_bytes=sample.get('image_bytes')
        )

        if isinstance(result, dict):
            prediction = result.get('prediction')
            raw_output = result.get('raw_output', prediction)
            if 'reasoning_details' in result:
                extras['reasoning_details'] = result['reasoning_details']
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

        return raw_output, prediction, extras
