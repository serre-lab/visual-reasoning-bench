"""Metrics computation for VLM benchmarking."""

from typing import List, Dict, Any


def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute accuracy metric.
    
    Args:
        predictions: List of predicted answers
        ground_truth: List of ground truth answers
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    # Normalize answers for comparison (lowercase, strip whitespace)
    normalized_preds = [p.lower().strip() for p in predictions]
    normalized_gt = [g.lower().strip() for g in ground_truth]
    
    correct = sum(p == g for p, g in zip(normalized_preds, normalized_gt))
    return correct / len(predictions)


def compute_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    """Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted answers
        ground_truth: List of ground truth answers
        
    Returns:
        Dictionary containing all computed metrics
    """
    accuracy = compute_accuracy(predictions, ground_truth)
    
    return {
        'accuracy': accuracy,
        'total_samples': len(predictions),
        'correct': int(accuracy * len(predictions))
    }
