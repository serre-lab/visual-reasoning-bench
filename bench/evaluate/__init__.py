"""Evaluation module for VLM benchmarking."""

from .evaluator import Evaluator
from .metrics import compute_accuracy

__all__ = ["Evaluator", "compute_accuracy"]
