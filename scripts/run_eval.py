#!/usr/bin/env python
"""Main script for running VLM benchmark evaluations."""

import argparse
import sys
from pathlib import Path

# Add bench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.datasets import PathfinderDataset, VPTDataset
from bench.models import ChatGPTVisionModel, LLaVAModel
from bench.evaluate import Evaluator
from bench.utils import load_config, save_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM benchmark evaluation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/example.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation.json',
        help='Path to output results file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show progress bar during evaluation'
    )
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("VLM Benchmark Evaluation Pipeline")
    print("=" * 60)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Initialize dataset
    dataset_config = config['dataset']
    print(f"\nInitializing dataset: {dataset_config['name']}")
    
    dataset_params = dataset_config.get('params', {})
    
    if dataset_config['name'] == 'pathfinder':
        dataset = PathfinderDataset(
            data_dir=dataset_config['data_dir'],
            **dataset_params
        )
    elif dataset_config['name'] == 'vpt':
        dataset = VPTDataset(
            data_dir=dataset_config['data_dir'],
            **dataset_params
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_config['name']}")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Initialize model
    model_config = config['model']
    print(f"\nInitializing model: {model_config['name']}")
    
    if model_config['name'] == 'llava':
        model = LLaVAModel(
            model_path=model_config.get('model_path'),
            **model_config.get('params', {})
        )
    elif model_config['name'] == 'chatgpt':
        model = ChatGPTVisionModel(
            **model_config.get('params', {})
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    evaluator = Evaluator(model=model, dataset=dataset)
    results = evaluator.evaluate(verbose=args.verbose)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Model: {results['model']['model_name']}")
    print(f"Dataset size: {results['dataset_size']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"Correct: {results['metrics']['correct']}/{results['metrics']['total_samples']}")
    
    # Save results
    save_results(results, args.output)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
