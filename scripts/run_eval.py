#!/usr/bin/env python
"""Main script for running VLM benchmark evaluations."""

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Dict, List

# Add bench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.datasets import PathfinderDataset, VPTDataset
from bench.models import LLaVAModel, OpenRouterVisionModel
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
        default='results',
        help='Path to output file or directory (per-model files when multiple models are provided)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Optional run identifier. Defaults to <model>-<benchmark>-<timestamp>.'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='Number of concurrent in-flight model calls (async).'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show progress bar during evaluation'
    )
    return parser.parse_args()

def _normalize_models(config: Dict) -> List[Dict]:
    """Return a list of model config blocks."""
    if 'models' in config:
        if not isinstance(config['models'], list):
            raise ValueError("`models` must be a list when provided.")
        return config['models']
    if 'model' in config:
        return [config['model']]
    raise ValueError("Configuration must include `model` or `models`.")


def _build_model(model_config: Dict):
    """Instantiate a model based on the config block."""
    name = model_config.get('name', 'openrouter')
    params = model_config.get('params', {})

    if name == 'llava':
        return LLaVAModel(
            model_path=model_config.get('model_path'),
            **params
        )
    if name == 'openrouter':
        return OpenRouterVisionModel(**params)

    raise ValueError(f"Unknown model: {name}")


def _model_label(model_config: Dict) -> str:
    params = model_config.get('params', {})
    return params.get('model_slug') or model_config.get('name', 'model')


def _safe_label(text: str) -> str:
    return "".join(c if c.isalnum() or c in ['-', '_'] else "_" for c in text)


def _resolve_output_path(base_output: Path, identifier: str, multiple: bool) -> Path:
    """Derive a per-model output path while keeping single-model behavior reasonable."""
    safe_label = _safe_label(identifier)

    if base_output.suffix and not base_output.is_dir():
        if multiple:
            return base_output.with_name(f"{base_output.stem}_{safe_label}{base_output.suffix}")
        return base_output

    return (base_output if base_output.suffix == '' else base_output.parent) / f"{safe_label}.json"


def _with_default_system_prompt(model_config: Dict, dataset_name: str) -> Dict:
    """Ensure models have a concise system prompt tailored to the dataset."""
    params = model_config.get('params', {})
    if params.get('system_prompt'):
        return model_config

    prompt = (
        f"You are answering questions for the {dataset_name} dataset. "
        "Answer with a single word: yes/no. Do not explain."
    )
    updated = dict(model_config)
    updated_params = dict(params)
    updated_params['system_prompt'] = prompt
    updated['params'] = updated_params
    return updated


def _dataset_descriptor(dataset_config: Dict) -> Dict:
    """Standardize dataset metadata for output files."""
    params = dataset_config.get('params', {})
    dataset_name = dataset_config['name']
    hf_config = params.get('hf_config')
    split = params.get('split')

    parts = [dataset_name]
    if hf_config:
        parts.append(str(hf_config))
    if split:
        parts.append(str(split))

    dataset_id = dataset_config.get('id') or "-".join(parts)

    return {
        'id': dataset_id,
        'name': dataset_name,
        'split': split,
        'hf_config': hf_config,
        'data_dir': dataset_config.get('data_dir'),
        'params': params
    }


def _default_run_id(model_label: str, dataset_meta: Dict) -> str:
    ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    return f"{_safe_label(model_label)}-{_safe_label(dataset_meta['id'])}-{ts}"


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
    
    model_blocks = _normalize_models(config)
    multiple = len(model_blocks) > 1
    output_base = Path(args.output)
    all_results = []

    for idx, model_config in enumerate(model_blocks, start=1):
        model_config = _with_default_system_prompt(model_config, dataset_config['name'])
        label = _model_label(model_config)
        print(f"\n[{idx}/{len(model_blocks)}] Initializing model: {label}")
        model = _build_model(model_config)

        print("Running evaluation...")
        evaluator = Evaluator(model=model, dataset=dataset)
        results = evaluator.evaluate(verbose=args.verbose, concurrency=max(1, args.concurrency))

        dataset_meta = _dataset_descriptor(dataset_config)
        run_id = args.run_id or config.get('run_id') or _default_run_id(label, dataset_meta)
        timestamp = datetime.now(timezone.utc).isoformat()

        results.update({
            'run_id': run_id,
            'timestamp': timestamp,
            'dataset': dataset_meta,
            'config_path': args.config,
        })
        all_results.append(results)

        # Print results
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Run ID: {results['run_id']}")
        print(f"Model: {results['model']['model_name']}")
        print(f"Dataset size: {results['dataset_size']}")
        print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
        print(f"Correct: {results['metrics']['correct']}/{results['metrics']['total_samples']}")

        # Save results per model
        output_path = _resolve_output_path(output_base, run_id, multiple)
        save_results(results, str(output_path))
    
    if multiple:
        print("\nSummary:")
        for res in all_results:
            model_id = res['model']['config'].get('model_slug') or res['model']['model_name']
            metrics = res['metrics']
            print(f"- {model_id}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total_samples']})")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
