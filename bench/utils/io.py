"""I/O utilities for configuration and results."""

import json
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to output JSON file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def load_json(json_path: str) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
