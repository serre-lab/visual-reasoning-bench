# Visual Reasoning Bench 🔍

A minimal Python scaffold for benchmarking Vision-Language Models (VLMs) on visual reasoning tasks.

## Overview

Visual Reasoning Bench provides a clean, modular architecture for evaluating VLMs. It includes:

- **Datasets**: Extensible dataset interface yielding `{id, image_path, question, answer}`
- **Models**: Base model class with `predict(image_path, question) → str` interface
- **Evaluation**: Pipeline for running models on datasets and computing accuracy metrics
- **Utilities**: I/O and image processing helpers

## Project Structure

```
visual-reasoning-bench/
├── bench/
│   ├── datasets/
│   │   ├── base.py          # Base dataset class
│   │   └── pathfinder.py    # Pathfinder visual reasoning dataset
│   ├── models/
│   │   ├── base.py          # Base model interface
│   │   └── llava.py         # LLaVA model wrapper
│   ├── evaluate/
│   │   ├── evaluator.py     # Evaluation pipeline
│   │   └── metrics.py       # Accuracy and other metrics
│   └── utils/
│       ├── io.py            # Config loading, result saving
│       └── images.py        # Image loading and preprocessing
├── scripts/
│   └── run_eval.py          # Main evaluation script
├── configs/
│   └── example.yaml         # Example configuration
└── website/
    └── index.html           # Project landing page
```

## Quick Start

### Installation

```bash
git clone https://github.com/serre-lab/visual-reasoning-bench.git
cd visual-reasoning-bench
```

### Running an Evaluation

```bash
python scripts/run_eval.py --config configs/example.yaml --verbose
```

### Command Line Options

- `--config`: Path to YAML configuration file (default: `configs/example.yaml`)
- `--output`: Path to save results JSON (default: `results/evaluation.json`)
- `--verbose`: Show progress bar during evaluation

## Configuration

Edit `configs/example.yaml` to customize your evaluation:

```yaml
dataset:
  name: pathfinder
  data_dir: ./data/pathfinder

model:
  name: llava
  model_path: null
  params:
    temperature: 0.0
    max_tokens: 512
```

## Architecture

### Dataset Interface

All datasets inherit from `BaseDataset` and must implement `_load_data()`:

```python
from bench.datasets import BaseDataset

class MyDataset(BaseDataset):
    def _load_data(self):
        self.samples = [
            {
                'id': 'sample_0',
                'image_path': '/path/to/image.png',
                'question': 'What do you see?',
                'answer': 'A cat'
            },
            # ... more samples
        ]
```

### Model Interface

All models inherit from `BaseModel` and must implement `predict()`:

```python
from bench.models import BaseModel

class MyModel(BaseModel):
    def predict(self, image_path: str, question: str) -> str:
        # Your inference code here
        prediction = self.model.generate(image_path, question)
        return prediction
```

### Evaluator

The evaluator runs a model on a dataset and computes metrics:

```python
from bench.datasets import PathfinderDataset
from bench.models import LLaVAModel
from bench.evaluate import Evaluator

dataset = PathfinderDataset(data_dir='./data/pathfinder')
model = LLaVAModel(model_path='path/to/checkpoint')
evaluator = Evaluator(model=model, dataset=dataset)

results = evaluator.evaluate(verbose=True)
print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
```

## Extending the Framework

### Adding a New Dataset

1. Create a new file in `bench/datasets/`
2. Inherit from `BaseDataset`
3. Implement `_load_data()` method
4. Register in `bench/datasets/__init__.py`

### Adding a New Model

1. Create a new file in `bench/models/`
2. Inherit from `BaseModel`
3. Implement `predict()` method
4. Register in `bench/models/__init__.py`

### Adding New Metrics

Add metric functions to `bench/evaluate/metrics.py`:

```python
def compute_f1_score(predictions, ground_truth):
    # Your metric implementation
    return f1_score
```

## Development

This is a scaffold implementation designed to be extended. Key areas for enhancement:

- **Dataset Loading**: Add proper data loading from various formats
- **Model Integration**: Integrate actual VLM implementations
- **Image Processing**: Add PIL/OpenCV for real image operations
- **Metrics**: Add more evaluation metrics (F1, BLEU, etc.)
- **Visualization**: Add result visualization tools

## License

MIT License (or specify your license)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{visual_reasoning_bench,
  title={Visual Reasoning Bench},
  author={Serre Lab},
  year={2024},
  url={https://github.com/serre-lab/visual-reasoning-bench}
}
```