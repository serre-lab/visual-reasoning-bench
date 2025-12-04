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

### VPT Integration

The `VPTDataset` streams directly from Hugging Face (`3D-PC/3D-PC`). Install the `datasets` package (included in `requirements.txt`), then pick one of the configs below and run the evaluation script.

- `configs/vpt_openrouter.yaml`: routes prompts through OpenRouter (preferred for hosted VLMs).
- `configs/vpt_chatgpt.yaml`: talks to OpenAI's API directly (useful for local experiments).

Tweak either config to choose `hf_config` (`depth`, `vpt-basic`, or `vpt-strategy`), pick a split (`train`, `validation`, `test`, `human`), or set `limit` for quick smoke tests. The loader automatically uses the dataset-provided prompt/statement when available; for `depth` it deterministically alternates between “green closer than red?” and the inverted phrasing, flipping the ground-truth answer accordingly. Images stay in memory as raw bytes, so any model wrapper that accepts `image_bytes` can benchmark VPT without extra preprocessing.

### OpenRouter Vision Models

Set your OpenRouter credentials and run the config that targets the OpenRouter API:

```bash
export OPENROUTER_API_KEY=sk-your-openrouter-key
python scripts/run_eval.py --config configs/vpt_openrouter.yaml --verbose
```

`configs/vpt_openrouter.yaml` lets you swap `model_slug` (e.g., `openai/gpt-4o-mini`, `google/gemini-1.5-pro`), adjust decoding params, and pass headers such as `http_referer` or `x_title` if your OpenRouter account requires them.

Prefer to call OpenAI directly? Export `OPENAI_API_KEY` and point to `configs/vpt_chatgpt.yaml` instead.

### ChatGPT Vision Demo

Set `OPENAI_API_KEY` (or pass `api_key` to the class) and run:

```bash
export OPENAI_API_KEY=sk-your-key
python scripts/demo_chatgpt_vlm.py --question "What color is this square?"
```

The script instantiates `ChatGPTVisionModel`, feeds `assets/demo_red_square.png`, and prints a real response from the ChatGPT VLM. Adjust `--image`, `--openai-model`, and decoding params to probe other prompts.

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
                'image_bytes': None,  # Use raw bytes when no local path exists
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
    def predict(self, image_path: str | None, question: str, image_bytes: bytes | None = None) -> str:
        # Your inference code here
        use_bytes = image_bytes if image_bytes is not None else open(image_path, 'rb').read()
        prediction = self.model.generate(use_bytes, question)
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
