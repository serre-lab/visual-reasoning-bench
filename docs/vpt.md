# VPT Dataset Integration

This repository can evaluate models on the Visual Perspective Taking (VPT) benchmark released alongside the 3D-PC paper. The loader now streams data directly from Hugging Face (`3D-PC/3D-PC`), so you only need the `datasets` dependency and a bit of disk space for the hub cache.

## 1. Install dependencies

```bash
pip install -r requirements.txt  # includes datasets, pyarrow, etc.
```

If the dataset is gated for your account, run `huggingface-cli login` once so the hub cache can authenticate.

## 2. Configure the loader

`VPTDataset` accepts the following key parameters (see `configs/vpt_chatgpt.yaml`):

```yaml
dataset:
  name: vpt
  params:
    hf_dataset: "3D-PC/3D-PC"
    hf_config: "depth"       # depth | vpt-basic | vpt-strategy
    split: "validation"      # train | validation | test | human
    hf_cache_dir: null       # default: ~/.cache/huggingface
    limit: null              # set an integer for quick smoke tests
```

`VPTDataset` pulls the human-written prompt/statement directly from each example when available. For the `depth` config (which lacks explicit text), it deterministically alternates between the “green closer than red” and “red closer than green” questions so that models can’t memorize a single phrasing; the ground-truth answer is flipped automatically in the inverted case. Override `question_template`, `positive_answer`, or `negative_answer` only if you need a custom phrasing.

## 3. Run an evaluation

Pick the model backend that fits your environment:

```bash
# OpenRouter-hosted VLMs (preferred)
export OPENROUTER_API_KEY=sk-your-openrouter-key
python scripts/run_eval.py --config configs/vpt_openrouter.yaml --verbose

# Direct OpenAI access (optional)
export OPENAI_API_KEY=sk-your-openai-key
python scripts/run_eval.py --config configs/vpt_chatgpt.yaml --verbose
```

The loader yields samples with raw `image_bytes`, so models like `OpenRouterVisionModel` or `ChatGPTVisionModel` (both accept bytes and on-disk paths) can run inference without touching the filesystem. Swap the model block in either config to test additional wrappers as you integrate them.
