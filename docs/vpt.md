# VPT Dataset Integration

This repository can evaluate models on the Visual Perspective Taking (VPT) benchmark released alongside the 3D-PC paper. Follow the steps below to make the data available locally and to connect it with `VPTDataset`.

## 1. Obtain the data

You have two primary download options (use whichever is easier for you):

- **Direct download:** grab the archives from the Brown mirror → https://connectomics.clps.brown.edu/tf_records/VPT/
- **Hugging Face:** `datasets.load_dataset("pzhou10/3D-PC", "vpt-basic")` (requires `datasets` + an auth token for large downloads).

Each archive expands to a directory that contains:

```
<vpt_root>/
├── train/                         # image folders (used for train + val splits)
├── test/                          # held-out splits (test + human in the paper)
├── train_perspective.csv          # manifests that map filenames → labels
├── train_perspective_balanced.csv
├── val_perspective_balanced.csv
├── test_perspective_balanced.csv
├── human_perspective_balanced.csv
├── train_depth*.csv               # analogous CSVs for the depth task
└── ...
```

If you just want to sanity-check the loader, this repository also includes `samples/vpt_min`, a synthetic slice with the same structure and CSV naming convention.

## 2. Point the config to your data

Update `dataset.data_dir` inside `configs/vpt_chatgpt.yaml` (or any other config) so that it references the root shown above:

```yaml
dataset:
  name: vpt
  data_dir: /path/to/VPT
  params:
    task: perspective      # or depth
    split: val             # train | val | test | human
    balanced: true         # set false to read the raw CSVs
```

`VPTDataset` automatically reads the correct CSV (e.g., `val_perspective_balanced.csv`), joins the relative file paths with `train/` or `test/`, and emits `{id, image_path, question, answer}` samples that the evaluation pipeline understands.

## 3. Run an evaluation

With an OpenAI key exported, run:

```bash
python scripts/run_eval.py --config configs/vpt_chatgpt.yaml --verbose
```

Swap the model block if you want to test other wrappers (e.g., LLaVA once you have local weights). The evaluator will print accuracy and store per-sample logs under `results/`.
