# Mini VPT Sample

This directory holds a tiny, synthetic slice of the Visual Perspective Taking (VPT) dataset so that contributors can sanity-check the `VPTDataset` loader without downloading the full benchmark.

Structure:

```
samples/vpt_min/
├── train/
│   ├── yes/
│   │   └── perspective_yes.png
│   └── no/
│       └── perspective_no.png
├── test/
│   ├── yes/
│   │   └── test_yes.png
│   └── no/
│       └── test_no.png
├── train_perspective_balanced.csv
├── val_perspective_balanced.csv
├── test_perspective_balanced.csv
├── human_perspective_balanced.csv
├── train_depth_balanced.csv
├── val_depth_balanced.csv
├── test_depth_balanced.csv
└── human_depth_balanced.csv
```

The CSV manifests mirror the official VPT release: each file lists relative image paths (with respect to `train/` or `test/`) paired with a binary label. Use this sample for fast local tests, but switch `dataset.data_dir` to the downloaded VPT dataset when running real benchmarks.
