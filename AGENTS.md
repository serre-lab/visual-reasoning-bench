# Repository Guidelines

## Project Structure & Module Organization
Core logic lives in `bench/` with submodules for `datasets/`, `models/`, `evaluate/`, and `utils/`. Dataset classes such as `PathfinderDataset` and `VPTDataset` inherit from `bench/datasets/base.py` and load samples from `data_dir`. Models extend `bench/models/base.py` and expose `predict(image_path, question)`. Metrics and evaluation flow sit in `bench/evaluate/{evaluator,metrics}.py`, while helper I/O lives in `bench/utils`. Configuration files stay under `configs/` (e.g., `configs/example.yaml`, `configs/vpt_chatgpt.yaml`), executable entry points under `scripts/`, and the static marketing site in `website/`. Synthetic smoke-test assets live in `samples/` (notably `samples/vpt_min` for VPT).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create a local isolated environment.
- `pip install -r requirements.txt`: install the minimal runtime (PyYAML, tqdm) before running any scripts.
- `python scripts/run_eval.py --config configs/example.yaml --verbose`: execute an evaluation pipeline end to end; swap the config to exercise new datasets or models.
- `python scripts/run_eval.py --config configs/vpt_chatgpt.yaml --verbose`: run ChatGPT on the VPT sample set (point `dataset.data_dir` at the full download for real experiments).
- `PYTHONPATH=. python -m pytest bench/tests`: run contributor-authored tests once they exist so CI can mirror that workflow.

## Coding Style & Naming Conventions
Python files use 4-space indentation, type hints, and module-level docstrings (see `bench/datasets/base.py`). Keep public methods documented with short docstrings that explain inputs/outputs. Class names follow `PascalCase` (`LLaVAModel`, `BaseDataset`), helper functions use `snake_case`, and metric helpers begin with `compute_`. Favor `Path`/`str` inputs over global state and use f-strings for formatting. When adding tools such as `ruff` or `black`, run them before committing and mention any intentional deviations in the PR.

## Testing Guidelines
Prefer `pytest` for new coverage. Organize regression tests under `bench/tests/` mirroring the module tree (e.g., `bench/tests/test_evaluator.py`). Test datasets with small synthetic samples (**see `samples/vpt_min`**) and ensure iterator lengths are deterministic. For models, mock inference calls so tests remain CPU-only. Whenever you add a metric, include golden-value tests that assert the expected float with a tolerance (e.g., `pytest.approx`). Aim for covering new logic with focused unit tests plus at least one integration shard invoking `scripts/run_eval.py` against `configs/example.yaml` (or the VPT config when relevant).

## Commit & Pull Request Guidelines
Existing history uses short imperative subjects (“Create minimal Python scaffold…”). Follow that pattern, limit subjects to ~72 characters, and include a concise body when rationale is non-obvious. For pull requests, provide: (1) a summary of behavior change, (2) how you validated it (commands, dataset slices), (3) linked issues or roadmap items, and (4) screenshots or logs when the website or CLI output changes. Run lint/tests locally and paste the command outputs so reviewers can replicate quickly.

## Configuration & Security Tips
Never commit real dataset paths or credentials into `configs/*.yaml`; instead document placeholders (`/path/to/data`). Use environment variables for secrets and reference them inside configs if needed. Results written through `bench/utils/io.py` may contain sensitive predictions—store them under `results/` (gitignored) and sanitize before sharing.
