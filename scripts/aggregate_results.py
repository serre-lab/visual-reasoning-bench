#!/usr/bin/env python
"""Aggregate per-run evaluation outputs into a central index for the website."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.utils import load_json


PRIMARY_METRIC_ORDER = ("accuracy", "acc", "score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate result JSON files into a central index."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing per-run result JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/central_index.json",
        help="Path to write the aggregated index JSON.",
    )
    return parser.parse_args()


def _primary_metric(metrics: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    for key in PRIMARY_METRIC_ORDER:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, float(metrics[key])

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return key, float(value)

    return "score", None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None




def _extract_entry(path: Path) -> Dict[str, Any]:
    data = load_json(str(path))
    metrics = data.get("metrics", {}) or {}
    dataset = data.get("dataset", {}) or {}
    if not dataset:
        raise ValueError(
            f"Missing dataset metadata in {path.name}. "
            "Re-run evaluation so results include the `dataset` block."
        )
    model = data.get("model", {}) or {}

    metric_name, score = _primary_metric(metrics)
    dataset_size = data.get("dataset_size") or metrics.get("total_samples")
    benchmark_id = dataset.get("id") or dataset.get("name") or "unknown-dataset"
    benchmark_name = dataset.get("id") or dataset.get("name") or benchmark_id

    run_id = data.get("run_id") or path.stem
    timestamp = data.get("timestamp")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_name,
        "dataset_split": dataset.get("split"),
        "hf_config": dataset.get("hf_config"),
        "metric": metric_name,
        "score": _safe_float(score),
        "dataset_size": dataset_size,
        "model_name": model.get("config", {}).get("model_slug")
        or model.get("model_name")
        or "unknown-model",
        "provider": model.get("model_name"),
        "latency": metrics.get("latency"),
        "timestamp": timestamp,
        "config_path": data.get("config_path"),
        "path": str(path),
    }


def collect_results(results_dir: Path, index_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for path in sorted(results_dir.glob("*.json")):
        if path.name == index_path.name or "central_index" in path.name:
            continue
        try:
            entries.append(_extract_entry(path))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Skipping {path}: {exc}")

    entries.sort(
        key=lambda e: e.get("timestamp", "") or e.get("run_id", ""),
        reverse=True,
    )
    return entries


def dedupe_latest(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the latest run per (benchmark_id, model_name)."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    for run in runs:
        key = (run.get("benchmark_id"), run.get("model_name"))
        if key not in latest:
            latest[key] = run
            latest[key]["is_latest"] = True
        # since runs are sorted newest-first, the first seen is the latest
    return list(latest.values())


def build_benchmarks(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group runs by benchmark for the website."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        bench_id = run["benchmark_id"]
        bucket = grouped.setdefault(
            bench_id,
            {
                "id": bench_id,
                "name": run.get("benchmark_name") or bench_id,
                "metric": run.get("metric") or "score",
                "dataset_size": run.get("dataset_size"),
                "description": "",
                "models": [],
            },
        )
        bucket.setdefault("dataset_size", run.get("dataset_size"))
        bucket["models"].append(
            {
                "name": run.get("model_name"),
                "provider": run.get("provider"),
                "score": run.get("score"),
                "latency": run.get("latency"),
                "run_id": run.get("run_id"),
                "config": run.get("config_path"),
            }
        )
    for bucket in grouped.values():
        bucket["models"].sort(
            key=lambda m: (m.get("score") is None, -(m.get("score") or 0.0))
        )
    return list(grouped.values())


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    runs = collect_results(results_dir, output_path)
    runs = dedupe_latest(runs)
    index = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "benchmarks": build_benchmarks(runs),
    }

    import json

    output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs)} runs to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
