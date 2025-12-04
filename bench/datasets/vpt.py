"""Hugging Face loader for the Visual Perspective Taking (VPT) benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple

from .base import BaseDataset

try:
    from datasets import Image as HFImage
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'datasets' package is required to use VPTDataset. "
        "Install it via `pip install datasets`."
    ) from exc


class VPTDataset(BaseDataset):
    """Dataset wrapper that streams VPT splits from Hugging Face."""

    DEFAULT_QUESTIONS: Dict[str, str] = {
        "depth": "Is the green object closer to the camera than the red ball? Answer only with 'yes' or 'no'.",
        "vpt-basic": "Can the green observer see the red ball in this scenario?",
        "vpt-strategy": "Can the green observer see the red ball in this scenario?",
        "default": "Is the statement about this {task} scene true?",
    }

    SPLIT_ALIASES = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "validation": "validation",
        "dev": "validation",
        "test": "test",
        "human": "test",
    }

    def __init__(
        self,
        data_dir: str = ".",
        hf_dataset: str = "3D-PC/3D-PC",
        hf_config: str = "depth",
        split: str = "validation",
        hf_cache_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        question_template: Optional[str] = None,
        positive_answer: str = "yes",
        negative_answer: str = "no",
        limit: Optional[int] = None,
        shuffle_samples: bool = False,
        shuffle_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the loader.

        Args:
            data_dir: Unused placeholder to satisfy BaseDataset (keep default).
            hf_dataset: Dataset identifier on Hugging Face Hub.
            hf_config: Configuration name (e.g., ``depth``, ``vpt-basic``).
            split: Split name or alias (``train``, ``val``, ``test``).
            hf_cache_dir: Optional override for Hugging Face's cache directory.
            use_auth_token: Optional token for gated datasets.
            question_template: Format string describing the task; ``{task}`` expands
                to the config name. Falls back to sensible defaults per config.
            positive_answer: String to use for label ``1``.
            negative_answer: String to use for label ``0``.
            limit: Optional cap on the number of samples to load for debugging.
            shuffle_samples: Whether to shuffle before applying ``limit``. Useful to avoid
                always sampling the first N items.
            shuffle_seed: Optional seed for deterministic shuffling.
            **kwargs: Ignored additional params to stay compatible with BaseDataset.
        """
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self.split = split.lower()
        self.hf_cache_dir = hf_cache_dir
        self.use_auth_token = use_auth_token
        self.question_template = (
            question_template
            or self.DEFAULT_QUESTIONS.get(
                hf_config, self.DEFAULT_QUESTIONS["default"]
            )
        )
        self.positive_answer = positive_answer
        self.negative_answer = negative_answer
        self.limit = limit
        self.shuffle_samples = shuffle_samples
        self.shuffle_seed = shuffle_seed

        super().__init__(data_dir=data_dir, **kwargs)

    def _load_data(self) -> None:
        normalized_split = self._normalize_split(self.split)
        # Shuffle-aware loading: if shuffling, pull the whole split then shuffle/select.
        if self.shuffle_samples:
            dataset = load_dataset(
                self.hf_dataset,
                self.hf_config,
                split=normalized_split,
                cache_dir=self.hf_cache_dir,
                use_auth_token=self.use_auth_token,
            )
            dataset = dataset.shuffle(seed=self.shuffle_seed)
            if self.limit is not None:
                dataset = dataset.select(range(self.limit))
        else:
            split_expr = (
                f"{normalized_split}[:{self.limit}]"
                if self.limit is not None
                else normalized_split
            )
            dataset = load_dataset(
                self.hf_dataset,
                self.hf_config,
                split=split_expr,
                cache_dir=self.hf_cache_dir,
                use_auth_token=self.use_auth_token,
            )

        dataset = dataset.cast_column("image", HFImage(decode=False))

        for idx, example in enumerate(dataset):
            image_info = example["image"]
            image_bytes = image_info.get("bytes")
            if image_bytes is None:
                raise ValueError("Encountered image without byte content in VPT dataset.")

            filename_hint = example.get("img_id") or image_info.get("path") or f"{normalized_split}_{idx}.png"
            original_label = self._to_int_label(example["label"])
            question, adjusted_label, prompt_meta = self._build_prompt_and_label(
                example, original_label, idx
            )
            answer = self.positive_answer if adjusted_label == 1 else self.negative_answer
            sample_id = self._build_sample_id(idx, filename_hint, adjusted_label, normalized_split)

            metadata = {
                "hf_dataset": self.hf_dataset,
                "hf_config": self.hf_config,
                "split": normalized_split,
                "label": adjusted_label,
                "source_label": original_label,
                "category": example.get("category"),
                "scene": example.get("scene"),
                "setting": example.get("setting"),
                "img_id": example.get("img_id"),
                "shuffled": self.shuffle_samples,
                "shuffle_seed": self.shuffle_seed,
            }
            metadata.update(prompt_meta)

            self.samples.append(
                {
                    "id": sample_id,
                    "image_path": None,
                    "image_bytes": image_bytes,
                    "question": question,
                    "answer": answer,
                    "metadata": metadata,
                }
            )

    def _normalize_split(self, split: str) -> str:
        if split.lower() not in self.SPLIT_ALIASES:
            raise ValueError(
                f"Unsupported VPT split '{split}'. "
                f"Valid values: {', '.join(sorted(self.SPLIT_ALIASES.keys()))}"
            )
        return self.SPLIT_ALIASES[split.lower()]

    @staticmethod
    def _to_int_label(value) -> int:
        if isinstance(value, bool):
            return int(value)
        try:
            num = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unable to parse label value '{value}' as binary int.") from exc
        return 1 if num == 1 else 0

    def _build_sample_id(self, idx: int, filename: str, label: int, split: str) -> str:
        slug = Path(filename).stem.replace(" ", "_")
        return f"vpt-{self.hf_config}-{split}-{label}-{idx:05d}-{slug}"

    def _build_prompt_and_label(
        self, example: Dict[str, Any], label: int, idx: int
    ) -> Tuple[str, int, Dict[str, Any]]:
        dataset_question = self._resolve_dataset_question(example)
        if dataset_question:
            return dataset_question, label, {}

        if self.hf_config == "depth":
            return self._build_depth_prompt(example, label, idx)

        question = self.question_template.format(
            task=self.hf_config.replace("-", " ")
        )
        return question, label, {}

    def _resolve_dataset_question(self, example: Dict[str, Any]) -> Optional[str]:
        for key in ("prompt", "question", "statement"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _build_depth_prompt(
        self, example: Dict[str, Any], label: int, idx: int
    ) -> Tuple[str, int, Dict[str, Any]]:
        identifier = example.get("img_id") or str(idx)
        variant = self._deterministic_variant(identifier)

        if variant == 0:
            question = "Is the green object closer to the camera than the red ball?"
            adjusted_label = label
            variant_name = "green_vs_red"
        else:
            question = "Is the red ball closer to the camera than the green object?"
            adjusted_label = 1 - label
            variant_name = "red_vs_green"

        meta = {"prompt_variant": variant_name}
        return question, adjusted_label, meta

    @staticmethod
    def _deterministic_variant(identifier: str) -> int:
        digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
        return int(digest, 16) % 2
