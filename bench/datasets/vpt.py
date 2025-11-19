"""Hugging Face loader for the Visual Perspective Taking (VPT) benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
        "vpt-basic": "From the observer's viewpoint, can they see the target object?",
        "vpt-strategy": "Does the observer maintain line of sight to the target in this scenario?",
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

        super().__init__(data_dir=data_dir, **kwargs)

    def _load_data(self) -> None:
        normalized_split = self._normalize_split(self.split)
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
            label = self._to_int_label(example["label"])
            answer = self.positive_answer if label == 1 else self.negative_answer
            sample_id = self._build_sample_id(idx, filename_hint, label, normalized_split)

            question = self._resolve_question(example)
            metadata = {
                "hf_dataset": self.hf_dataset,
                "hf_config": self.hf_config,
                "split": normalized_split,
                "label": label,
                "category": example.get("category"),
                "scene": example.get("scene"),
                "setting": example.get("setting"),
                "img_id": example.get("img_id"),
            }

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

    def _resolve_question(self, example: Dict[str, any]) -> str:
        """Prefer dataset-provided textual prompts, fall back to template."""
        for key in ("prompt", "question", "statement"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self.question_template.format(
            task=self.hf_config.replace("-", " ")
        )
