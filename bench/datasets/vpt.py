"""Dataset loader for the Visual Perspective Taking (VPT) benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple

from .base import BaseDataset


class VPTDataset(BaseDataset):
    """Dataset for the 3D-PC / VPT benchmark released by Serre Lab."""

    VALID_TASKS = {"perspective", "depth"}
    VALID_SPLITS = {"train", "val", "test", "human"}
    DEFAULT_QUESTIONS = {
        "perspective": "Is the target object visible from the observer's viewpoint?",
        "depth": "Is the highlighted object closer to the camera than its partner?",
    }

    def __init__(
        self,
        data_dir: str,
        task: str = "perspective",
        split: str = "val",
        balanced: bool = True,
        question_template: str | None = None,
        positive_answer: str = "yes",
        negative_answer: str = "no",
        limit: int | None = None,
        **kwargs,
    ):
        """Load samples from a VPT CSV manifest.

        Args:
            data_dir: Directory containing train/test folders and CSV manifests.
            task: Either ``perspective`` or ``depth`` to pick which CSV to read.
            split: One of ``train``, ``val``, ``test``, or ``human``.
            balanced: Whether to read ``*_balanced.csv`` (default) or the raw CSV.
            question_template: Optional override for the prompt text. Can reference
                ``{task}`` which will expand to the task name.
            positive_answer: String to use for label ``1``.
            negative_answer: String to use for label ``0``.
            limit: Optional cap on how many samples to load (useful for smoke tests).
            **kwargs: Ignored but kept for compatibility with BaseDataset signature.
        """
        self.task = task.lower()
        self.split = split.lower()
        if self.task not in self.VALID_TASKS:
            raise ValueError(f"Unsupported VPT task: {task}")
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported VPT split: {split}")

        self.balanced = balanced
        self.question_template = (
            question_template or self.DEFAULT_QUESTIONS[self.task]
        )
        self.positive_answer = positive_answer
        self.negative_answer = negative_answer
        self.limit = limit

        super().__init__(data_dir=data_dir, **kwargs)

    def _load_data(self) -> None:
        csv_path = self._resolve_csv_path()
        base_dir = self._resolve_base_dir()
        rows = list(self._iter_rows(csv_path))

        if not rows:
            raise ValueError(f"No entries found in {csv_path}")

        for idx, (relative_path, label_value) in enumerate(rows):
            if self.limit is not None and len(self.samples) >= self.limit:
                break

            image_path = base_dir / relative_path
            label_int, answer = self._label_to_answer(label_value)
            question = self._format_question()
            sample_id = self._build_sample_id(idx, relative_path, label_int)

            self.samples.append(
                {
                    "id": sample_id,
                    "image_path": str(image_path),
                    "question": question,
                    "answer": answer,
                    "metadata": {
                        "task": self.task,
                        "split": self.split,
                        "label": label_int,
                        "relative_path": relative_path,
                    },
                }
            )

    def _resolve_csv_path(self) -> Path:
        suffix = "_balanced" if self.balanced else ""
        csv_name = f"{self.split}_{self.task}{suffix}.csv"
        candidate = Path(self.data_dir) / csv_name
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected CSV manifest not found: {candidate}"
            )
        return candidate

    def _resolve_base_dir(self) -> Path:
        split_dir = "train" if self.split in {"train", "val"} else "test"
        base_dir = Path(self.data_dir) / split_dir
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Base directory for split '{self.split}' not found: {base_dir}"
            )
        return base_dir

    def _iter_rows(self, csv_path: Path) -> Iterable[Tuple[str, str]]:
        with csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                if not row:
                    continue
                first = row[0].strip()
                if not first:
                    continue
                if idx == 0 and self._looks_like_header(row):
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"Malformed row in {csv_path}: expected at least 2 columns, got {row}"
                    )
                yield first, row[1].strip()

    @staticmethod
    def _looks_like_header(row: List[str]) -> bool:
        header_tokens = {"path", "image", "img", "file", "relative_path"}
        return row[0].strip().lower() in header_tokens

    def _label_to_answer(self, value: str) -> Tuple[int, str]:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "positive"}:
            label = 1
        elif normalized in {"0", "false", "no", "negative"}:
            label = 0
        else:
            try:
                label = int(float(normalized))
            except ValueError as exc:
                raise ValueError(f"Unrecognized label '{value}' in VPT CSV") from exc

        answer = self.positive_answer if label == 1 else self.negative_answer
        return label, answer

    def _format_question(self) -> str:
        return self.question_template.format(task=self.task.replace("_", " "))

    def _build_sample_id(self, idx: int, relative_path: str, label: int) -> str:
        slug = Path(relative_path).stem.replace(" ", "_")
        return f"vpt-{self.task}-{self.split}-{label}-{idx:05d}-{slug}"
