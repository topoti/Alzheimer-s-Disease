"""Data layer for the OASIS MRI 4-class task (Phase 3).

This module is intentionally narrow — it owns three things and nothing else:

* :func:`parse_subject_id` — pull the ``OAS1_XXXX`` patient ID out of a slice filename.
* :class:`OasisDataset` — a ``torch`` ``Dataset`` that turns ``(root, file_list, labels)``
  into ``(image_tensor, label)`` pairs, applying the Phase 3 preprocessing pipeline.
* :func:`get_transforms` — build the albumentations pipeline for a given input size.

The **patient-level split** (which subjects go to train/val/test) lives in ``src/split.py``,
not here; this module just consumes the file lists that the split produces. Dataloaders are
assembled at the call site (the training notebook / ``src/train.py``) from an
:class:`OasisDataset` plus the split's per-class file lists.

Preprocessing pipeline (RESEARCH_PLAN.md, Phase 3):
    1. load slice
    2. resize to ``input_size`` (224 for ResNet50/DenseNet121, 300 for EfficientNetB3)
    3. grayscale -> duplicate to 3 channels (ImageNet backbones expect RGB)
    4. ImageNet normalization
    5. *train only* — light online augmentation: rotation ±10°, horizontal flip,
       brightness ±10%

Heavy dependencies (albumentations, cv2, torch, PIL) are imported lazily inside the
functions/methods so that lightweight consumers — e.g. ``src/split.py`` importing only
:func:`parse_subject_id` — do not have to load the deep-learning stack.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

# Subclass torch's Dataset when torch is available, but fall back to ``object`` so that
# lightweight consumers (e.g. ``src/split.py`` importing only :func:`parse_subject_id`) can
# import this module without the deep-learning stack installed. ``OasisDataset`` is only
# ever *instantiated* where torch is present.
try:  # pragma: no cover - depends on environment
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:  # pragma: no cover
    _DatasetBase = object  # type: ignore[assignment,misc]

# ImageNet statistics — the backbones are ImageNet-pretrained, so inputs must be normalized
# with the same mean/std the weights were trained under.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# The two locked input sizes (RESEARCH_PLAN.md, Phase 2). ``input_size`` is a plain int
# parameter everywhere in this module, so any value works; these name the canonical two.
INPUT_SIZE_DEFAULT = 224          # ResNet50, DenseNet121
INPUT_SIZE_EFFICIENTNET = 300     # EfficientNetB3

# Augmentation magnitudes (train only), per Phase 3.
_ROTATION_LIMIT_DEG = 10
_BRIGHTNESS_LIMIT = 0.1           # ±10%

SUBJECT_RE = re.compile(r"(OAS1_\d+)", re.IGNORECASE)


def parse_subject_id(filename: str) -> str:
    """Extract the ``OAS1_XXXX`` patient (subject) ID from an OASIS slice filename.

    OASIS filenames keep the source subject prefix (e.g.
    ``OAS1_0001_MR1_mpr-1_100.jpg``), so the ``OAS1_XXXX`` token identifies the patient.
    This ID is what the patient-level split groups on, so a slice can never leak the same
    brain across train/val/test.

    Args:
        filename: A slice filename or path (only the basename is inspected).

    Returns:
        The subject ID in upper case, e.g. ``"OAS1_0001"``.

    Raises:
        ValueError: If no ``OAS1_XXXX`` token is present.
    """
    match = SUBJECT_RE.search(Path(filename).name)
    if match is None:
        raise ValueError(f"Could not parse an OAS1_XXXX subject ID from: {filename!r}")
    return match.group(1).upper()


def get_transforms(input_size: int, train: bool):
    """Build the albumentations transform pipeline for one split.

    Val/test pipelines contain only the deterministic steps (resize + normalize) so that
    reported metrics reflect the real image. Training adds light, label-preserving online
    augmentation (rotation ±10°, horizontal flip, brightness ±10%).

    The grayscale→3-channel duplication is handled in :meth:`OasisDataset.__getitem__`
    (the image is already 3-channel by the time it reaches this pipeline), so normalization
    here uses the 3-channel ImageNet statistics.

    Args:
        input_size: Target square side length in pixels (e.g. 224 or 300).
        train: If ``True``, include augmentation; otherwise resize + normalize only.

    Returns:
        An ``albumentations.Compose`` that maps ``image=<HxWx3 uint8 array>`` to a
        normalized ``CxHxW`` float tensor.
    """
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    steps: List = [A.Resize(height=input_size, width=input_size)]

    if train:
        steps += [
            # Rotate on a black (0) background — natural for MRI, which sits on black.
            A.Rotate(
                limit=_ROTATION_LIMIT_DEG,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            # Brightness only (contrast_limit=0) to match the ±10% brightness jitter.
            A.RandomBrightnessContrast(
                brightness_limit=_BRIGHTNESS_LIMIT,
                contrast_limit=0.0,
                p=0.5,
            ),
        ]

    steps += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(steps)


class OasisDataset(_DatasetBase):
    """A ``torch`` ``Dataset`` yielding ``(image_tensor, label)`` for OASIS MRI slices.

    Given the paths and labels for one split, this applies the Phase 3 preprocessing
    pipeline and returns a normalized ``3xHxW`` float tensor plus the integer label.

    Args:
        root: Base directory prepended to each entry in ``file_list`` (entries that are
            already absolute or already exist are used as-is).
        file_list: Slice filenames or paths, one per sample.
        labels: Integer class labels aligned with ``file_list``.
        input_size: Target square side length (224 for ResNet50/DenseNet121, 300 for
            EfficientNetB3).
        train: If ``True``, apply online augmentation; otherwise deterministic transforms.
    """

    def __init__(
        self,
        root: Union[str, Path],
        file_list: Sequence[str],
        labels: Sequence[int],
        input_size: int,
        train: bool = True,
    ) -> None:
        if len(file_list) != len(labels):
            raise ValueError(
                f"file_list and labels length mismatch: {len(file_list)} vs {len(labels)}"
            )
        self.root = Path(root)
        self.file_list: List[str] = list(file_list)
        self.labels: List[int] = [int(x) for x in labels]
        self.input_size = int(input_size)
        self.train = bool(train)
        self.transform = get_transforms(self.input_size, self.train)

    def __len__(self) -> int:
        return len(self.file_list)

    def _resolve_path(self, entry: str) -> Path:
        """Resolve a ``file_list`` entry against ``root`` (absolute/existing kept as-is)."""
        p = Path(entry)
        if p.is_absolute() or p.exists():
            return p
        return self.root / entry

    def __getitem__(self, idx: int) -> Tuple["object", int]:
        from PIL import Image

        path = self._resolve_path(self.file_list[idx])
        # Force single-channel read, then duplicate to 3 channels so ImageNet-pretrained
        # backbones (which expect RGB) receive the grayscale MRI in every channel.
        gray = np.array(Image.open(path).convert("L"), dtype=np.uint8)  # (H, W)
        rgb = np.stack([gray, gray, gray], axis=-1)                     # (H, W, 3)
        image = self.transform(image=rgb)["image"]                     # (3, H, W) float
        return image, self.labels[idx]


__all__ = [
    "parse_subject_id",
    "get_transforms",
    "OasisDataset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_SIZE_DEFAULT",
    "INPUT_SIZE_EFFICIENTNET",
]
