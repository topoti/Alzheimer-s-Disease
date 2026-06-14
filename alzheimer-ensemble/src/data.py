"""Dataset, patient-level split, transforms, and dataloaders for the OASIS MRI 4-class task.

The single most important design choice here is the **subject-disjoint (patient-grouped)
split**. The OASIS export has many 2D slices per patient (filenames keep the
``OAS1_XXXX`` subject prefix). A naive slice-level split leaks the same patient's
brain into train *and* test, which inflates every metric — the reason a lot of OASIS
papers report dishonest ~99% numbers. We split *subjects*, then expand to slices, and
assert that no subject ID appears in two splits.

Measured class/subject counts in this repo's ``datasets/Data/``:

    Class                 Images   Unique subjects
    Non Demented          67,222   266
    Very mild Dementia    13,725    58
    Mild Dementia          5,002    21
    Moderate Dementia        488     2   <-- only TWO patients

With only 2 Moderate subjects a disjoint train/val/test split is impossible for that
class. We therefore allocate Moderate as **1 subject -> train, 1 subject -> test, none
in val**, and its test metric must be read as a single-held-out-patient probe, not a
population estimate. See RESEARCH_PLAN.md (Phase 3, split) for the rationale.

Run directly to build and persist the split::

    python -m src.data                      # uses configs/default.yaml + local datasets/
    python -m src.data --data-root /path    # override the image root
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# NOTE: torch / torchvision / PIL are imported lazily inside the transform/Dataset/loader
# helpers below, so the split-building path (build_index -> split_subjects -> save_splits,
# i.e. `python -m src.data`) runs in a lightweight env without the deep-learning stack.

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]          # .../alzheimer-ensemble
PROJECT_ROOT = REPO_ROOT.parent                          # .../Alzheimer-s-Disease
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
# Local image root (used when the config's Kaggle path is absent).
LOCAL_DATA_ROOT = PROJECT_ROOT / "datasets" / "Data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUBJECT_RE = re.compile(r"(OAS1_\d+)", re.IGNORECASE)

# Map a normalized class-folder name -> (canonical name, integer label).
# Folder names differ slightly between exports ("Non Demented", "NonDemented", ...),
# so we normalize to lowercase-letters-only before matching.
CLASS_BY_NORMALIZED: Dict[str, Tuple[str, int]] = {
    "nondemented": ("non_dementia", 0),
    "nondementia": ("non_dementia", 0),
    "verymilddementia": ("very_mild_dementia", 1),
    "verymilddemented": ("very_mild_dementia", 1),
    "milddementia": ("mild_dementia", 2),
    "milddemented": ("mild_dementia", 2),
    "moderatedementia": ("moderate_dementia", 3),
    "moderatedemented": ("moderate_dementia", 3),
}
CLASS_NAMES = ["non_dementia", "very_mild_dementia", "mild_dementia", "moderate_dementia"]


def _normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML config as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_data_root(config: dict, override: Optional[str] = None) -> Path:
    """Pick the image root: CLI override > config path (if it exists) > local datasets/Data."""
    if override:
        root = Path(override)
        if not root.exists():
            raise FileNotFoundError(f"--data-root does not exist: {root}")
        return root
    cfg_root = Path(config.get("data", {}).get("root", ""))
    if cfg_root.exists():
        return cfg_root
    if LOCAL_DATA_ROOT.exists():
        return LOCAL_DATA_ROOT
    raise FileNotFoundError(
        f"Could not locate image root. Tried config root '{cfg_root}' and "
        f"local '{LOCAL_DATA_ROOT}'. Pass --data-root explicitly."
    )


# --------------------------------------------------------------------------------------
# Index building: scan files -> (filepath, label, subject) records
# --------------------------------------------------------------------------------------


def parse_subject_id(filename: str) -> Optional[str]:
    """Extract the ``OAS1_XXXX`` subject (patient) ID from a slice filename."""
    match = SUBJECT_RE.search(filename)
    return match.group(1).upper() if match else None


def build_index(data_root: Path) -> pd.DataFrame:
    """Scan the dataset directory into a slice-level DataFrame.

    Returns columns: ``filepath, class_name, label, subject_id``.
    Raises if any class folder is unrecognized or a file has no parseable subject.
    """
    data_root = Path(data_root)
    records: List[dict] = []
    unmatched_dirs: List[str] = []

    class_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class sub-directories found under {data_root}")

    for class_dir in class_dirs:
        mapping = CLASS_BY_NORMALIZED.get(_normalize_class_name(class_dir.name))
        if mapping is None:
            unmatched_dirs.append(class_dir.name)
            continue
        class_name, label = mapping
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            subject = parse_subject_id(img_path.name)
            if subject is None:
                raise ValueError(
                    f"Could not parse OAS1 subject ID from filename: {img_path.name}"
                )
            records.append(
                {
                    "filepath": str(img_path),
                    "class_name": class_name,
                    "label": label,
                    "subject_id": subject,
                }
            )

    if unmatched_dirs:
        raise ValueError(
            f"Unrecognized class folder(s): {unmatched_dirs}. "
            f"Add them to CLASS_BY_NORMALIZED in src/data.py."
        )
    if not records:
        raise ValueError(f"No images found under {data_root}")

    df = pd.DataFrame.from_records(records)

    # Sanity: each subject must belong to exactly one class.
    multi = df.groupby("subject_id")["label"].nunique()
    bad = multi[multi > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} subject(s) span multiple classes, e.g. {list(bad.index[:5])}")

    return df


# --------------------------------------------------------------------------------------
# Patient-level split
# --------------------------------------------------------------------------------------


def split_subjects(
    index: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each subject to 'train' / 'val' / 'test', stratified by class.

    Splitting is done *per class* (which guarantees stratification) on the unique
    subject list, so no subject can land in two splits. Rare classes get explicit
    fallbacks because the ratios cannot be honored:

      * 1 subject  -> train only (warns; class is untestable)
      * 2 subjects -> 1 train, 1 test, 0 val (the Moderate case)
      * >=3        -> rounded 80/10/10 with >=1 subject in each split
    """
    train_r, val_r, _test_r = ratios
    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}

    subjects_by_class: Dict[int, List[str]] = defaultdict(list)
    for subject, label in index.groupby("subject_id")["label"].first().items():
        subjects_by_class[int(label)].append(subject)

    for label in sorted(subjects_by_class):
        subjects = sorted(subjects_by_class[label])          # deterministic order
        perm = rng.permutation(len(subjects))
        subjects = [subjects[i] for i in perm]                # seeded shuffle
        n = len(subjects)

        if n == 1:
            splits = ["train"]
            print(
                f"  [warn] class '{CLASS_NAMES[label]}' has 1 subject -> train only; "
                f"it cannot be evaluated."
            )
        elif n == 2:
            splits = ["train", "test"]
            print(
                f"  [warn] class '{CLASS_NAMES[label]}' has 2 subjects -> "
                f"1 train / 1 test / 0 val (single-held-out-patient probe)."
            )
        else:
            n_train = max(1, round(train_r * n))
            n_val = max(1, round(val_r * n))
            n_test = n - n_train - n_val
            if n_test < 1:                                    # borrow from train
                n_test = 1
                n_train = n - n_val - n_test
            splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

        for subject, split in zip(subjects, splits):
            assignment[subject] = split

    return assignment


def apply_split(index: pd.DataFrame, assignment: Dict[str, str]) -> pd.DataFrame:
    """Attach a 'split' column to the slice-level index from the subject assignment."""
    out = index.copy()
    out["split"] = out["subject_id"].map(assignment)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "subject_id"].unique()[:5]
        raise ValueError(f"Subjects without a split assignment, e.g. {list(missing)}")
    return out


def verify_no_leakage(index: pd.DataFrame) -> None:
    """Assert subjects are disjoint across splits; raise loudly if not."""
    by_split = {s: set(g["subject_id"]) for s, g in index.groupby("split")}
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise AssertionError(
                    f"Subject leakage between {splits[i]} and {splits[j]}: "
                    f"{sorted(overlap)[:5]}"
                )


def summarize_split(index: pd.DataFrame) -> pd.DataFrame:
    """Per-split, per-class counts of slices and unique subjects (for printing/logging)."""
    rows = []
    for (split, label), g in index.groupby(["split", "label"]):
        rows.append(
            {
                "split": split,
                "class": CLASS_NAMES[int(label)],
                "slices": len(g),
                "subjects": g["subject_id"].nunique(),
            }
        )
    summary = pd.DataFrame(rows)
    order = {"train": 0, "val": 1, "test": 2}
    return summary.sort_values(
        by=["split", "class"], key=lambda s: s.map(order).fillna(s) if s.name == "split" else s
    ).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def save_splits(index: pd.DataFrame, splits_dir: Path = DEFAULT_SPLITS_DIR) -> None:
    """Persist the slice-level index and the subject->split table to CSV."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(splits_dir / "slice_index.csv", index=False)
    subject_table = (
        index.groupby("subject_id")
        .agg(class_name=("class_name", "first"), label=("label", "first"), split=("split", "first"))
        .reset_index()
        .sort_values(["split", "label", "subject_id"])
    )
    subject_table.to_csv(splits_dir / "subject_split.csv", index=False)


def load_splits(splits_dir: Path = DEFAULT_SPLITS_DIR) -> pd.DataFrame:
    """Load a previously saved slice-level index (so training reuses one fixed split)."""
    path = Path(splits_dir) / "slice_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"No saved split at {path}. Run `python -m src.data` first.")
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# Transforms + Dataset + DataLoaders
# --------------------------------------------------------------------------------------


def build_transforms(
    input_size: int,
    train: bool,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
    aug: Optional[dict] = None,
):
    """Build a torchvision transform pipeline.

    MRIs are grayscale; pretrained ImageNet backbones expect 3 channels, so we replicate
    the gray channel. Train transforms add light, label-preserving augmentation; val/test
    use only resize + normalize so metrics reflect the real image.
    """
    from torchvision import transforms

    aug = aug or {}
    steps: List = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
    ]
    if train:
        rot = aug.get("rotation_deg", 0)
        if rot:
            steps.append(transforms.RandomRotation(rot))
        if aug.get("horizontal_flip", False):
            steps.append(transforms.RandomHorizontalFlip())
        bright = aug.get("brightness_jitter", 0)
        if bright:
            steps.append(transforms.ColorJitter(brightness=bright))
        if aug.get("noise_removal") == "gaussian":
            sigma = aug.get("gaussian_sigma", 0.5)
            steps.append(transforms.GaussianBlur(kernel_size=3, sigma=(sigma, sigma)))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
    ]
    return transforms.Compose(steps)


def _make_dataset(frame: pd.DataFrame, transform):
    """Construct an OASISDataset (defined lazily so torch isn't needed for splitting)."""
    from torch.utils.data import Dataset
    from PIL import Image

    class OASISDataset(Dataset):
        """Returns ``(image_tensor, label)`` for one MRI slice."""

        def __init__(self, frame: pd.DataFrame, transform):
            self.filepaths = frame["filepath"].tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.transform = transform

        def __len__(self) -> int:
            return len(self.filepaths)

        def __getitem__(self, idx: int):
            image = Image.open(self.filepaths[idx]).convert("L")  # force single-channel read
            return self.transform(image), self.labels[idx]

    return OASISDataset(frame, transform)


def make_weighted_sampler(labels: Sequence[int]):
    """Inverse-frequency sampler so minority classes appear often in minibatches.

    Note: this is *one* imbalance corrector. Don't combine it at full strength with a
    heavy class-weighted loss AND full oversampling — see RESEARCH_PLAN.md (Phase 4).
    """
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels)
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    inv = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = inv[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(), num_samples=len(labels), replacement=True
    )


def build_dataloaders(
    index: pd.DataFrame,
    input_size: int,
    config: dict,
    use_weighted_sampler: bool = False,
    batch_size: Optional[int] = None,
) -> Dict[str, "object"]:
    """Build train/val/test dataloaders for a given backbone input size.

    ``input_size`` is per-model (224 for ResNet/DenseNet, 300 for EfficientNetB3), so call
    this once per backbone. Val/test never use augmentation or the weighted sampler.
    """
    from torch.utils.data import DataLoader

    norm = config["data"]["normalize"]
    aug = config.get("augmentation", {})
    train_cfg = config.get("training", {})
    batch_size = batch_size or train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 2)

    loaders: Dict[str, object] = {}
    for split in ("train", "val", "test"):
        frame = index[index["split"] == split]
        if frame.empty:
            continue
        is_train = split == "train"
        tfm = build_transforms(input_size, is_train, norm["mean"], norm["std"], aug)
        dataset = _make_dataset(frame, tfm)

        sampler = None
        shuffle = False
        if is_train:
            if use_weighted_sampler:
                sampler = make_weighted_sampler(frame["label"].tolist())
            else:
                shuffle = True

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )
    return loaders


# --------------------------------------------------------------------------------------
# Script entry point: build + verify + save the split
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the patient-level OASIS split.")
    parser.add_argument("--data-root", default=None, help="Override the image root directory.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config.")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR), help="Where to save CSVs.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    seed = config.get("seed", 42)
    split_cfg = config["data"]["split"]
    ratios = (split_cfg["train"], split_cfg["val"], split_cfg["test"])

    data_root = resolve_data_root(config, args.data_root)
    print(f"Scanning images under: {data_root}")
    index = build_index(data_root)
    print(f"Found {len(index):,} slices across {index['subject_id'].nunique()} subjects.")

    assignment = split_subjects(index, ratios=ratios, seed=seed)
    index = apply_split(index, assignment)
    verify_no_leakage(index)
    print("Subject-disjoint split verified (no patient appears in two splits).\n")

    summary = summarize_split(index)
    print(summary.to_string(index=False))

    save_splits(index, Path(args.splits_dir))
    print(f"\nSaved split to {Path(args.splits_dir)}")


if __name__ == "__main__":
    main()
