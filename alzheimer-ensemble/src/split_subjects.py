"""Build the leak-free, subject-level 3-class split for the OASIS Alzheimer task.

This implements the split described in
``learn-from-session/research.md`` (§4, §8 "Evaluation protocol", §10):

* **3 classes** — Moderate (2 subjects) is merged into Mild, giving
  ``no_dementia`` / ``very_mild`` / ``mild_moderate``.
* **Split by SUBJECT, never by image.** Every slice of a patient stays in one
  place, so a skull can never leak across train/test → no fake 98-99% accuracy.
* **Locked held-out test set** (~20% of subjects, stratified by class) that is
  touched exactly once, plus **StratifiedKFold** over the remaining ~80% of
  subjects for model selection (5 folds). Because folds are assigned at the
  subject level, "groups = subject IDs" is guaranteed by construction.

It writes two CSVs to ``data/splits/`` and prints a sanity table so the fold
balance and the zero-overlap guarantee can be eyeballed immediately.

Usage
-----
    python -m src.split_subjects                 # defaults below
    python src/split_subjects.py --data-root /path/to/Data --seed 42

Outputs (under ``--out-dir``, default ``data/splits/``)
-------------------------------------------------------
* ``subject_split_3class.csv`` — one row per subject:
  ``subject_id, orig_class, label, fold``
* ``slice_index_3class.csv``   — one row per image:
  ``filepath, subject_id, orig_class, label, fold``

``fold`` is the whole partition: ``-1`` = held-out test, ``0..k-1`` = CV folds
(the fold that acts as validation on run ``fold``). ``label`` is the 3-class
target (0/1/2); the human-readable class name is a 1:1 function of it, so it is
not stored. ``orig_class`` is kept because it is NOT recoverable from ``label``
(it distinguishes the pre-merge ``mild`` vs ``moderate`` subjects).

Only stdlib + pandas + scikit-learn are needed (no torch), so this runs on the
CPU-only local machine before any GPU time is spent on Kaggle/Colab.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# --- Constants -------------------------------------------------------------

SUBJECT_RE = re.compile(r"(OAS1_\d+)", re.IGNORECASE)

# Raw OASIS folder name -> canonical original-class slug.
FOLDER_TO_ORIG = {
    "Non Demented": "non_dementia",
    "Very mild Dementia": "very_mild_dementia",
    "Mild Dementia": "mild_dementia",
    "Moderate Dementia": "moderate_dementia",
}

# 3-class mapping: Moderate is folded into Mild (research.md §5).
ORIG_TO_3CLASS = {
    "non_dementia": ("no_dementia", 0),
    "very_mild_dementia": ("very_mild", 1),
    "mild_dementia": ("mild_moderate", 2),
    "moderate_dementia": ("mild_moderate", 2),
}
CLASS_NAMES = ["no_dementia", "very_mild", "mild_moderate"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# --- Core steps ------------------------------------------------------------

def parse_subject_id(name: str) -> str:
    """Pull the ``OAS1_XXXX`` patient ID out of a slice filename."""
    m = SUBJECT_RE.search(Path(name).name)
    if m is None:
        raise ValueError(f"No OAS1_XXXX subject ID in: {name!r}")
    return m.group(1).upper()


def scan_images(data_root: Path) -> pd.DataFrame:
    """Walk ``data_root`` and return one row per image slice."""
    rows = []
    for folder, orig in FOLDER_TO_ORIG.items():
        class_dir = data_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Pass the correct --data-root (should contain: "
                f"{', '.join(FOLDER_TO_ORIG)})."
            )
        class_name, label = ORIG_TO_3CLASS[orig]
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append(
                {
                    "filepath": str(p),
                    "subject_id": parse_subject_id(p.name),
                    "orig_class": orig,
                    "class_name": class_name,
                    "label": label,
                }
            )
    if not rows:
        raise RuntimeError(f"No images found under {data_root}")
    return pd.DataFrame(rows)


def build_subject_table(slices: pd.DataFrame) -> pd.DataFrame:
    """Collapse the image table to one row per subject.

    Also asserts OASIS's one-CDR-per-subject property: a subject must belong to
    exactly one class (research.md §13.1). Any violation is a data error and
    stops the run rather than silently corrupting the split.
    """
    grp = slices.groupby("subject_id")
    bad = grp["label"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise ValueError(
            "Cross-class subject overlap detected (should be impossible for "
            f"OASIS-1): {list(bad.index)}"
        )
    subjects = (
        grp.agg(
            orig_class=("orig_class", "first"),
            class_name=("class_name", "first"),
            label=("label", "first"),
            n_slices=("filepath", "size"),
        )
        .reset_index()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    return subjects


def assign_split(subjects: pd.DataFrame, test_frac: float, n_folds: int,
                 seed: int) -> pd.DataFrame:
    """Add ``fold``: ``-1`` = held-out test, ``0..k-1`` = the k CV folds.

    A single ``fold`` column carries the whole partition — ``fold == -1`` is the
    test set, and any ``fold >= 0`` subject is a dev subject that acts as
    validation on run ``fold`` and as training on the other k-1 runs. So no
    separate ``split`` (dev/test) column is needed — it would be redundant.
    """
    subjects = subjects.copy()
    labels = subjects["label"].to_numpy()

    # 1. Lock a stratified held-out test set of subjects (touched once, at the end).
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    dev_idx, test_idx = next(sss.split(subjects, labels))

    subjects["fold"] = -1  # default = test; overwritten below for dev subjects

    # 2. Stratified k folds over the DEV subjects only (subject-level → no leak).
    dev = subjects.iloc[dev_idx].reset_index()  # keep original index in 'index'
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_local) in enumerate(skf.split(dev, dev["label"])):
        orig_positions = dev.loc[val_local, "index"].to_numpy()
        subjects.loc[orig_positions, "fold"] = fold
    return subjects


def verify_no_leakage(subjects: pd.DataFrame, n_folds: int) -> None:
    """Assert every subject sits in exactly one place (test XOR one dev fold)."""
    test = set(subjects.loc[subjects.fold == -1, "subject_id"])
    fold_sets = [
        set(subjects.loc[subjects.fold == f, "subject_id"])
        for f in range(n_folds)
    ]
    # Every subject must land in exactly one bucket: fold in {-1, 0..k-1}.
    bad = subjects[~subjects.fold.between(-1, n_folds - 1)]
    if len(bad):
        raise AssertionError(f"{len(bad)} subjects have an out-of-range fold.")
    if not test:
        raise AssertionError("Test set is empty.")
    # Test disjoint from every fold, and folds pairwise disjoint.
    for f, s in enumerate(fold_sets):
        if test & s:
            raise AssertionError(f"Test subjects leaked into fold {f}: {test & s}")
        for g in range(f + 1, n_folds):
            overlap = s & fold_sets[g]
            if overlap:
                raise AssertionError(f"Fold {f} and {g} share subjects: {overlap}")
    print("  [OK] zero subject overlap across test set and all folds.")


# --- Reporting -------------------------------------------------------------

def print_sanity_table(subjects: pd.DataFrame, slices: pd.DataFrame,
                        n_folds: int) -> None:
    """Print per-fold / test subject + image counts per class."""

    def counts(mask, unit_df, key):
        sub = subjects[mask]
        sc = sub.groupby("class_name")["subject_id"].nunique()
        subj_ids = set(sub["subject_id"])
        img = unit_df[unit_df.subject_id.isin(subj_ids)]
        ic = img.groupby("class_name")["filepath"].size()
        return sc, ic

    print("\n== SUBJECT counts (images in parentheses) ==")
    header = f"{'partition':<10}" + "".join(f"{c:>18}" for c in CLASS_NAMES) + f"{'total':>16}"
    print(header)
    print("-" * len(header))

    def row(name, mask):
        sc, ic = counts(mask, slices, None)
        cells = ""
        tot_s = tot_i = 0
        for c in CLASS_NAMES:
            s = int(sc.get(c, 0))
            i = int(ic.get(c, 0))
            tot_s += s
            tot_i += i
            cells += f"{s:>8} ({i:>6})".rjust(18)
        print(f"{name:<10}{cells}{f'{tot_s} ({tot_i})':>16}")

    row("test", subjects.fold == -1)
    for f in range(n_folds):
        row(f"fold {f}", subjects.fold == f)
    row("DEV all", subjects.fold >= 0)
    row("TOTAL", subjects.subject_id.notna())

    # Majority baseline anchor (research.md §0): predicting no_dementia for all.
    maj = subjects["class_name"].value_counts(normalize=True).max()
    print(f"\n  Majority-class baseline (subject level): "
          f"{maj*100:.1f}% accuracy -- beat MACRO-F1, not this.")


# --- Entry point -----------------------------------------------------------

def main(argv=None) -> int:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../Alzheimer-s-Disease
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path,
                        default=repo_root / "datasets" / "Data",
                        help="Folder containing the 4 OASIS class subfolders.")
    parser.add_argument("--out-dir", type=Path,
                        default=here.parents[1] / "data" / "splits",
                        help="Where to write the split CSVs.")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Scanning images under: {args.data_root}")
    slices = scan_images(args.data_root)
    subjects = build_subject_table(slices)
    print(f"  {len(slices):,} images  |  {len(subjects)} subjects  |  "
          f"{subjects['class_name'].nunique()} classes")

    subjects = assign_split(subjects, args.test_frac, args.folds, args.seed)
    verify_no_leakage(subjects, args.folds)

    # Join the fold assignment back onto the image table.
    slices = slices.merge(
        subjects[["subject_id", "fold"]], on="subject_id", how="left"
    )

    # Written columns: label is the 3-class target; class_name is dropped because it
    # is a 1:1 function of label, and there is no split column (fold carries it).
    # orig_class is kept — it is NOT derivable from label (traces the pre-merge class).
    drop_cols = ["class_name", "n_slices"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subj_out = args.out_dir / "subject_split_3class.csv"
    slice_out = args.out_dir / "slice_index_3class.csv"
    subjects.drop(columns=drop_cols).to_csv(subj_out, index=False)
    slices.drop(columns=["class_name"]).to_csv(slice_out, index=False)

    print_sanity_table(subjects, slices, args.folds)
    print(f"\nWrote:\n  {subj_out}\n  {slice_out}")
    print(f"Seed={args.seed}  test_frac={args.test_frac}  folds={args.folds}  "
          f"(reproducible; commit these CSVs alongside the paper).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
