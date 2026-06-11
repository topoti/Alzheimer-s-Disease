# Phase 3 · Step 3 — Stratified Split & the PyTorch Dataset Class

> **Phase 3 — Dataset & Preprocessing**
> The goal of this step: divide the 86,437 images into train/val/test *correctly* (preserving class ratios **and** keeping every patient in a single split) and wrap them in a `Dataset` that feeds the dataloaders.

The one question this step answers:

> **How do I split the data so each split is representative — and so neither synthetic samples nor *patients* leak between them?**

---

## 1. Why split, why *stratified*, and why *grouped by patient*

You hold out data the model never trains on, to honestly measure generalization (Step 1's overfitting idea). Three splits:
- **Train** — the model learns from these.
- **Validation** — you tune/early-stop and compute ensemble weights on these (never train on them).
- **Test** — touched **once**, at the very end, for the final reported numbers.

**Stratified** means each split keeps the *same class proportions* as the whole dataset. With a ~138× imbalance, a naive random split could put nearly all 488 Moderate images in train and almost none in test (or vice versa) — meaningless metrics. Stratifying on the label prevents that.

**Grouped by patient** is the second, dataset-specific requirement (Phase 1 Step 2). This OASIS export has **many 2D slices per patient**. If slices of the same patient appear in both train and test, the model can memorize patient-specific anatomy and your test score is fiction. So the split must be done over **patients (subject IDs), not over individual slices** — all of a patient's slices move together into one split. Conveniently, in OASIS each subject has a single diagnosis, so a patient maps cleanly to one class, and we can stratify *and* group at the same time.

---

## 2. The split (locked) — subject-level, stratified, 80/10/10

1. From EDA (Step 1) you have, for every image, its **class** and its **subject ID** (`OAS1_XXXX`).
2. Build the **list of unique subjects** and each subject's class.
3. Split the **subjects** 80/10/10, **stratified by class**, with `random_state=42`. Practical options:
   - `sklearn.model_selection.StratifiedGroupKFold` (groups = subject IDs, then take folds for val/test), or
   - manual: stratified split of the *subject list* by class, applied twice (carve off test subjects, then val subjects).
   - `GroupShuffleSplit` also guarantees disjoint groups but does **not** stratify — only use it if you then check class balance per split.
4. Expand each subject set back to its slices → the train/val/test image lists.
5. **Save the split to disk** (`data/splits/`, including the subject→split assignment) so every notebook uses the *exact same* split. Re-splitting per notebook is a silent leakage/inconsistency bug.

Resulting approximate train counts (≈80% of 86,437 ≈ 69,000 slices — exact numbers shift because you're splitting *whole patients*, so per-class slice counts won't land on a clean 80%):

| Class | ~Train slices |
|---|---|
| Non Demented | ~53,800 |
| Very mild Dementia | ~11,000 |
| Mild Dementia | ~4,000 |
| Moderate Dementia | ~390 |

> ⚠️ **The rare-subject caveat (state it honestly).** Subject-level splitting is correct, but the **Moderate** class may come from only a handful of distinct patients. If a class has too few subjects to place some in each of train/val/test, you cannot fully honor both grouping *and* three-way coverage for that class — document exactly how you handled it (e.g. how many Moderate subjects exist and how they were allocated). This is a genuine limitation to report in Phase 7, **not** something to hide by silently falling back to slice-level splitting. Even after grouping, Moderate stays tiny — which is exactly why ADASYN (next step) exists, and why test must stay imbalanced (it must reflect reality).

---

## 3. The PyTorch `Dataset` class

A `Dataset` is a thin object that answers two questions: "how many samples?" (`__len__`) and "give me sample i" (`__getitem__` → returns `(image_tensor, label)`). It applies the transforms from Step 2.

```
# conceptual — lives in src/data.py
class OasisDataset(Dataset):
    def __init__(self, file_paths, labels, transform): ...
    def __len__(self):  return len(self.file_paths)
    def __getitem__(self, i):
        img = load(self.file_paths[i]); img = self.transform(img)
        return img, self.labels[i]
```

Wrap each split in a `DataLoader` (batching, shuffling for train only). Remember: **train loader shuffles; val/test loaders do not.** And you'll instantiate the Dataset with the right `img_size` per model (224 vs 300).

---

## 4. The leakage discipline (recurring theme)

- Split **once**, by **patient**, save indices, reuse everywhere.
- **No patient appears in more than one split** (the slice-leakage rule).
- ADASYN comes **after** this split, **train only** (next step).
- Ensemble weights come from **val** (Phase 2 Step 2).
- Test is opened **once**.

These rules are the same principle wearing several hats: *information from evaluation data — including which patient it came from — must never touch training.*

### Resources (a few hours total)
- `sklearn.model_selection.train_test_split` docs — focus on `stratify`.
- PyTorch `Dataset` / `DataLoader` tutorial (the "custom dataset" section).

---

## 5. Your task (verifies Step 3 — do it before moving on)

- [ ] **1.** Implement the **subject-level** stratified 80/10/10 split with `random_state=42` (split patients, then expand to slices); **save the subject→split assignment + indices** to `data/splits/`.
- [ ] **2.** Print per-class counts in each split *and* confirm **no subject ID appears in two splits** (the leakage check); note how the rare Moderate subjects were allocated.
- [ ] **3.** Implement `OasisDataset` and build train/val/test `DataLoader`s (shuffle train only).
- [ ] **4.** Pull one batch from the train loader; confirm shapes `(B, 3, H, W)` and labels in `{0,1,2,3}`.

✅ When the split is saved, reproducible, **patient-disjoint**, and a batch loads cleanly, move on.

## When you're ready

> **Phase 3 · Step 4 — Applying ADASYN to the training set (and only the training set).**
