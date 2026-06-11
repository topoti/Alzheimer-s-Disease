# Phase 3 · Step 3 — Stratified Split & the PyTorch Dataset Class

> **Phase 3 — Dataset & Preprocessing**
> The goal of this step: divide the 6,400 images into train/val/test *correctly* (preserving class ratios) and wrap them in a `Dataset` that feeds the dataloaders.

The one question this step answers:

> **How do I split the data so each split is representative — and so nothing leaks between them?**

---

## 1. Why split, and why *stratified*

You hold out data the model never trains on, to honestly measure generalization (Step 1's overfitting idea). Three splits:
- **Train** — the model learns from these.
- **Validation** — you tune/early-stop and compute ensemble weights on these (never train on them).
- **Test** — touched **once**, at the very end, for the final reported numbers.

**Stratified** means each split keeps the *same class proportions* as the whole dataset. With a 50× imbalance, a naive random split could put nearly all 64 Moderate images in train and almost none in test (or vice versa) — meaningless metrics. Stratifying on the label prevents that.

---

## 2. The split (locked)

- **80 / 10 / 10**, stratified on class label.
- Use `sklearn.model_selection.train_test_split(..., stratify=y, random_state=42)` — apply it twice (first carve off test, then split the remainder into train/val), each time stratified.
- **`random_state=42`** everywhere for reproducibility.
- **Save the split indices to disk** (`data/splits/`) so every notebook uses the *exact same* split. Re-splitting per notebook is a silent leakage/inconsistency bug.

Resulting approximate train counts (≈80% of 6,400 = 5,120):

| Class | ~Train count |
|---|---|
| Non-Dementia | ~2,560 |
| Very Mild | ~1,792 |
| Mild | ~717 |
| Moderate | ~51 |

Note Moderate drops to ~**51** training images — this is exactly why ADASYN (next step) exists, and why test must stay imbalanced (it must reflect reality).

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

- Split **once**, save indices, reuse everywhere.
- ADASYN comes **after** this split, **train only** (next step).
- Ensemble weights come from **val** (Phase 2 Step 2).
- Test is opened **once**.

These four rules are the same principle wearing four hats: *information from evaluation data must never touch training.*

### Resources (a few hours total)
- `sklearn.model_selection.train_test_split` docs — focus on `stratify`.
- PyTorch `Dataset` / `DataLoader` tutorial (the "custom dataset" section).

---

## 5. Your task (verifies Step 3 — do it before moving on)

- [ ] **1.** Implement the stratified 80/10/10 split with `random_state=42`; **save indices** to `data/splits/`.
- [ ] **2.** Print the per-class counts in each split; confirm proportions match the full dataset (stratification worked).
- [ ] **3.** Implement `OasisDataset` and build train/val/test `DataLoader`s (shuffle train only).
- [ ] **4.** Pull one batch from the train loader; confirm shapes `(B, 3, H, W)` and labels in `{0,1,2,3}`.

✅ When the split is saved and reproducible and a batch loads cleanly, move on.

## When you're ready

> **Phase 3 · Step 4 — Applying ADASYN to the training set (and only the training set).**
