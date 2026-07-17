# Alzheimer's Disease Prediction — Research Plan & Session Handoff

> **Purpose of this file:** A complete, self-contained record of the research
> decisions made so a fresh session (e.g. with Fable) can continue without
> re-deriving anything. Read this top-to-bottom first.

---

## 0. TL;DR (read this if nothing else)

- The dataset is **OASIS-1**, 2D axial slices. Real sample size = **347 patients**,
  NOT 86,000 images. Images from one patient are near-duplicates.
- **The #1 risk is DATA LEAKAGE, not overfitting from augmentation.** You MUST
  split by **subject ID**, never by image. Image-shuffled splits are why so many
  published OASIS/Kaggle Alzheimer papers report a fake 98–99% accuracy.
- **Task decision:** do **3-class** — `No` / `Very_mild` / `Mild+Moderate`.
  Merge Moderate into Mild (Moderate has only 2 subjects → cannot be split).
- **Drop ADASYN.** It's a tabular method; on raw pixels it produces meaningless
  blurry "average brains." For images use: **image augmentation (train only) +
  class weights / focal loss** to fix imbalance.
- **Model:** transfer learning with a **small** pretrained CNN (ResNet18 or
  EfficientNet-B0). Big models overfit 347 patients.
- **Evaluate at the subject level** (aggregate slice predictions per patient),
  with **macro-F1 / balanced accuracy / per-class recall / confusion matrix** —
  never plain accuracy. Use **StratifiedGroupKFold** (groups = subject IDs).
- **Anchor everything to the majority baseline: predicting "No" for everyone
  scores ~77% accuracy** (266/347 subjects). Beat *macro-F1*, not accuracy.
- **Lock a held-out test set of subjects** (or use nested CV). Do NOT report the
  same folds you tuned on.
- **Realistic target: macro-F1 ≈ 0.55–0.70.** If you see >0.90, suspect leakage —
  do not celebrate it.

---

## 1. Who I am / research goal

- Researcher doing **Alzheimer's disease stage prediction** from brain MRI.
- Self-described "not pro in ML, lacks basics" → prefer robust, standard,
  hard-to-get-wrong methods over clever/fragile ones.
- Wants guidance on: tools, model, hyperparameters, feasibility, and the
  train/test/validation split.

---

## 2. The dataset — what it REALLY is

Source: **OASIS-1** cross-sectional MRI. Each brain volume was sliced along the
z-axis (256 slices); slices **100–160** were kept. Each patient contributed
~240 images (≈4 MPRAGE acquisitions × ~60 slices).

Filename format:
```
OAS1_0351_MR1_mpr-4_160.jpg
      ^^^^ subject  ^^^ acquisition  ^^^ slice number
```
- `OAS1_0351` = **subject/patient ID** (this is the grouping key for splitting)
- `MR1` = session, `mpr-4` = acquisition, `160` = slice index

### Verified counts (from the actual files in `datasets/Data/`)

| Class | Images | Unique subjects |
|---|---|---|
| Non Demented | 67,222 | **266** |
| Very mild Dementia | 13,725 | **58** |
| Mild Dementia | 5,002 | **21** |
| Moderate Dementia | 488 | **2** |
| **TOTAL** | ~86,437 | **347** |

Folder paths:
```
datasets/Data/Non Demented/
datasets/Data/Very mild Dementia/
datasets/Data/Mild Dementia/
datasets/Data/Moderate Dementia/
```

### Key consequences
1. **Real N = 347 people.** The image count is an illusion of size — slices from
   one brain are highly correlated, not independent samples.
2. **This is a SMALL dataset with SEVERE imbalance**, masked by big image counts.
3. **No cross-class patient overlap.** OASIS-1 gives each subject a single CDR
   rating, so a patient belongs to exactly ONE class. (This resolves the original
   worry "same patient in multiple sets" — the real leakage risk is a patient's
   *slices* landing in both train and test.)

---

## 3. Original concerns — addressed

| Concern | Verdict |
|---|---|
| "Dataset is smaller on 2 sections" | True and worse than it looks: Mild=21 subjects, Moderate=2 subjects. Merge them. |
| "Not pro in ML" | Use transfer learning + standard recipe below. Avoid fragile tricks. |
| "Same patient data in multiple sets" | Not across classes (OASIS = 1 CDR/subject). The real danger is slice leakage across train/test → **split by subject**. |
| "Augmentation may overfit eventually" | Augmentation on 23 patients does NOT create new patients; it only reduces *slice* memorization. Real overfitting control = frozen pretrained backbone + class weights + subject-level CV. Augment, but don't rely on it to fix small-N. |

---

## 4. THE LEAKAGE TRAP (most important section)

If you split by **image** (random shuffle), one patient's slice #120 goes to
train and slice #121 to test. The model learns to recognize *that skull*, not the
disease → **fake 98–99% accuracy** that collapses on new patients. This is the
single most common fatal error in OASIS/Kaggle Alzheimer papers.

**Rules:**
- Split by **subject ID**, never by image.
- Use `sklearn.model_selection.StratifiedGroupKFold`
  (`groups` = subject IDs, stratify = class label). All slices of a patient stay
  in the same fold.
- **Evaluate at the subject level:** predict every slice, then aggregate to ONE
  prediction per patient (mean probability across the patient's slices, or
  majority vote). This is the honest, clinically meaningful unit.
- Apply augmentation / class balancing **only inside each fold's training portion**.

---

## 5. Task decision: 3-class

Moderate has only **2 subjects** → cannot be placed in both train and test → any
"Moderate accuracy" claim is dishonest. Merge Moderate into Mild.

Final classes:

| Class | Subjects | Notes |
|---|---|---|
| Non Demented | 266 | majority |
| Very mild | 58 | workable |
| Mild + Moderate | 21 + 2 = **23** | small but usable |

- **3-class (No / Very-mild / Mild+Moderate)** = defensible; report with a caveat
  that Moderate = 2 subjects.
- **Do NOT publish 4-class staging** (can't honestly claim a 2-person class).
- **Binary (demented vs not)** is the safe fallback if 3-class results are shaky.

---

## 6. ADASYN — decision: DROP IT

ADASYN/SMOTE are for **tabular feature vectors** (age, MMSE, volumes). On raw MRI
pixels (~50k-dim) they interpolate meaningless blurry "average brains" and suffer
the curse of dimensionality (the ADASYN notes' own §7 warning).

Replace with, for images:
1. **Image augmentation (training set only):** rotation ±10–15°, small
   shift/zoom, brightness/contrast, mild Gaussian noise. **Horizontal flip OK**
   (brain ~symmetric). **NO vertical flip, NO 90° rotation** (label-destroying —
   such brains never occur at test time).
2. **Class weights** in the loss (weighted `CrossEntropyLoss`, inverse frequency)
   or **focal loss**. Best imbalance fix, costs nothing.
3. Optional: **oversample minority subjects** in the batch sampler.

> ADASYN could only re-enter as an *advanced optional* path: extract a feature
> vector per image from a pretrained CNN (embeddings), then balance in that
> low-dim feature space + a classic classifier. Not the recommended main path.

---

## 7. Recommended model & hyperparameters (transfer learning)

- **Backbone:** start **small** — `ResNet18` or `EfficientNet-B0`,
  ImageNet-pretrained. (Big models overfit 347 patients.)
- **Input:** grayscale → replicate to 3 channels; resize **224×224**; ImageNet
  normalization (mean/std).
- **Preprocessing (do this BEFORE augmentation, applied to every image):**
  - Per-image intensity normalization (OASIS slices vary in brightness) — scale
    to [0,1] then ImageNet-normalize, or z-score per image.
  - Optional but recommended: **crop/center the brain** (remove black borders) so
    the model can't cheat on framing/skull-outline artifacts.
  - Skull-stripping is nice-to-have, not required for a first pass — the OASIS
    JPGs are already fairly clean. Note whichever you did in the methods section.
- **Two-stage training:**
  1. Freeze backbone, train only the new classifier head (few epochs).
  2. Unfreeze top block(s), fine-tune with a low LR.
- **Starting hyperparameters:**
  - Optimizer: **AdamW**
  - LR: **1e-4** (head) / **1e-5** (backbone during fine-tune)
  - Weight decay: **1e-4**
  - Batch size: **32**
  - Dropout in head: **0.3–0.5**
  - Epochs: up to **30–40** with **early stopping on validation macro-F1**
  - Loss: **weighted cross-entropy** (or focal loss)
- **Augmentation:** as in §6, on-the-fly during training.

### Consider: use fewer slices
Using all ~60 slices/patient is redundant. Consider keeping only central slices
(e.g. 120–140) or subsampling — less redundancy, faster training, similar signal.

---

## 8. Evaluation (never plain accuracy)

- Metrics: **macro-F1, balanced accuracy, per-class recall, confusion matrix.**
- Compute them at the **subject level** (aggregate slice predictions per patient).
- Report **mean ± std across the 5 CV folds** (small data → single split is noisy).
- Also useful: **PR-AUC** per class (better than ROC-AUC under imbalance).

### Evaluation protocol (avoid tuning-on-test bias)
- **Aggregation:** average the softmax probabilities over a patient's slices →
  one probability vector per subject → argmax. (Mean-prob usually beats majority
  vote; try both, pick on validation.)
- **Held-out design:** lock ~20% of *subjects* as a final test set (stratified by
  class), do 5-fold CV on the remaining 80% for model selection, then evaluate the
  chosen recipe ONCE on the locked test subjects. Report that number as headline.
- **Always show the majority-class baseline** row in every results table so the
  reader sees the real lift over "predict No."

### Reproducibility
- Fix a global **seed** (Python/NumPy/PyTorch, `cudnn.deterministic`), log the
  fold subject-IDs, and save the split table — so results and leakage checks are
  auditable. Version the split file alongside the paper.

---

## 9. Feasibility verdict

- ✅ **Feasible & publishable:** 3-class + subject-level StratifiedGroupKFold +
  transfer learning + class weights + augmentation + honest subject-level metrics.
  Genuinely valuable *because* so many prior works got the leakage wrong.
- ❌ **Not honest:** 4-class with a real Moderate claim; any image-shuffled split.
- ⚠️ **Biggest risk = leakage from splitting**, not augmentation overfitting.
  Get the split right and the rest is standard.

---

## 9b. Interpretability & limitations (for the paper)

- **Grad-CAM** on a few correct/incorrect subjects → shows the model attends to
  brain tissue (e.g. ventricle/hippocampal regions) rather than skull edges or
  scanner artifacts. Cheap credibility, and a sanity check against hidden leakage.
- **Limitations to state honestly:** single-site OASIS-1 (no external validation);
  Moderate = 2 subjects (merged); 2D slices ignore 3D volumetric context; class
  imbalance; no clinical/cognitive features fused in. These are strengths in a
  paper *if* stated — they show you understood the failure modes others hid.

## 10. Pipeline (correct order)

1. Parse subject IDs from filenames → build a table: `image_path, subject_id, class`.
2. Merge Moderate → Mild (3-class labels).
3. **StratifiedGroupKFold** (groups=subject_id) → 5 folds. No subject in two folds.
4. Within each fold's TRAIN only: apply augmentation + class weights.
5. Train small pretrained CNN (two-stage), early-stop on val macro-F1.
6. Predict slices on val/test → aggregate to subject-level prediction.
7. Report macro-F1 / balanced-acc / per-class recall / confusion matrix, mean±std.

---

## 11. Tools / libraries

- `pytorch` + `torchvision` (models, transforms) — or `timm` for EfficientNet.
- `albumentations` (optional, richer image augmentation).
- `scikit-learn` (`StratifiedGroupKFold`, metrics, confusion matrix).
- `pandas` / `numpy` for the filename→subject table.
- (NOT `imbalanced-learn`/ADASYN for the image path — see §6.)

---

## 12. Next steps I offered to build (pick up here next session)

1. `split_subjects.py` — build leak-free StratifiedGroupKFold splits from
   filenames; print per-fold class & subject counts as a sanity check.
2. A written **research plan / methods section** for the paper.
3. A **PyTorch training skeleton** (dataset class, weighted loss, augmentation,
   two-stage transfer learning, subject-level evaluation).

> Suggested order to resume: **(1) split_subjects.py first** — it makes the
> leakage-safe foundation concrete and verifies subject counts, then (3) the
> training skeleton, then (2) the write-up.

---

## 13. Open questions — CONFIRMED (checked against the real files, 2026-07-17)

- **13.1 Cross-class overlap — CONFIRMED NONE. ✅**
  No subject ID appears in more than one class. Total = **347 unique subjects**
  (266+58+21+2). OASIS-1 assigns one CDR per subject. The original "same patient
  in multiple sets" worry is ruled out at the class level. (The slice-leakage
  across train/test in §4 still applies and is the one that matters.)

- **13.2 Slice selection — structure CONFIRMED.**
  - Slices **100–160** present uniformly: exactly **1,417 images per slice index**
    × 61 indices = 86,437 total. ✅
  - Acquisitions: **mpr-1..mpr-4** are standard (~365 series each); **mpr-5/mpr-6**
    exist for only **one** subject. So each subject ≈ 4 near-identical repeat scans
    × 61 slices ≈ 240 images → redundancy is high.
  - **Decision (recommended):** subsample — e.g. central slices (~120–140) and/or
    one acquisition per subject. Grouping by subject ID already keeps ALL of a
    patient's acquisitions in the same fold (no leakage from repeat scans).

- **13.3 3-class vs binary — DECISION (not a fact).**
  Run both; lead the paper with whichever is honestly stronger. Default headline:
  3-class (`No` / `Very-mild` / `Mild+Moderate`).

- **13.4 GPU / compute — CONFIRMED: NONE locally. ⚠️**
  `nvidia-smi` not on PATH; `torch` not installed in this environment. CPU training
  on 86k images is impractical. **Use Kaggle Notebooks (free T4/P100) or Google
  Colab.** Combined with the subsampling in 13.2, training becomes tractable.
