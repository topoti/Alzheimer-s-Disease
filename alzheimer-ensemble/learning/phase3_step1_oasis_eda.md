# Phase 3 · Step 1 — The OASIS Dataset & Exploratory Data Analysis

> **Phase 3 — Dataset & Preprocessing** (code starts here)
> The goal of this step: get the data in hand and *look at it* — confirm the class counts, image format, and the imbalance you've been theorizing about since Step 1.

The one question this step answers:

> **What exactly is in this dataset, and does it match what the plan promised?**

---

## 1. The dataset

- **`ninadaithal/imagesoasis`** ("OASIS Alzheimer's Detection") on Kaggle — derived from **OASIS-1**. (A *larger* OASIS export than Mahmud et al.'s set; still OASIS, still 4 classes.)
- **4 classes**, **2D axial slice images**, already pre-extracted from 3D MRI volumes (so you're working with pictures, not raw volumes). One subfolder per class.
- Files keep the **OASIS naming convention** (e.g. `OAS1_0308_MR1_mpr-1_101.jpg`) — the `OAS1_0308` prefix is the **patient/subject ID**. You'll need it for the subject-level split in Step 3, so parse it now.

| Class (folder name) | Meaning | # images |
|---|---|---|
| Non Demented | healthy brain | 67,222 |
| Very mild Dementia | earliest decline | 13,725 |
| Mild Dementia | clear symptoms | 5,002 |
| Moderate Dementia | significant impairment | 488 |
| **Total** | | **86,437** |

That 67,222 : 488 (**~138×**) ratio is the villain from Step 1, now made concrete — even harsher than the reference paper's set.

---

## 2. What EDA must confirm (don't trust the table blindly — verify it)

EDA = "look before you leap." Build `notebooks/01_dataset_eda.ipynb` and check:

1. **Counts per class** — does the folder structure actually give 67222/13725/5002/488? (Datasets get re-uploaded with different splits; confirm *your* copy.)
2. **Image properties** — dimensions, channels (grayscale vs already-3-channel?), file format, pixel value range (0–255?).
3. **A visual grid** — plot a few images from each class. Do they look like brain slices? Any corrupt/blank images? Any obvious non-brain artifacts?
4. **Subjects per class (do NOT skip — the split depends on it)** — parse the `OAS1_XXXX` subject ID from every filename and count **how many unique patients** are in each class, and **how many slices per patient**. This dataset has many slices per patient, so a naive slice-level split would leak patients across train/test (Phase 1 Step 2). You'll split at the *patient* level in Step 3 — but first you must know whether the rare classes have enough distinct patients to split at all (the Moderate class, in particular, may come from very few subjects). Record the subject counts; they become a reported number and a stated limitation.

The output of this step is a **bar chart of class counts** (goes straight into the paper), a **subjects-per-class count**, and a sentence confirming format/size.

---

## 3. Why this matters for everything downstream

- The **counts** drive your ADASYN target (Phase 3 Step 4) and your choice to headline macro-F1 (Phase 6).
- The **channels** decide whether you need the grayscale→3-channel trick (next step).
- The **pixel range and size** decide your normalization and resize (next step).

> Spending an hour here saves days later — a wrong assumption about channels or counts silently corrupts the whole pipeline.

### Resources (a few hours total)
- The Kaggle dataset page (read its description + any "data card").
- `pandas` `value_counts()` and `matplotlib`/`seaborn` `countplot` for the distribution chart.
- `PIL.Image` / `cv2` to inspect a sample image's `.size` and `.mode`.

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Load the dataset and print the **actual** count per class; compare to 67222/13725/5002/488.
- [ ] **2.** Print one sample image's dimensions, mode (grayscale?), and pixel min/max.
- [ ] **3.** Save a labeled image grid (a few per class) and a class-count bar chart to `results/figures/`.
- [ ] **4.** Parse the `OAS1_XXXX` subject ID from filenames; print **unique patients per class** and median slices-per-patient. Flag any class with very few subjects.
- [ ] **5.** Write one sentence noting any surprise (wrong counts, unexpected channels, corrupt files, too-few-subjects classes).

✅ When the chart matches the table (or you've documented the difference), move on.

## When you're ready

> **Phase 3 · Step 2 — The preprocessing pipeline (resize, grayscale→3ch, normalize, …).**
