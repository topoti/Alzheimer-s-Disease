# Phase 3 · Step 1 — The OASIS Dataset & Exploratory Data Analysis

> **Phase 3 — Dataset & Preprocessing** (code starts here)
> The goal of this step: get the data in hand and *look at it* — confirm the class counts, image format, and the imbalance you've been theorizing about since Step 1.

The one question this step answers:

> **What exactly is in this dataset, and does it match what the plan promised?**

---

## 1. The dataset

- **OASIS MRI**, from Kaggle (search *"Alzheimer MRI Disease Classification Dataset"* — the same 6,400-image split as Mahmud et al.).
- **4 classes**, **2D axial slice JPGs**, already pre-extracted from 3D MRI volumes (so you're working with pictures, not raw volumes).

| Class | Meaning | # images |
|---|---|---|
| Non-Dementia | healthy brain | 3,200 |
| Very Mild Dementia | earliest decline | 2,240 |
| Mild Dementia | clear symptoms | 896 |
| Moderate Dementia | significant impairment | 64 |
| **Total** | | **6,400** |

That 3,200 : 64 (**50×**) ratio is the villain from Step 1, now made concrete.

---

## 2. What EDA must confirm (don't trust the table blindly — verify it)

EDA = "look before you leap." Build `notebooks/01_dataset_eda.ipynb` and check:

1. **Counts per class** — does the folder structure actually give 3200/2240/896/64? (Datasets get re-uploaded with different splits; confirm *your* copy.)
2. **Image properties** — dimensions, channels (grayscale vs already-3-channel?), file format, pixel value range (0–255?).
3. **A visual grid** — plot a few images from each class. Do they look like brain slices? Any corrupt/blank images? Any obvious non-brain artifacts?
4. **Duplicates / leakage risk** — are slices from the *same patient* spread across classes or near-duplicated? (Worth noting; full patient-level splitting may be out of scope but flag it.)

The output of this step is a **bar chart of class counts** (goes straight into the paper) and a sentence confirming format/size.

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

- [ ] **1.** Load the dataset and print the **actual** count per class; compare to 3200/2240/896/64.
- [ ] **2.** Print one sample image's dimensions, mode (grayscale?), and pixel min/max.
- [ ] **3.** Save a labeled image grid (a few per class) and a class-count bar chart to `results/figures/`.
- [ ] **4.** Write one sentence noting any surprise (wrong counts, unexpected channels, corrupt files).

✅ When the chart matches the table (or you've documented the difference), move on.

## When you're ready

> **Phase 3 · Step 2 — The preprocessing pipeline (resize, grayscale→3ch, normalize, …).**
