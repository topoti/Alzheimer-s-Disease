# Phase 3 · Step 5 — Verify the Balance & Eyeball Synthetic Samples

> **Phase 3 — Dataset & Preprocessing** (closes Phase 3)
> The goal of this step: *prove* ADASYN did what you think — train is balanced, test is still imbalanced, and the synthetic images aren't garbage.

The one question this step answers:

> **Did ADASYN actually balance the training set without corrupting the data or leaking into test?**

---

## 1. Two numbers that must be true

Build `notebooks/02_adasyn_sanity_check.ipynb` and print, side by side:

1. **Training distribution before vs after ADASYN** — before: ~2560/1792/717/51; after: all four ≈ ~2,560. *Roughly equal* is the pass condition.
2. **Test distribution** — must be **unchanged and still imbalanced** (~400/280/112/8 at 10%). If test moved, ADASYN touched data it shouldn't have — stop and fix.

This is Phase 3's official verification (from the plan): *post-ADASYN training is roughly balanced across 4 classes; test retains original imbalance.*

---

## 2. Eyeball the synthetic samples

ADASYN in pixel space (Option A) can produce blurry or odd-looking images. Save a handful of **synthetic Moderate** samples as PNGs next to **real** Moderate samples and look:
- Do they still read as brain slices?
- Blurry-but-plausible = fine (expected for pixel-space interpolation).
- Pure noise / nonsensical blobs = a red flag → consider Option B (embedding-space) from Step 4.

This visual check is cheap insurance and produces a figure you can show in the paper or rebuttal ("synthetic samples remain anatomically plausible").

---

## 3. What "balanced" buys you (connect back)

A balanced training set means the model no longer gets rewarded for the lazy "always predict Non-Dementia" shortcut (Step 1). Combined with the F1-based evaluation (Phase 6), this is the mechanism by which **Moderate-class recall** improves over the reference paper — your headline ablation result.

> Remember: balancing happens on **train**. The fact that **test stays imbalanced** is *why* your final metrics are honest and comparable to Mahmud et al.

### Resources (a few hours total)
- `pandas value_counts` / `seaborn countplot` for before/after bar charts.
- `matplotlib` `imshow` (or PIL `save`) to dump real-vs-synthetic image grids.

---

## 4. Your task (verifies Step 5 — and closes Phase 3)

- [ ] **1.** Print and chart train distribution **before vs after** ADASYN; confirm ≈ balanced.
- [ ] **2.** Print test distribution; confirm it is **unchanged and still imbalanced**.
- [ ] **3.** Save a real-vs-synthetic image grid for the Moderate class; write one sentence on whether they look plausible.
- [ ] **4.** If samples look like noise, note the decision to try Option B; otherwise confirm Option A stands.

✅ Phase 3 verification (plan): train balanced, test imbalanced. When both charts confirm it, you're ready to train.

## When you're ready

> **Phase 4 · Step 1 — Transfer learning: the two-stage freeze-then-fine-tune strategy.**
