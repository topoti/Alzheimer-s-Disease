# Phase 6 · Step 2 — Confusion Matrices & ROC Curves

> **Phase 6 — Evaluation**
> The goal of this step: produce the two diagnostic figures that show *how* the model succeeds and fails — not just a single score.

The one question this step answers:

> **Which classes get confused with which, and how well does the model separate each class across thresholds?**

---

## 1. The confusion matrix (4×4)

Rows = true class, columns = predicted class. The diagonal is correct; off-diagonals are mistakes. Read it for *patterns*, because not all mistakes are equal (the ordinal idea from Phase 1 Step 1):

- **Adjacent confusions** (Mild ↔ Very Mild, Moderate ↔ Mild) — **expected**; these stages are clinically similar.
- **Distant confusions** (Moderate predicted as Non-Dementia) — **dangerous**; a sick patient sent home. Call these out explicitly.
- **The Moderate row is the weakest** — only 64 originals; even with ADASYN it's the hardest. Pay special attention to its recall (its diagonal cell ÷ row sum).

Plot it with `seaborn.heatmap` (annotated counts), save to `results/figures/`. Make one for the **ensemble** and keep the single-model ones for the appendix/ablation.

---

## 2. ROC curves (one-vs-rest, per class)

For a 4-class problem you draw **one ROC curve per class** (that class vs the rest), each with its AUC. ROC is **threshold-independent** — it shows how well the model *ranks* a class's members above non-members across all decision thresholds. Clinically useful because the operating threshold differs between screening (favor recall) and confirmatory diagnosis (favor precision).

Plot all four curves on one axes with their AUCs in the legend; save to `results/figures/`.

---

## 3. Why both figures, not just metrics

A single macro-F1 hides *structure*. The confusion matrix tells a clinician exactly which errors to distrust; the ROC curves tell them how much threshold-tuning headroom exists. Together they turn "0.94 macro-F1" into an actionable clinical picture — and they're standard expected figures for this kind of paper.

### Resources (a few hours total)
- `sklearn.metrics.confusion_matrix` + `seaborn.heatmap` (with `annot=True`).
- `sklearn.metrics.roc_curve` / `roc_auc_score(multi_class='ovr')`; `label_binarize` for one-vs-rest.

---

## 4. Your task (verifies Step 2 — do it before moving on)

- [ ] **1.** Generate the ensemble's 4×4 confusion matrix figure (annotated heatmap).
- [ ] **2.** Identify any **distant** confusions (especially Moderate↔Non-Dementia) and note them.
- [ ] **3.** Plot one-vs-rest ROC curves for all four classes with AUCs.
- [ ] **4.** Write two sentences interpreting the Moderate row's recall and the most worrying confusion.

✅ When both figures exist and you can read the error structure off them, move on.

## When you're ready

> **Phase 6 · Step 3 — The two ablation studies (without ADASYN; same-family ensemble).**
