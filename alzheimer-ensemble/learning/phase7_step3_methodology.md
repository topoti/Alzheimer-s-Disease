# Phase 7 · Step 3 — Draft the Methodology

> **Phase 7 — Paper Writing**
> The goal of this step: write the section that lets another researcher *reproduce* your work — the factual core of the paper.

The one question this step answers:

> **How do I describe exactly what I did, precisely enough to be reproduced?**

---

## 1. The seven subsections (follow your own pipeline order)

4.1 **Dataset** — OASIS (`ninadaithal/imagesoasis`, ~86,437 slices from OASIS-1) description, 4 classes, counts, **unique-subjects-per-class**, and the **subject-level (patient-grouped) stratified** 80/10/10 split, `random_state=42`. State that it differs from the reference paper's export. (Phase 3 Steps 1, 3.)
4.2 **Preprocessing pipeline** — the 6 steps with a **figure** showing each stage; note ImageNet normalization and the two input sizes (224/300). (Phase 3 Step 2.)
4.3 **ADASYN** — algorithm summary (the `rᵢ` weighting), your **pixel-space (Option A)** application, the **train-only-after-split** rule, before/after distribution table. (Phase 1 Step 2, Phase 3 Step 4.)
4.4 **Three backbones** — one paragraph each (paradigm + why), plus an **architecture figure**; the params/input-size table. (Phase 2 Step 1.)
4.5 **Ensemble fusion** — the **weighted soft-voting formula** with val-F1 weights. (Phase 2 Step 2.)
4.6 **XAI methods** — Grad-CAM, SHAP, LIME with their formulas + how you applied each to the *ensemble* (averaged heatmaps vs predict-function). (Phase 1 Step 4, Phase 5.)
4.7 **Training protocol** — the **hyperparameter table** and two-stage transfer-learning recipe. (Phase 4 Steps 1–2.)

---

## 2. The reproducibility checklist a reviewer looks for

Make sure the section states, somewhere:
- exact **split ratios + stratification + patient-level grouping + seed** (and how rare-subject classes were allocated),
- **ADASYN placement** (after split, train only) and target distribution,
- every **hyperparameter** (optimizer, LRs, batch size, epochs, scheduler, early-stopping criterion),
- **framework + library versions** (point to `requirements.txt`),
- **hardware** (Kaggle T4/P100),
- **selection criterion** (best checkpoint by val macro-F1).

If a peer couldn't re-run it from your Methodology, it's not done.

---

## 3. Figures and equations carry the load

- **Equations:** softmax, the weighted-voting sum, macro-F1, and a one-line ADASYN `rᵢ`/`G` summary.
- **Figures:** preprocessing-pipeline diagram, three-backbone architecture sketch, ensemble-fusion schematic. Generate with a **consistent style**.
- Reference every figure/table/equation from the prose — no orphans.

### Resources (a few hours total)
- Your Phase 1–5 learning notes — almost all the text is already there; this step is *assembly*, not invention.
- LaTeX `algorithm`/`algorithmic` packages for the ADASYN pseudocode; `booktabs` for clean tables.
- A diagram tool (draw.io / TikZ) for the architecture and pipeline figures.

---

## 4. Your task (verifies Step 3 — do it before moving on)

- [ ] **1.** Draft all 7 subsections in pipeline order, pulling values from your notes (no invented numbers).
- [ ] **2.** Create the preprocessing, architecture, and fusion figures in a consistent style.
- [ ] **3.** Include the hyperparameter table and the before/after-ADASYN distribution table.
- [ ] **4.** Run the reproducibility checklist (§2) — patch any missing detail.

✅ When a peer could reproduce your pipeline from this section alone, move on.

## When you're ready

> **Phase 7 · Step 4 — Draft Results, Discussion, and Conclusion.**
