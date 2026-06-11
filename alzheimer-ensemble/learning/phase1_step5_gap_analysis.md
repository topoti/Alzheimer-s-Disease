# Phase 1 · Step 5 — Gap Analysis & the 1-Page Gap Statement

> **Phase 1 — Research Foundation** (the capstone of Phase 1 — last reading step)
> The goal of this step: combine Steps 1–4 into a single crisp argument for *exactly how your work improves on the paper you're beating* — this becomes the backbone of your Introduction and Related Work.

The one question this step answers:

> **In one page, what is the gap in Mahmud et al. (2024), and how do my three pillars close it?**

---

## 1. The paper you're beating, in one breath

**Mahmud et al. (2024), *Diagnostics*** — *"An Explainable AI Paradigm for Alzheimer's Diagnosis Using Deep Transfer Learning."*
- **Result:** ~96% accuracy on OASIS 4-class MRI classification.
- **How:** same-family ensembles (VGG16+VGG19, DenseNet169+DenseNet201), **plain augmentation** for imbalance, **Grad-CAM + Saliency** for explanation.

Know their **per-class** numbers, not just the headline 96% — especially recall on **Moderate** (the rare class). That's where your improvement will show up most, and it's what a reviewer checks.

---

## 2. The gap table (the heart of your contribution)

| Aspect | Reference paper | Your work | Why it's an improvement |
|---|---|---|---|
| **Imbalance handling** | Plain image augmentation | **ADASYN** | Targets the *hard* borderline minority samples, not just bulk count (Step 2) |
| **Ensemble composition** | Same-family (VGG+VGG, DenseNet+DenseNet) | **Heterogeneous paradigms** (residual + compound-scaled + densely-connected) | Different design biases → less correlated errors → bigger ensemble gain (Step 3) |
| **XAI methods** | Grad-CAM + Saliency | **Grad-CAM + SHAP + LIME** | Adds *quantitative* feature attribution on top of pure visual heatmaps (Step 4) |
| **Backbone modernity** | VGG (2014), DenseNet (2017) | ResNet / DenseNet / EfficientNet (2015–2019) | EfficientNet brings modern compound scaling |

This table appears (in some form) in your **Related Work** and **Discussion**. Memorize the middle two columns.

---

## 3. The novelty-positioning sentence (write it now, reuse it forever)

This sentence goes in both the Introduction and the Discussion. Draft your own version of:

> *"Unlike Mahmud et al. (2024), who combined models from the same architectural family (e.g., VGG16 + VGG19), our ensemble integrates three CNN backbones built on fundamentally different design paradigms — residual learning, compound scaling, and dense connectivity — which we hypothesize produces more decorrelated errors and thus stronger ensemble gain. We further replace plain image augmentation with ADASYN, which specifically targets borderline minority-class samples, and extend explainability beyond Grad-CAM by adding SHAP and LIME for quantitative feature attribution."*

Notice it (a) names the baseline, (b) states all three pillars, (c) uses the safe word "paradigms," (d) hedges with "we hypothesize." That's the template for honest novelty claims.

---

## 4. Don't forget the license check

Before you build anything on OASIS, confirm its **data-use / license terms permit publication** of derived results and figures. OASIS has a data-use agreement; note its citation requirements. A locked-down license discovered *after* you write the paper is a disaster — clear it now.

### Resources (a few hours total)
- **Mahmud et al. (2024)** in *Diagnostics* — re-read Abstract, Intro, and the **results table**; copy their per-class metrics into your notes.
- 2–3 recent (2023–2025) **review/survey papers** on deep learning for Alzheimer's MRI — for the Related Work citations and to confirm nobody already did your exact combo.
- The **OASIS** dataset page / data-use agreement.

---

## 5. Your task (verifies Step 5 — and closes Phase 1)

- [ ] **1. Write the 1-page gap statement:** one tight page that states the problem (Step 1), the baseline's three weaknesses, and your three fixes, ending with the novelty-positioning sentence. *This is a real deliverable — you'll paste chunks of it into the paper.*
- [ ] **2.** Fill in Mahmud et al.'s **per-class recall** (especially Moderate) in your notes, so you have concrete numbers to beat.
- [ ] **3.** Record the **OASIS license terms** and required citation in one line — confirm publication is allowed.
- [ ] **4. Self-test:** can you recite the gap table's middle two columns from memory? If yes, Phase 1 is complete.

✅ Phase 1 verification (from the plan): *you can write the 1-page gap statement from memory.* When you can, you're ready to design.

## When you're ready

> **Phase 2 · Step 1 — Meet the three backbones: ResNet50, EfficientNetB3, DenseNet121.**
