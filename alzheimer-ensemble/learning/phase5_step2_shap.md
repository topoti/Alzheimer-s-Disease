# Phase 5 · Step 2 — SHAP on the Ensemble

> **Phase 5 — XAI**
> The goal of this step: produce signed, quantitative attributions with SHAP — *how much* each region pushed toward or away from the predicted class.

The one question this step answers:

> **Beyond *where* the model looked, *how much* did each region contribute, and in which direction?**

---

## 1. SHAP recap (from Phase 1 Step 4)

SHAP assigns each feature/region a **Shapley value** — a game-theory-fair share of credit for the prediction. Key properties:
- Values are **signed**: positive = pushed toward the class, negative = pushed away.
- They **sum to the prediction** (minus a baseline) — a satisfying accounting property no heatmap gives you.
- It's **slow** — which is why you run it on a *subset*.

SHAP answers **"how much (and which direction) did each region matter?"**

---

## 2. The ensemble move (decision from Phase 1 Step 4)

SHAP only needs a function `predict(x) → probabilities`. You already built the ensemble's combined `predict` callable in Phase 4 Step 4 — **hand that to SHAP** and it explains the *ensemble* as if it were one model. No per-model averaging needed here (unlike Grad-CAM).

For images, use `shap.DeepExplainer` or `shap.GradientExplainer` with a small **background set** (a handful of representative training images the explainer uses as the "baseline" reference).

---

## 3. Scope it (SHAP is expensive)

Run SHAP on **~20 representative test images** (the plan's number) — a few per class, mixing correct and incorrect predictions, and ideally the *same* images you'll use for LIME (Step 3) so the 4-column figure compares like-for-like. Trying to run SHAP on the whole test set will blow your time/memory budget; Grad-CAM (cheap) gives you full coverage, SHAP gives you depth on a curated few.

> If SHAP still OOMs: shrink the background set, reduce image resolution for the explainer, or fall back to `GradientExplainer` (lighter than `DeepExplainer`).

Save the SHAP attribution maps to `results/figures/` for the third column of the XAI figure.

### Resources (a few hours total)
- **`shap`** docs — `DeepExplainer` / `GradientExplainer` image examples, and `shap.image_plot`.
- **Lundberg & Lee (2017)** abstract — the citation + the "sums to the prediction" property.

---

## 4. Your task (verifies Step 2 — do it before moving on)

- [ ] **1.** Pass the ensemble's `predict(x)→probs` callable to a SHAP image explainer with a small background set.
- [ ] **2.** Pick **~20 representative test images** (few per class, correct + incorrect); reuse them for LIME.
- [ ] **3.** Generate and save signed SHAP attribution maps; confirm positive/negative regions read sensibly.
- [ ] **4.** Note your memory/time mitigation if SHAP was slow (background size, resolution, explainer choice).

✅ When you have ~20 SHAP maps on the ensemble and they make rough clinical sense, move on.

## When you're ready

> **Phase 5 · Step 3 — LIME on the ensemble.**
