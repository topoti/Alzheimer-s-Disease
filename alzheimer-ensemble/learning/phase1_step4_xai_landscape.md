# Phase 1 · Step 4 — The XAI Landscape: Grad-CAM, SHAP, LIME (Pillar 3)

> **Phase 1 — Research Foundation** (still pure understanding)
> The goal of this step: understand what each of the three explainability methods *shows*, what question each *answers*, and why a doctor (and a regulator) needs them — because triple-XAI is your third contribution.

The one question this step answers:

> **Once the model says "Mild Dementia, 87% confident," how do we show *why* — and why do three different explanations beat one?**

---

## 1. Why explainability is non-negotiable in medical AI

A black box that says "Mild Dementia, 87%" is **useless and unsafe** in a clinic:
- **Regulators** (FDA, EMA) require *some* explainability for AI medical devices.
- **Doctors won't trust** a number they can't interrogate — they need to see the *evidence*.
- If the model's "reason" is a known Alzheimer's region (hippocampus, ventricles), trust goes up. If it's reacting to a scanner artifact in the corner, you've caught a fake.

The reference paper used **Grad-CAM + Saliency** (two *visual* methods). You add **SHAP + LIME** so you also get *quantitative* attribution — numbers you can analyze across the whole test set, not just pretty heatmaps.

---

## 2. The three methods, side by side

| Method | What it produces | What a doctor sees | The question it answers |
|---|---|---|---|
| **Grad-CAM** | A heatmap overlaid on the MRI showing which spatial regions most drove the prediction | A colored blob on the brain | **"*Where* did the model look?"** |
| **SHAP** | Per-pixel / per-region attribution values that **sum to the prediction** (rooted in cooperative game theory) | A map of positive/negative contribution scores | **"*How much* did each region push toward or away from this class?"** |
| **LIME** | The handful of superpixels (image chunks) that most support the prediction, found by perturbing the image and watching the output | A few clearly outlined regions | **"Which simple chunks, if removed, would flip the prediction?"** |

Plain-language intuition for each:
- **Grad-CAM** is *gradient-based* — it asks the network "which spatial locations, if nudged, change this class score most?" Fast, but coarse (low-resolution blob).
- **SHAP** comes from **game theory** (Shapley values): treat each feature/region as a "player" and fairly distribute credit for the final prediction among them. Principled and signed (push-for vs push-against), but slow.
- **LIME** builds a *simple local surrogate*: it jiggles the input (turns superpixels on/off), sees how the prediction reacts, and fits a tiny interpretable model to those reactions near this one image. Intuitive, but only locally faithful.

> **One-line mnemonic:** Grad-CAM = *where*, SHAP = *how much (signed)*, LIME = *which chunks matter*.

---

## 3. Why three instead of one

No single XAI method is "the truth" — each has blind spots. Grad-CAM is coarse; LIME depends on how you carve superpixels; SHAP is expensive. When **all three independently point at the hippocampal region** on a dementia case, that agreement is far more convincing than any one heatmap alone. Triangulation *is* the contribution — and it gives you both visual (Grad-CAM) and quantitative (SHAP, LIME) evidence in the same paper.

---

## 4. The ensemble subtlety (preview of Phase 5)

You're explaining an **ensemble of three models**, not one model — something the reference paper sidesteps. The plan's resolution (you'll implement it in Phase 5):
- **Grad-CAM:** run it on each of the 3 backbones, then **average the normalized heatmaps** → one "what the committee saw" map.
- **SHAP & LIME:** these only need a `predict(x) → probabilities` function, so you hand them the **ensemble's combined probability output** directly — the ensemble behaves as one model.

Hold this idea loosely now; just know XAI-on-an-ensemble is a real design choice you've already got an answer for.

### Resources (a few hours total)
- **Selvaraju et al. (2017)** — *Grad-CAM* paper, abstract + Fig 1.
- **Lundberg & Lee (2017)** — *"A Unified Approach to Interpreting Model Predictions"* (SHAP), intro.
- **Ribeiro et al. (2016)** — *"Why Should I Trust You?"* (LIME), intro + the husky-vs-wolf example.
- Library docs you'll actually use: `pytorch-grad-cam`, `shap`, `lime`.

---

## 5. Your task (verifies Step 4 — do it before moving on)

- [ ] **1. Fill the table from memory:** for each method, write the *one question it answers* in your own words.
- [ ] **2.** Explain in two sentences why you add SHAP and LIME on top of Grad-CAM (hint: visual vs quantitative, triangulation).
- [ ] **3.** State how you'll apply each method to an *ensemble* (which one averages heatmaps, which two use the combined predict-function).
- [ ] **4.** Name one Alzheimer's-relevant brain region you'd *expect* a correct explanation to highlight (so you can sanity-check later).

✅ When you can say "Grad-CAM = where, SHAP = how much, LIME = which chunks" and justify using all three, move on.

## When you're ready

> **Phase 1 · Step 5 — Pull it together: the gap analysis vs Mahmud et al. and a 1-page gap statement.**
