# Phase 5 · Step 3 — LIME on the Ensemble

> **Phase 5 — XAI**
> The goal of this step: produce LIME explanations — the handful of image chunks (superpixels) that most support the ensemble's prediction.

The one question this step answers:

> **Which simple visual chunks, if removed, would change the ensemble's prediction?**

---

## 1. LIME recap (from Phase 1 Step 4)

LIME builds a **local surrogate**: it carves the image into **superpixels**, randomly turns some on/off to create perturbed variants, watches how the ensemble's prediction reacts, and fits a tiny interpretable (linear) model to those reactions *near this one image*. The superpixels with the biggest positive weights are the explanation.

LIME answers **"which chunks matter?"** — coarser than SHAP, but very intuitive (a clinician sees a few outlined regions, not a per-pixel field).

---

## 2. The ensemble move (same as SHAP)

Like SHAP, LIME just needs a `predict(x) → probabilities` function — feed it the **ensemble's combined predict callable** from Phase 4 Step 4. `LimeImageExplainer.explain_instance` does the perturbation and surrogate fitting; you then visualize the top supporting superpixels.

---

## 3. Use the *same* images as SHAP

Run LIME on the **same ~20 test images** you chose for SHAP. Why: the paper's headline XAI figure is a 4-column comparison (MRI | Grad-CAM | SHAP | LIME) for the *same* cases — so the reader can see all three methods agree (or disagree) on identical images. Different image sets would break the comparison.

A couple of LIME knobs that matter:
- **Segmentation** (superpixel algorithm, e.g. `quickshift`/`slic`) — controls how the image is chunked; mention your choice since LIME's output depends on it.
- **`num_samples`** — more perturbations = more stable explanation but slower.

Save the superpixel-overlay images to `results/figures/` for the fourth column.

### Resources (a few hours total)
- **`lime`** docs — `LimeImageExplainer`, `explain_instance`, `get_image_and_mask`.
- **Ribeiro et al. (2016)** abstract — the citation + the husky-vs-wolf "wrong reason" cautionary example.
- `skimage.segmentation` (`slic`, `quickshift`) for the superpixel step.

---

## 4. Your task (verifies Step 3 — do it before moving on)

- [ ] **1.** Pass the ensemble's `predict(x)→probs` callable to `LimeImageExplainer` on the **same ~20 images** as SHAP.
- [ ] **2.** Choose and record a superpixel segmentation method and `num_samples`.
- [ ] **3.** Generate and save superpixel-overlay explanations for each image.
- [ ] **4.** Note one case where LIME and SHAP agree, and one where they differ (useful for Discussion).

✅ When LIME explanations exist for the same images as SHAP, move on to assemble the figures.

## When you're ready

> **Phase 5 · Step 4 — Build the side-by-side XAI figures and run the clinical sanity check.**
