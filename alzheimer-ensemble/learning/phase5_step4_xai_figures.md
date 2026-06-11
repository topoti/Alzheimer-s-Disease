# Phase 5 · Step 4 — Side-by-Side XAI Figures & Clinical Sanity Check

> **Phase 5 — XAI** (closes Phase 5)
> The goal of this step: assemble the triple-XAI comparison figures that go in the paper, and have a clinician confirm the highlighted regions make medical sense.

The one question this step answers:

> **Do all three explanation methods point at clinically meaningful regions — and can I show it in one figure?**

---

## 1. The 4-column figure (your paper's XAI centerpiece)

For each chosen case, lay out **four panels side by side**:

| Original MRI | Grad-CAM | SHAP | LIME |
|---|---|---|---|
| the input slice | where the ensemble looked | signed how-much attribution | supporting superpixels |

Build `notebooks/07_xai_figures.ipynb` to generate these for the **~20 curated cases** (same images across SHAP/LIME, plus the Grad-CAM you made in Step 1). Include a mix of:
- **Correct dementia cases** — to show the methods converge on plausible regions (the trust-building story).
- **Incorrect cases** — to show *why* the model failed (the honest failure-analysis story).

Keep a **consistent visual style** (same colormap, same layout) so the figures look like a coherent set.

---

## 2. The triangulation argument

The contribution isn't any single method — it's that **three independent methods agree**. In your figure captions and Discussion, point out where Grad-CAM (where), SHAP (how much), and LIME (which chunks) all land on the same region. That convergence is far more convincing than one heatmap, and it's exactly what the reference paper (Grad-CAM + Saliency only) can't claim quantitatively.

---

## 3. The clinical sanity check (don't skip)

Have a **clinician / advisor** look at the highlighted regions and confirm they correspond to known Alzheimer's pathology (medial temporal lobe, hippocampus, ventricular enlargement). This is the plan's Phase 5 verification and it's what turns "the heatmaps look reasonable to me" into a defensible clinical claim. Record their assessment in a sentence or two — it goes in the Discussion.

> If the explanations consistently highlight *non-brain* regions (skull, background, scanner text), that's a real finding — investigate before claiming clinical validity. Better to catch it now than in review.

### Resources (a few hours total)
- `matplotlib` subplots for the 4-column grid; keep a shared colormap.
- Your Step 1–3 saved overlays (Grad-CAM, SHAP, LIME) as the inputs.
- A reference figure on Alzheimer's MRI biomarkers (hippocampal atrophy, ventricle enlargement) for the sanity check.

---

## 4. Your task (verifies Step 4 — and closes Phase 5)

- [ ] **1.** Build the 4-column (MRI | Grad-CAM | SHAP | LIME) figures for the curated cases, consistent style.
- [ ] **2.** Include both correct and incorrect cases; write captions noting where the three methods agree.
- [ ] **3.** Get a clinician/advisor to confirm regions match known Alzheimer's pathology; record their note.
- [ ] **4.** Flag any case where explanations highlight non-brain regions.

✅ Phase 5 verification (plan): explanations consistently highlight medial-temporal/hippocampal regions on dementia cases, confirmed by a clinician. When done, you're ready to evaluate rigorously.

## When you're ready

> **Phase 6 · Step 1 — The metrics suite: why macro-F1 and per-class recall over accuracy.**
