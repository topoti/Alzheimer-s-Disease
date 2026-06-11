# Phase 5 · Step 1 — Grad-CAM: Per-Model & Averaged Ensemble Heatmaps

> **Phase 5 — XAI** (explaining the model)
> The goal of this step: produce Grad-CAM heatmaps for each backbone and combine them into one "what the ensemble saw" map.

The one question this step answers:

> **Where in the MRI did each model — and the ensemble as a whole — look when it made its prediction?**

---

## 1. Grad-CAM recap (from Phase 1 Step 4)

Grad-CAM answers **"where did the model look?"** It uses the gradients flowing into the **last convolutional layer** to weight that layer's feature maps, producing a coarse heatmap over the image. Hot regions = spatial locations that most increased the predicted class score.

You need to pick the **target layer** for each backbone — the last conv block:
- ResNet50 → last bottleneck block (`layer4`).
- EfficientNetB3 → final conv stage.
- DenseNet121 → last dense block / final norm.

`pytorch-grad-cam` handles the mechanics; you just supply the model and target layer.

---

## 2. The ensemble twist (the decision from Phase 1 Step 4)

You're explaining an *ensemble*, so:
1. Run Grad-CAM on **each of the 3 backbones** for the same image.
2. **Normalize** each heatmap to [0,1].
3. **Pixel-wise average** the three → one ensemble heatmap.

This reflects what the committee collectively attended to, not just one member. (Heatmaps are on the same spatial grid once resized back to the input image, so averaging is well-defined — resize all three to the original MRI size first.)

---

## 3. What to generate

Per the plan: Grad-CAM for **5 correct + 5 incorrect predictions per class** (≈40 images) **per model**, then the averaged ensemble version. The correct-vs-incorrect contrast is gold for the paper:
- **Correct dementia cases:** does the heatmap sit on medial-temporal / hippocampal regions and ventricles? (clinical plausibility)
- **Incorrect cases:** is the model distracted by edges, skull, or scanner artifacts? (failure analysis)

Save overlays (heatmap on the original MRI) to `results/figures/` — these become the Grad-CAM column of your 4-column XAI figure (Phase 5 Step 4).

### Resources (a few hours total)
- **`pytorch-grad-cam`** (Jacob Gildenblat) README — `GradCAM`, `target_layers`, `ClassifierOutputTarget`, and the `show_cam_on_image` overlay helper.
- A "which layer for Grad-CAM" note for each architecture family.

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Identify and confirm the correct **target layer** for each of the three backbones.
- [ ] **2.** Generate per-model Grad-CAM for a handful of correct + incorrect cases per class.
- [ ] **3.** Normalize and **pixel-wise average** the three heatmaps → ensemble Grad-CAM; save overlays.
- [ ] **4.** Write one sentence on whether correct dementia cases highlight clinically plausible regions.

✅ Phase 5 verification (plan): heatmaps consistently highlight medial-temporal/hippocampal regions on dementia-positive cases. When you see that pattern, move on.

## When you're ready

> **Phase 5 · Step 2 — SHAP on the ensemble.**
