# Phase 4 · Step 4 — Ensemble Inference: Weighted Soft Voting

> **Phase 4 — Model Training** (closes Phase 4)
> The goal of this step: combine the three trained checkpoints into one predictor using the weighted soft voting you designed in Phase 2 Step 2.

The one question this step answers:

> **How do I turn three saved models into one ensemble prediction — and confirm the ensemble actually helps?**

---

## 1. Compute the weights (from validation, once)

From Phase 4 Step 3 you have each model's **val macro-F1**. Normalize them to sum to 1:

```
w_i = F1_i / (F1_resnet + F1_effnet + F1_densenet)
```

Freeze these three weights. They come from **validation**, never test (the leakage rule from Phase 2 Step 2).

---

## 2. The inference loop (lives in `src/ensemble.py`)

```
load 3 checkpoints (eval mode)
for each test sample x:
    p_resnet   = softmax(ResNet50(x_224))        # (4,)
    p_effnet   = softmax(EfficientNetB3(x_300))   # (4,)  ← 300×300 input!
    p_densenet = softmax(DenseNet121(x_224))      # (4,)
    p_ensemble = w1*p_resnet + w2*p_effnet + w3*p_densenet
    prediction = argmax(p_ensemble)
```

Two things the loop must get right:
- **Route each model its correct input resolution** (224 vs 300) — the recurring gotcha. Easiest: keep two parallel test tensors/loaders.
- **`model.eval()` + `torch.no_grad()`** for all three — no dropout, no gradient, deterministic.

---

## 3. Confirm the ensemble earns its place

Phase 4's headline verification: **ensemble val F1 > best single-model val F1.** Check on **validation** first (where you're allowed to look freely). If the ensemble *doesn't* beat the best single model:
- Re-examine the weights (a much weaker model dragging the average → its F1 weight should already down-weight it, but double-check).
- Confirm all three are genuinely decorrelated (if two always agree, you effectively have two votes).
- Consider equal weights as a baseline comparison.

Only after validation looks right do you run the ensemble on **test once** — those are your headline numbers for Phase 6.

---

## 4. Save the ensemble's predictions

Persist the test-set predicted labels and the combined probability vectors. Phase 6 (metrics, confusion matrix, ROC) and Phase 5 (SHAP/LIME use the ensemble's `predict` function) both consume these. Wrapping the ensemble as a single `predict(x) → probs` callable now makes Phase 5 trivial.

### Resources (a few hours total)
- Re-read your own Phase 2 Step 2 notes (soft voting formula + weight rule).
- `torch.nn.functional.softmax`, `model.eval()`, `torch.no_grad()` docs.

---

## 5. Your task (verifies Step 4 — and closes Phase 4)

- [ ] **1.** Compute normalized weights from the three **val** macro-F1 scores.
- [ ] **2.** Implement the weighted soft-voting loop, routing correct input sizes; expose a `predict(x)→probs` callable.
- [ ] **3.** Confirm **ensemble val F1 > best single-model val F1** (debug per §3 if not).
- [ ] **4.** Run on **test once**; save predicted labels + probability vectors for Phases 5 & 6.

✅ Phase 4 verification (plan): all 3 models >85% val acc **and** ensemble val F1 > best single. When green, you're ready to explain the model.

## When you're ready

> **Phase 5 · Step 1 — Grad-CAM: per-model heatmaps and the averaged ensemble heatmap.**
