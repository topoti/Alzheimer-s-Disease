# Phase 4 · Step 3 — Train & Checkpoint All Three Models

> **Phase 4 — Model Training**
> The goal of this step: actually train ResNet50, EfficientNetB3, and DenseNet121 to completion and save the best checkpoint of each.

The one question this step answers:

> **How do I get three trained, saved models — within Kaggle's limits — that are good enough to ensemble?**

---

## 1. Run the three, one at a time

Call `train_one_model()` once per backbone, each in its own notebook (`03_train_resnet50`, `04_train_effnetb3`, `05_train_densenet121`) so a crash in one doesn't lose the others:

- **ResNet50** → train → log → save best (224×224 input).
- **EfficientNetB3** → train → log → save best (**300×300** input; drop batch size to 16 if OOM).
- **DenseNet121** → train → log → save best (224×224 input).

Each saves a checkpoint to `checkpoints/` plus its **val macro-F1** (you need these three F1 scores for the ensemble weights in Phase 4 Step 4).

---

## 2. Survive Kaggle (the practical reality)

Kaggle gives ~30 GPU-hrs/week and **kills long idle/over-limit sessions**. Defend yourself:
- **Checkpoint every epoch**, not just at the end — so a timeout costs one epoch, not the whole run.
- **Resume from last checkpoint** support in your loop.
- Start with **shorter epochs / a subset** to confirm the full pipeline before committing hours.
- Budget ~**12 GPU-hrs** total (≈3 hrs × 3 models + ensemble + XAI). EffNetB3 at 300×300 is the slowest — give it the most headroom.

---

## 3. The bar each model must clear

Phase 4's verification (from the plan): **each model reaches >85% val accuracy**, and (next step) the **ensemble's val F1 beats the best single model's val F1.**

If a model stalls well below 85%:
- Check the input size (EffNet at 224 instead of 300 is the classic mistake).
- Check normalization (ImageNet stats), and that labels are 0–3.
- Confirm you're reading the **ADASYN-balanced** train cache for training but **original** val for evaluation.
- Consider class-weighted or focal loss if Moderate recall stays near zero (a Phase 6 risk-mitigation lever).

---

## 4. What you walk away with

Three files in `checkpoints/` + a small table of each model's **val accuracy, val macro-F1, val per-class recall**. That table is the first row-block of your Phase 6 comparison table, and the F1 values are the raw material for the ensemble weights.

### Resources (a few hours total)
- Kaggle "save & version" / output persistence docs (getting checkpoints out of `/kaggle/working/`).
- `torch.save` / `torch.load` checkpoint patterns (save model state_dict + epoch + best metric).

---

## 5. Your task (verifies Step 3 — do it before moving on)

- [ ] **1.** Train all three models; save best-by-val-F1 checkpoints to `checkpoints/`.
- [ ] **2.** Record each model's **val accuracy + val macro-F1 + per-class recall**.
- [ ] **3.** Confirm each model clears **>85% val accuracy** (debug per §3 if not).
- [ ] **4.** Verify checkpoints reload correctly (load → forward → same metric).

✅ When three checkpoints exist and each clears 85% val accuracy, move on to the ensemble.

## When you're ready

> **Phase 4 · Step 4 — Ensemble inference: weighted soft voting over the three checkpoints.**
