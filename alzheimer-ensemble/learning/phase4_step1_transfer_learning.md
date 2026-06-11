# Phase 4 · Step 1 — Transfer Learning: Freeze, Then Fine-Tune

> **Phase 4 — Model Training** (the longest phase — 1–2 weeks)
> The goal of this step: understand the two-stage transfer-learning strategy you'll apply identically to all three models.

The one question this step answers:

> **How do I adapt an ImageNet-pretrained backbone to MRI without destroying what it already knows?**

---

## 1. Why not just train everything from scratch?

You have only ~10,240 (post-ADASYN) training images — far too few to train a 25M-parameter network from random weights. Transfer learning reuses ImageNet knowledge (edges, textures, shapes) and only *adapts* it. The risk: if you blast the pretrained weights with a big learning rate, you **erase** that knowledge before the model adapts. The two-stage recipe avoids this.

---

## 2. The two stages

**Stage 1 — Feature extraction (≈3 epochs).**
- **Freeze** the entire backbone (no gradient updates).
- Train **only the new 4-class head** (from Phase 2 Step 3).
- This "warms up" the randomly-initialized head so it produces sensible outputs *before* you let it influence the backbone.

**Stage 2 — Fine-tuning (≈15–25 epochs).**
- **Unfreeze the last 2 blocks** of the backbone + keep the head trainable.
- Train with a **small learning rate (1e-4)** so the pretrained features get *gently* adjusted, not wrecked.
- **Keep early layers frozen** — they detect generic edges/textures that transfer perfectly to MRI; no reason to disturb them.

> **Memorize the analogy:** Stage 1 = hire a specialist (the head) to interpret reports written by seasoned general analysts (the frozen backbone). Stage 2 = let the specialist coach the *senior* analysts (last blocks) to tweak their reports slightly for MRI — while the junior analysts (early layers) keep doing their reliable basics.

---

## 3. Why this ordering matters

If you skipped Stage 1 and fine-tuned everything immediately, the head's large random-init gradients would flow back into the backbone and corrupt good features on the very first batches. Warming up the head first means that by the time the backbone unfreezes, the gradients flowing into it are already *meaningful*.

This whole strategy is implemented once inside `train_one_model()` (next step) and reused for all three backbones — you don't write it three times.

### Resources (a few hours total)
- PyTorch transfer-learning tutorial — the "finetuning vs feature extraction" section (`requires_grad=False` to freeze).
- Any "discriminative / layer-wise learning rate" note for the intuition behind small backbone LR.

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Explain in your own words why the head is warmed up *before* the backbone unfreezes.
- [ ] **2.** State which layers are frozen vs trainable in each stage, and the learning rate used in Stage 2.
- [ ] **3.** Explain why early layers stay frozen even in Stage 2.
- [ ] **4.** Sketch how you'd toggle `requires_grad` to implement freeze/unfreeze in PyTorch.

✅ When the two-stage recipe and *why each stage exists* is clear, move on.

## When you're ready

> **Phase 4 · Step 2 — The reusable training loop, hyperparameters, AMP, and early stopping.**
