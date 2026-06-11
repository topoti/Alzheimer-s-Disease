# Phase 4 · Step 2 — The Reusable Training Loop & Hyperparameters

> **Phase 4 — Model Training**
> The goal of this step: write **one** `train_one_model()` function — with the right hyperparameters, logging, mixed precision, and early stopping — that you reuse for all three backbones.

The one question this step answers:

> **What does one training run look like, and what knobs control it?**

---

## 1. One function, three models

The plan is explicit: implement a single `train_one_model(model_name, dataloaders, config)` in `src/train.py` and call it three times. Writing the loop once means a bug fixed once is fixed everywhere, and all three models are trained under identical conditions (fair comparison).

---

## 2. The hyperparameters (starting point)

| Hyperparameter | Value | Note |
|---|---|---|
| Optimizer | **AdamW** | Decoupled weight decay → better than Adam for transfer learning |
| Initial LR (head only, Stage 1) | **1e-3** | Standard for a fresh head |
| Fine-tune LR (backbone, Stage 2) | **1e-4** | 10× smaller, protects pretrained features |
| LR scheduler | **CosineAnnealingLR** | Smooth decay, little tuning |
| Batch size | **32** (16 if OOM) | EffNetB3 at 300×300 may need 16 on a T4 |
| Epochs | **3 + 25** | Stage 1 + Stage 2 |
| Loss | **CrossEntropyLoss** | Standard multi-class |
| Weight decay | **1e-4** | Mild regularization |
| Early stopping | **patience = 5** on **val F1** | Stops before overfitting |
| Mixed precision | **`torch.cuda.amp`** | ~2× speedup on T4 |

> Two values to defend in the paper: **F1 (not accuracy) for early stopping & checkpointing**, because F1 respects the minority class; and **AdamW**, because decoupled weight decay regularizes transfer learning better than plain Adam.

---

## 3. What to log every epoch

Track and log (TensorBoard or `wandb`): **train loss, val loss, val accuracy, val macro-F1, val per-class recall.** Per-class recall is the one you watch most — it tells you if Moderate is actually being learned, which overall accuracy hides.

**Save the best checkpoint by val macro-F1, not val accuracy.** On imbalanced data a model can gain accuracy while getting *worse* at Moderate; F1 won't let that hide.

---

## 4. The loop skeleton (conceptual)

```
def train_one_model(model_name, dataloaders, config):
    model = build_model(model_name)            # Phase 2 Step 3 factory
    freeze_backbone(model)                      # Stage 1
    optim = AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler()                        # AMP
    for epoch in range(3):                       # Stage 1: head only
        train_epoch(...); val = evaluate(model, val_loader)
        log(val); maybe_save_best(val.f1)
    unfreeze_last_blocks(model)                  # Stage 2
    optim = AdamW(all_trainable, lr=1e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=25)
    for epoch in range(25):                      # Stage 2: fine-tune
        train_epoch(...); val = evaluate(model, val_loader)
        log(val); maybe_save_best(val.f1)
        if early_stop.update(val.f1): break
    return best_checkpoint_path
```

Inside `train_epoch`, the AMP pattern: `with autocast(): loss = ...`, then `scaler.scale(loss).backward(); scaler.step(optim); scaler.update()`.

The `evaluate(...)` here is the same function you formalize in Phase 6 Step 1 — write a minimal version now, enrich it later.

### Resources (a few hours total)
- PyTorch `torch.cuda.amp` (autocast + GradScaler) docs.
- `torch.optim.AdamW` and `CosineAnnealingLR` docs.
- `wandb` or `tensorboard` quickstart for logging.

---

## 5. Your task (verifies Step 2 — do it before moving on)

- [ ] **1.** Implement `train_one_model(...)` with the two-stage schedule from Step 1.
- [ ] **2.** Wire in AMP, AdamW, cosine schedule, early stopping (patience 5 on val F1).
- [ ] **3.** Log the five per-epoch metrics; checkpoint best by **val macro-F1**.
- [ ] **4.** Smoke-test on a tiny subset (a few batches, 1 epoch) to confirm it runs end-to-end without OOM.

✅ When one model trains for an epoch and logs sensible metrics, move on to train all three.

## When you're ready

> **Phase 4 · Step 3 — Train and checkpoint ResNet50, EfficientNetB3, DenseNet121.**
