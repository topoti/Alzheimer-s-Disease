# Phase 2 · Step 3 — The Classifier Head & Loading All Three from `timm`

> **Phase 2 — Architecture Selection** (closes Phase 2 — first tiny bit of code)
> The goal of this step: design the new 4-class head you bolt onto each pretrained backbone, and verify all three models load and forward without error.

The one question this step answers:

> **How do I turn an ImageNet-pretrained backbone (1000 classes) into my 4-class Alzheimer's classifier?**

---

## 1. Why a backbone needs a new head

A pretrained model has two parts:
- **Backbone** — the convolutional stack that turns an image into a feature vector. Pretrained on ImageNet, it already "knows" edges, textures, shapes. **You keep this.**
- **Head** — the final classifier that maps features → 1000 ImageNet classes. **You throw this away** and replace it with one that maps features → **4** Alzheimer's classes.

The backbone is the experienced analyst; the head is the new specialist you hire to read its reports for *your* specific question (Phase 4's transfer-learning analogy).

---

## 2. The head design (locked)

```
GlobalAveragePool  →  Dropout(0.3)  →  Linear(num_features, 4)
```

- **Global average pool** collapses the feature map to one vector per image.
- **Dropout(0.3)** randomly zeroes 30% of features during training → regularization that fights overfitting (still standard practice even on a dataset this large, ~86k images).
- **Linear(num_features, 4)** is the actual 4-class classifier. `num_features` differs per backbone (e.g. ~2048 for ResNet50, ~1536 for EffNetB3, ~1024 for DenseNet121) — read it from the model, don't hardcode.

`timm` makes this trivial: `timm.create_model(name, pretrained=True, num_classes=4)` already swaps in a fresh 4-class head. You can pass `drop_rate=0.3` for the dropout. (You may keep the default head or build the explicit `Sequential` above — both are fine; document which.)

---

## 3. The sanity check (Phase 2's verification)

The plan's success criterion for Phase 2 is literally: **all three models load via `timm.create_model(name, pretrained=True, num_classes=4)` without error.** Concretely, prove:

1. Each of `resnet50`, `efficientnet_b3`, `densenet121` instantiates with `num_classes=4`.
2. A dummy forward pass returns shape `(batch, 4)` — **using each model's correct input size** (224 for ResNet/DenseNet, 300 for EffNet).
3. You can read each backbone's `num_features` (so the head's `Linear` is correct).

This is the right place to write the **model factory** at `src/models.py`: one `build_model(name)` function returning a ready 4-class model, reused everywhere downstream.

```
# conceptual — the factory you'll write
def build_model(name):           # "resnet50" | "efficientnet_b3" | "densenet121"
    return timm.create_model(name, pretrained=True, num_classes=4, drop_rate=0.3)
```

---

## 4. Don't forget reproducibility (carry it from here on)

Before any model code runs, set seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`. You'll formalize this in Phase 8, but get in the habit now so sanity-check runs are repeatable.

### Resources (a few hours total)
- **`timm` quickstart** — `create_model`, `num_classes`, `drop_rate`, and `model.num_features`.
- PyTorch docs on `nn.Dropout` and `nn.Linear` (one-minute read each).

---

## 5. Your task (verifies Step 3 — and closes Phase 2)

- [ ] **1.** Write the three-layer head from memory and say what each layer is for.
- [ ] **2.** In a scratch notebook, load all three models with `num_classes=4` and run a dummy forward at the *correct input size* for each; confirm output shape `(B, 4)`.
- [ ] **3.** Print each backbone's `num_features` and note them down.
- [ ] **4.** Draft the `build_model(name)` factory signature you'll drop into `src/models.py`.

✅ Phase 2 verification (from the plan): all three load and forward without error. When that's green, you're ready to handle data.

## When you're ready

> **Phase 3 · Step 1 — The OASIS dataset and exploratory data analysis (EDA).**
