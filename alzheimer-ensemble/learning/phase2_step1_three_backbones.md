# Phase 2 · Step 1 — Meet the Three Backbones

> **Phase 2 — Architecture Selection** (design decisions, still little/no code)
> The goal of this step: know each backbone's design paradigm, parameter count, and required input size cold — and *why each one earns its seat* in the ensemble.

The one question this step answers:

> **Which three CNNs am I ensembling, and what does each bring that the others don't?**

---

## 1. The three backbones (locked decision)

| Model | Design paradigm | Why it suits MRI | Pretrained params | Input size |
|---|---|---|---|---|
| **ResNet50** | Residual learning (skip connections carry gradients through 50 layers) | Strong, reliable general feature extractor; the default baseline in medical imaging | ~25.6M | **224×224** |
| **EfficientNetB3** | Compound scaling (width/depth/resolution jointly scaled) + MBConv (depthwise-separable) blocks | Best accuracy-per-parameter; handles texture-rich MRI well | ~12M | **300×300** |
| **DenseNet121** | Dense connectivity (every layer sees all previous layers' feature maps) | Excellent feature reuse; the sibling of the reference paper's DenseNet169/201 | ~8M | **224×224** |

All three load pretrained on ImageNet via `timm` (and `torchvision`).

**The one operational gotcha to burn in now:** EfficientNetB3 wants **300×300** input, the other two want **224×224**. Your data pipeline (Phase 3) must produce *two* resolutions, or apply per-model transforms. Forgetting this is the #1 silent bug in this project.

---

## 2. Why these three and not three ResNets

You answered this in Phase 1 Step 3, but anchor it to the concrete picks:
- **ResNet50** — the residual idea: each block learns a *correction* to its input.
- **EfficientNetB3** — the efficiency idea: scale all three dimensions by a principled rule, use cheap separable convolutions.
- **DenseNet121** — the reuse idea: concatenate all earlier features so nothing is recomputed.

Three different ideas → three different error patterns → **decorrelated mistakes** → real ensemble gain. Three ResNets of different sizes would be the "siblings" the reference paper used.

> Note the params trend *down* (25.6M → 12M → 8M) but capability stays competitive — a nice efficiency story for the paper.

---

## 3. Where they live in code (preview)

The plan's folder structure puts a **model factory** at `src/models.py` — one function that, given a name (`"resnet50"`, `"efficientnet_b3"`, `"densenet121"`), returns the pretrained backbone with a fresh 4-class head. You'll write it in Phase 2 Step 3. For now, the mental model is: *one factory, three names, identical interface.*

### Resources (a few hours total)
- **`timm` model registry** — search `resnet50`, `efficientnet_b3`, `densenet121`; note the default `input_size` each expects.
- Abstracts of **ResNet** (He 2015), **EfficientNet** (Tan & Le 2019), **DenseNet** (Huang 2017) — just enough to write one sentence on each paradigm.

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** From memory, write each model's paradigm, param count (rough), and input size.
- [ ] **2.** State the input-size gotcha in one sentence and how you'll handle it in the data pipeline.
- [ ] **3.** In one sentence each, justify why all three deserve a seat (tie back to decorrelated errors).

✅ When the 224/224/300 split is something you'd never forget, move on.

## When you're ready

> **Phase 2 · Step 2 — How the three votes combine: weighted soft voting.**
