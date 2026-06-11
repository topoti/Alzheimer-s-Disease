# Phase 3 · Step 2 — The Preprocessing Pipeline

> **Phase 3 — Dataset & Preprocessing**
> The goal of this step: turn a raw MRI JPG into a tensor the pretrained backbones expect — in the right order, with the right normalization.

The one question this step answers:

> **What transforms does each image go through before it reaches the model, and why each one?**

---

## 1. The pipeline, in order (each step has a reason)

1. **Load & resize** — to **224×224** for ResNet/DenseNet, **300×300** for EfficientNet. Maintain two parallel tensors or use per-model transforms (the Step 1/Phase 2 gotcha).
2. **Grayscale → 3-channel** — MRIs are grayscale, but ImageNet-pretrained backbones expect **3 channels**. Duplicate the single gray channel three times. (Confirm in EDA whether the files are already 3-channel.)
3. **Normalize** — subtract ImageNet mean `[0.485, 0.456, 0.406]`, divide by std `[0.229, 0.224, 0.225]`. *Use ImageNet stats* because the backbones were pretrained with them — feeding differently-scaled inputs degrades transfer.
4. **Noise removal (optional, ablate it)** — mild Gaussian blur (σ=0.5) or 3×3 median filter to suppress scanner noise. Treat as a toggle you can turn off to measure its effect.
5. **Brain-region focus (lightweight)** — instead of true skull-stripping (FSL/BET — out of scope), crop a tight bounding box around the non-zero pixels. Simple, robust, beginner-friendly.
6. **Standard augmentation (train only!)** — small rotations (±10°), horizontal flip, brightness jitter (±10%). Applied **online during training**, **never to val/test**.

> **The cardinal split rule:** steps 1–3 (and optionally 4–5) apply to *all* images; step 6 augmentation applies to **training images only**. Augmenting the test set would mean you're not evaluating on real data.

---

## 2. Train transforms vs eval transforms

You'll define **two transform pipelines**:
- **Train:** resize → (crop) → (noise) → augment → grayscale-3ch → normalize → tensor.
- **Eval (val/test):** resize → (crop) → (noise) → grayscale-3ch → normalize → tensor. **No augmentation.**

This lives in `src/data.py`. `albumentations` (in your requirements) or `torchvision.transforms` both work; `albumentations` is fast and has clean rotation/flip/brightness ops.

---

## 3. Two resolutions, one dataset — the practical pattern

Because EffNet needs 300 and the others 224, the cleanest approach is to make the **resize size a parameter** of your transform/Dataset, and build per-model dataloaders. Don't bake 224 in as a constant — you'll forget and feed EffNet the wrong size.

### Resources (a few hours total)
- `albumentations` docs: `Resize`, `Rotate`, `HorizontalFlip`, `RandomBrightnessContrast`, `Normalize`, `ToTensorV2`.
- `torchvision.transforms` equivalents (`Resize`, `Normalize`, `RandomRotation`, etc.) if you prefer.
- Why ImageNet mean/std: any transfer-learning tutorial's "normalization" note.

---

## 4. Your task (verifies Step 2 — do it before moving on)

- [ ] **1.** Write the 6-step pipeline from memory, marking which steps are train-only.
- [ ] **2.** Implement a `build_transforms(img_size, train: bool)` helper in `src/data.py` that returns the right pipeline.
- [ ] **3.** Run one image through both the train and eval transforms; confirm output tensor shape is `(3, H, W)` with the right `H=W` per model and values roughly in normalized range.
- [ ] **4.** Note which steps you'll **ablate** (noise removal, crop) so you can measure their effect in Phase 6.

✅ When you can produce a correctly-normalized 3-channel tensor at both 224 and 300, move on.

## When you're ready

> **Phase 3 · Step 3 — The stratified train/val/test split and the PyTorch Dataset class.**
