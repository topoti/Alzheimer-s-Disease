# Research Plan — Explainable Cross-Paradigm CNN Ensemble with ADASYN for Alzheimer's Disease Classification

## Context

You are planning an academic research project that improves on Mahmud et al. (2024, *Diagnostics*) — "An Explainable AI Paradigm for Alzheimer's Diagnosis Using Deep Transfer Learning" — which achieved 96% accuracy on OASIS MRI 4-class classification using same-family ensembles (VGG16+VGG19, DenseNet169+DenseNet201), plain augmentation for imbalance, and Saliency + Grad-CAM for explainability.

Your contribution has three pillars:

1. **ADASYN** (Adaptive Synthetic Sampling) replaces plain augmentation — generates synthetic minority-class samples focused on hard-to-learn borderline cases.
2. **Heterogeneous-paradigm CNN ensemble** — ResNet50 (residual learning) + EfficientNetB3 (compound-scaled MBConv) + DenseNet121 (dense connectivity). These are three genuinely different *design paradigms* within the CNN family, not three siblings from one family like the reference paper.
3. **Triple-XAI** — Grad-CAM (visual) + SHAP (game-theoretic feature attribution) + LIME (local surrogate models), applied to the ensemble.

**Locked decisions** (from clarifying questions):
- Framework: **PyTorch** (with `timm` for pretrained backbones)
- Trio: **ResNet50 + EfficientNetB3 + DenseNet121** — novelty framed as *"cross-paradigm CNN ensemble"* not cross-architecture
- XAI: **Grad-CAM + SHAP + LIME** (triple)
- Compute: **Kaggle Notebooks** (free T4/P100, OASIS already hosted)

**Honest limitation to acknowledge in paper**: All three backbones are CNNs. Title and framing must say *"cross-paradigm"* or *"heterogeneous CNN ensemble"* — NOT *"cross-architecture"* — otherwise a reviewer will reject on novelty grounds. List "CNN + Transformer ensemble" as future work.

**Suggested revised title:**
> *"An Explainable Heterogeneous-Paradigm CNN Ensemble with Adaptive Synthetic Oversampling for Alzheimer's Disease Classification from MRI"*

---

## Phase 1 — Research Foundation

**Estimated time: 4–6 days (mostly reading + writing notes)**

### Concepts to internalize

**Alzheimer's as a 4-class ML problem.** The OASIS dataset labels each MRI slice into one of four severity stages:
- **Non-Dementia (3200 images)** — healthy brain
- **Very Mild Dementia (2240 images)** — earliest detectable cognitive decline
- **Mild Dementia (896 images)** — clear symptoms, manageable
- **Moderate Dementia (64 images)** — significant impairment

From an ML view, this is **ordinal multi-class classification** (the classes have an order: healthy → mild → moderate). You'll treat it as plain multi-class for now (4 mutually exclusive labels) because the reference paper does.

**Imbalanced dataset — analogy.** Imagine teaching a child to recognize animals using 3200 dog photos, 2240 cat photos, 896 rabbit photos, and only 64 hamster photos. The child will become excellent at dogs and cats, mediocre at rabbits, and almost never guess "hamster" — because guessing "dog" is usually right. A model trained on imbalanced data does exactly this: it learns to over-predict the majority class to maximize accuracy, while failing on the rare class. In medicine this is catastrophic — Moderate Dementia is the rarest class but arguably the most clinically urgent.

**ADASYN — what it is and why it beats plain augmentation.**
- *Plain augmentation* (rotation, flip, zoom on existing images) makes the rare class bigger but doesn't change *which* samples the model struggles with.
- *SMOTE* (Synthetic Minority Oversampling) creates synthetic minority samples by interpolating between a minority sample and its k-nearest minority neighbors. Better than augmentation but treats all minority samples equally.
- *ADASYN* extends SMOTE by **weighting**: it generates more synthetic samples near minority points that are surrounded by majority points (the "borderline" cases the model is most likely to misclassify). Analogy: instead of teaching the child equally about all hamster photos, ADASYN spends more time on hamster photos that *look kind of like rabbits* — the confusing edge cases.
- Caveat for images: ADASYN operates on feature vectors, not raw pixels. We apply it on the flattened feature embeddings from a pretrained CNN, or on the raw images flattened to vectors then reshaped — see Phase 3 for exact placement.

### Gap analysis vs. Mahmud et al. (2024)

| Aspect | Reference paper | Your work | Why it's an improvement |
|---|---|---|---|
| Imbalance handling | Plain image augmentation | ADASYN | Targets the *hard* minority samples, not just bulk count |
| Ensemble composition | Same-family (VGG+VGG, DenseNet+DenseNet) | Heterogeneous paradigms (residual + compound-scaled + densely-connected) | Different design biases → less correlated errors → better ensemble gain |
| XAI methods | Grad-CAM + Saliency Map | Grad-CAM + SHAP + LIME | Adds quantitative feature-attribution on top of pure visual heatmaps |
| Backbone modernity | VGG (2014), DenseNet (2017) | ResNet/DenseNet/EfficientNet (2015–2019) | EfficientNet adds modern compound scaling |

### Checklist
- [ ] Read Mahmud et al. (2024) end-to-end and take notes on their exact metrics per class
- [ ] Read the ADASYN paper (He et al., 2008) — focus on Algorithm 1
- [ ] Read 2–3 recent (2023–2025) reviews on Alzheimer's MRI deep learning
- [ ] Draft a one-page "gap statement" comparing the table above
- [ ] Confirm OASIS dataset license is compatible with publication

---

## Phase 2 — Architecture Selection

**Estimated time: 2–3 days**

### The three backbones (locked)

| Model | Design paradigm | Why it suits MRI | Pretrained params | Input size |
|---|---|---|---|---|
| **ResNet50** | Residual learning (skip connections let gradients flow through 50 layers) | Strong general feature extractor; widely used in medical imaging baselines | ~25.6M | 224×224 |
| **EfficientNetB3** | Compound scaling (width/depth/resolution jointly scaled) + MBConv (depthwise-separable) blocks | Best accuracy-per-parameter; handles texture-rich MRI well | ~12M | 300×300 |
| **DenseNet121** | Dense connectivity (every layer sees all previous layers' features) | Excellent feature reuse, used in reference paper's sibling DenseNet169/201 | ~8M | 224×224 |

All three are available pretrained on ImageNet via `torchvision.models` and `timm`.

### Same-family vs heterogeneous-paradigm ensemble — analogy

A *same-family ensemble* is like asking three siblings raised in the same house to vote on a movie — they'll often agree because they share assumptions. A *heterogeneous ensemble* asks three people from different backgrounds — they're more likely to disagree, but when they *do* agree the verdict is much stronger. In ML terms: ensemble gain comes from *uncorrelated errors* between models. Different design paradigms make different mistakes.

### Ensemble combination strategy — pick: **Weighted soft voting**

Three options exist:
- **Hard voting** — each model picks one class, majority wins. Loses confidence info.
- **Soft voting (averaging)** — average the 4-class probability vectors, take argmax. Standard, simple, works well.
- **Stacking** — train a meta-classifier on the 3 models' outputs. More powerful but adds complexity and needs careful validation to avoid leakage.

**Decision: weighted soft voting.** Weights proportional to each model's validation F1-score (so a stronger model on the val set gets more say). It's the sweet spot between simplicity and performance. Stacking can be tried in ablation.

### Checklist
- [ ] Verify each model loads from `timm` with `pretrained=True`
- [ ] Sanity-check input pipeline for the two different input sizes (224 for ResNet/DenseNet, 300 for EffNet)
- [ ] Decide final classifier head: `GlobalAvgPool → Dropout(0.3) → Linear(features, 4)`
- [ ] Draft ensemble pseudocode (3 forward passes → softmax → weighted average → argmax)

---

## Phase 3 — Dataset & Preprocessing

**Estimated time: 3–4 days**

### Dataset
- **OASIS MRI** from Kaggle (search: "Alzheimer MRI Disease Classification Dataset" — same 6400-image split as Mahmud et al.)
- 4 classes, 2D axial slice JPGs, already pre-extracted from 3D MRI volumes

### Preprocessing pipeline (in order)

1. **Load and resize** — to 224×224 for ResNet/DenseNet, 300×300 for EfficientNet. Keep two parallel data tensors or use per-model transforms.
2. **Grayscale → 3-channel** — MRIs are grayscale; pretrained ImageNet models expect 3 channels. Duplicate the gray channel three times.
3. **Normalize** — use ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) because backbones are ImageNet-pretrained.
4. **Noise removal** — apply mild Gaussian blur (σ=0.5) or median filter (3×3) to suppress scanner noise. Optional; ablate it.
5. **Brain region focus (lightweight "segmentation")** — for a beginner, skip true skull-stripping and instead crop a tight bounding box around non-zero pixels. Full segmentation (e.g., FSL/BET) is out of scope unless you have time.
6. **Standard augmentation** — small rotations (±10°), horizontal flip, brightness jitter (±10%). Applied online during training, NOT to test set.

### ADASYN placement — critical decision

**Rule: ADASYN goes AFTER train/val/test split, applied ONLY to the training set.**

Why: if you ADASYN first then split, synthetic samples generated from a test image's neighbors leak into training → inflated test metrics → reviewer rejection.

**On what features do we apply ADASYN?** Two options:
- **Option A (recommended for beginners):** Flatten resized images (e.g., 64×64×1 = 4096 features), apply ADASYN in pixel space, reshape back. Fast, works, but synthetic samples may look slightly blurry.
- **Option B (more rigorous):** Pass training images through a pretrained CNN, extract the penultimate-layer embedding (e.g., 2048-dim vector for ResNet50), apply ADASYN in that embedding space, train a classifier head on the resulting balanced embeddings. Cleaner but requires a separate embedding-extraction pass.

**Decision: Option A for the main pipeline; mention Option B as an ablation.**

### Expected post-ADASYN training distribution

Starting (train ≈ 80% of 6400 = 5120):
- Non-Dementia: ~2560
- Very Mild: ~1792
- Mild: ~717
- Moderate: ~51

After ADASYN (target = match majority class):
- All 4 classes: ~2560 each → ~10,240 training samples total
- Test set stays imbalanced (it must reflect reality)

### Train/Val/Test split

**80 / 10 / 10 stratified split.** Stratify on class label so each split preserves the original class proportions. Use `sklearn.model_selection.train_test_split(stratify=y)`. Set `random_state=42` for reproducibility.

### Checklist
- [ ] Download OASIS dataset to Kaggle notebook
- [ ] Implement stratified 80/10/10 split → save indices to disk
- [ ] Build a PyTorch `Dataset` class that returns `(image_tensor, label)`
- [ ] Apply ADASYN from `imblearn.over_sampling.ADASYN` on training subset only
- [ ] Print class distribution before/after ADASYN — verify balance
- [ ] Save a few synthetic samples as PNGs to visually sanity-check

---

## Phase 4 — Model Training Plan

**Estimated time: 1–2 weeks (training is the slowest phase)**

### Transfer learning strategy (per model)

**Stage 1 — Feature extraction (3 epochs):** Freeze all backbone layers; train only the new classifier head. This warms up the head without disturbing pretrained features.

**Stage 2 — Fine-tuning (15–25 epochs):** Unfreeze the last 2 blocks of the backbone + the head. Train with a small learning rate (1e-4). Keep early layers frozen — they learn generic edges/textures that transfer well.

Analogy: stage 1 is like hiring a specialist consultant (the classifier head) to interpret reports written by general-purpose analysts (the frozen backbone). Stage 2 is letting the specialist coach the senior analysts to tweak their reports slightly for the specific domain (MRI).

### Hyperparameters (starting point)

| Hyperparameter | Value | Note |
|---|---|---|
| Optimizer | AdamW | Better than Adam for transfer learning (decoupled weight decay) |
| Initial LR (head only) | 1e-3 | Standard for new heads |
| Fine-tune LR (backbone) | 1e-4 | 10× smaller to avoid destroying pretrained features |
| LR scheduler | CosineAnnealingLR | Smooth decay, less tuning than step schedules |
| Batch size | 32 (16 if OOM on Kaggle T4) | EfficientNetB3 may need 16 due to 300×300 input |
| Epochs | 3 + 25 | Stage 1 + Stage 2 |
| Loss | CrossEntropyLoss | Standard for multi-class |
| Weight decay | 1e-4 | Mild regularization |
| Early stopping | Patience = 5 epochs on val F1 | Prevents overfitting |
| Mixed precision | `torch.cuda.amp` | ~2× speedup on T4 |

### Ensemble construction

After each of the 3 models is trained and saved:

```
For each test sample:
    p_resnet     = softmax(ResNet50(x))         # shape: (4,)
    p_effnet     = softmax(EfficientNetB3(x))   # shape: (4,)
    p_densenet   = softmax(DenseNet121(x))      # shape: (4,)
    p_ensemble   = w1*p_resnet + w2*p_effnet + w3*p_densenet
    prediction   = argmax(p_ensemble)
```

Where `w1, w2, w3` are normalized validation-set F1 scores (sum to 1).

### Checklist
- [ ] Implement a single `train_one_model(model_name, dataloaders, config)` function — reuse for all 3
- [ ] Log per-epoch: train loss, val loss, val accuracy, val F1, val per-class recall
- [ ] Save best checkpoint by val F1 (not val accuracy — F1 handles imbalance better)
- [ ] Train ResNet50 → log → save
- [ ] Train EfficientNetB3 → log → save
- [ ] Train DenseNet121 → log → save
- [ ] Implement ensemble inference loop using the 3 saved checkpoints
- [ ] Compute final ensemble metrics on test set

---

## Phase 5 — XAI Plan

**Estimated time: 4–6 days**

### What each XAI method shows

| Method | What it produces | What a doctor sees | What it answers |
|---|---|---|---|
| **Grad-CAM** | Heatmap overlaid on the input MRI showing which spatial regions most influenced the prediction | A colored blob on the brain image | "Where in the image did the model look?" |
| **SHAP** | Per-pixel (or per-superpixel) attribution values that sum to the prediction | A second image where each pixel has a positive/negative contribution score | "How much did each region push toward or away from this class?" |
| **LIME** | Highlighted superpixels (image segments) that most support the prediction | An image with a few clear regions outlined | "Which simple visual chunks would change the prediction if removed?" |

Doctors care because:
- Regulators (FDA, EMA) require *some* level of explainability for AI medical devices
- Doctors won't trust a black-box "Mild Dementia, 87% confident" — they need to see *why*
- Grad-CAM hitting known Alzheimer's-relevant regions (hippocampus, ventricles) builds clinical trust
- SHAP/LIME provide quantitative attributions that can be statistically analyzed across the test set

### Applying XAI to an ensemble (not just one model)

This is a real subtlety the reference paper sidesteps. Options:

- **Per-model XAI then average heatmaps** — run Grad-CAM on each of the 3 backbones, then pixel-wise average the 3 normalized heatmaps. Recommended. Reflects what the ensemble "saw" collectively.
- **Pick best single model for XAI** — easier but loses the ensemble story.
- **Wrap the ensemble as a single callable for SHAP/LIME** — SHAP and LIME just need a `predict(x) → probs` function; pass it the ensemble's combined probability output. This works cleanly.

**Decision:** averaged Grad-CAM heatmaps across the 3 backbones; ensemble-as-predict-function for SHAP and LIME.

### Implementation libraries
- **Grad-CAM**: `pytorch-grad-cam` (Jacob Gildenblat)
- **SHAP**: `shap` library, `DeepExplainer` or `GradientExplainer`
- **LIME**: `lime` library, `LimeImageExplainer`

### Checklist
- [ ] Generate Grad-CAM for 5 correct + 5 incorrect predictions per class (40 total) per model
- [ ] Average heatmaps across the 3 models → ensemble Grad-CAM
- [ ] Run SHAP on the ensemble for ~20 representative test images (SHAP is slow)
- [ ] Run LIME on the same ~20 test images for direct comparison
- [ ] Produce side-by-side figures: original MRI | Grad-CAM | SHAP | LIME — these go in the paper
- [ ] Have a clinician (advisor) sanity-check whether highlighted regions match known Alzheimer's pathology

---

## Phase 6 — Evaluation Plan

**Estimated time: 3–4 days**

### Metrics and why each matters in medical imaging

| Metric | What it measures | Why it matters here |
|---|---|---|
| **Accuracy** | % correctly classified | Easy to communicate but misleading on imbalanced data — high accuracy by always predicting "Non-Dementia" |
| **Precision (per class)** | Of all "Mild Dementia" predictions, how many were actually Mild? | Low precision = lots of false alarms → unnecessary patient anxiety |
| **Recall / Sensitivity (per class)** | Of all actual Mild Dementia patients, how many did we catch? | **Most critical metric.** Missing a dementia case is far worse than a false alarm — patient goes undiagnosed |
| **F1-Score (per class + macro)** | Harmonic mean of precision and recall | Single number that balances both. Macro-F1 (unweighted average across 4 classes) is the right "headline" metric for imbalanced medical data |
| **AUC-ROC (one-vs-rest)** | How well the model ranks positives above negatives, threshold-independent | Clinically useful because doctors may adjust the decision threshold for screening vs. diagnosis |

**Headline metric for the paper: Macro-F1 + per-class Recall + Confusion Matrix.** Accuracy reported but de-emphasized.

### Confusion matrix interpretation (4-class)

A 4×4 matrix. Rows = true class, columns = predicted class. Diagonal = correct. Off-diagonal patterns to look for:
- **Adjacent confusions** (Mild ↔ Very Mild, Moderate ↔ Mild) — *expected*, these are clinically similar
- **Distant confusions** (Moderate predicted as Non-Dementia) — *dangerous*, indicates real failure
- **Moderate row should be the weakest** — only 64 original samples; even with ADASYN this class will be hardest. Pay extra attention to its recall.

### Comparison table (paper Table 4 or similar)

| Method | Accuracy | Macro-F1 | Recall (Moderate) | AUC-ROC | XAI |
|---|---|---|---|---|---|
| Mahmud et al. — VGG16 | 90% | — | — | — | Grad-CAM, Saliency |
| Mahmud et al. — DenseNet201 | 93% | — | — | — | Grad-CAM, Saliency |
| Mahmud et al. — Ensemble-2 (best) | 95% | — | — | — | Grad-CAM, Saliency |
| Mahmud et al. — Proposed | 96% | — | — | — | Grad-CAM, Saliency |
| **Ours — ResNet50 alone** | TBD | TBD | TBD | TBD | Grad-CAM+SHAP+LIME |
| **Ours — EfficientNetB3 alone** | TBD | TBD | TBD | TBD | Grad-CAM+SHAP+LIME |
| **Ours — DenseNet121 alone** | TBD | TBD | TBD | TBD | Grad-CAM+SHAP+LIME |
| **Ours — Heterogeneous Ensemble (proposed)** | TBD | TBD | TBD | TBD | Grad-CAM+SHAP+LIME |

Also run two **ablation studies**:
1. *Without ADASYN* (plain augmentation only) — quantifies ADASYN's contribution
2. *Same-family ensemble* (3 ResNets or 3 DenseNets) — quantifies heterogeneity's contribution

### Checklist
- [ ] Implement `evaluate(model_or_ensemble, test_loader)` → returns dict of all metrics
- [ ] Generate confusion matrix figure (use `seaborn.heatmap`)
- [ ] Generate ROC curve figure (one curve per class, one-vs-rest)
- [ ] Run the two ablation studies
- [ ] Fill in the comparison table
- [ ] Run McNemar's test or paired bootstrap to claim *statistically significant* improvement over single best model

---

## Phase 7 — Paper Writing Structure

**Estimated time: 3–5 weeks (writing + revisions)**

### Section-by-section structure

**1. Abstract (~250 words)** — Problem, dataset, your 3 pillars, headline result (e.g., "97% accuracy, +1% over Mahmud et al."), one sentence on clinical implication.

**2. Introduction (1.5–2 pages)**
- Paragraph 1: Alzheimer's epidemiology + societal burden
- Paragraph 2: Why early diagnosis matters + role of MRI
- Paragraph 3: Why ML/DL is promising + open challenges (imbalance, ensemble design, explainability)
- Paragraph 4: Your three contributions (bullet them: ADASYN, heterogeneous-paradigm ensemble, triple-XAI)
- Paragraph 5: Paper roadmap

**3. Related Work (2–3 pages)**
- 3.1 Deep learning for Alzheimer's MRI (cite 8–12 recent papers)
- 3.2 Class imbalance in medical imaging (SMOTE, ADASYN, augmentation)
- 3.3 Ensemble methods in medical image classification
- 3.4 Explainable AI in medical imaging
- End with the gap table from Phase 1

**4. Methodology (3–4 pages)**
- 4.1 Dataset (OASIS description, splits)
- 4.2 Preprocessing pipeline (figure showing each step)
- 4.3 ADASYN (algorithm summary + your application choices)
- 4.4 Three backbones (one paragraph each + architecture figure)
- 4.5 Ensemble fusion (weighted soft voting formula)
- 4.6 XAI methods (Grad-CAM, SHAP, LIME with formulas)
- 4.7 Training protocol (hyperparameters table)

**5. Results (3–4 pages)**
- 5.1 Individual model results (table)
- 5.2 Ensemble results (table)
- 5.3 Ablation studies (without ADASYN, same-family ensemble)
- 5.4 Confusion matrices and ROC curves (figures)
- 5.5 XAI visualizations (4-column figure: image | Grad-CAM | SHAP | LIME)
- 5.6 Comparison with Mahmud et al. and other state-of-the-art

**6. Discussion (1.5–2 pages)**
- Why heterogeneous-paradigm ensemble helps (cite the uncorrelated-error theory)
- Clinical implications of the XAI outputs
- Limitations: all backbones are CNNs (acknowledge openly), 2D slices not 3D volumes, single dataset
- Threats to validity

**7. Conclusion + Future Work (~0.5 page)** — Summarize contributions, list future work (CNN + Transformer ensemble, 3D volumetric input, multi-site validation, prospective clinical study).

**8. References** — Aim for 35–55 references. Use Zotero or Mendeley.

### Positioning your novelty (the critical paragraph)

In the introduction and again in discussion, write something like:

> "Unlike Mahmud et al. (2024), who combined models from the same architectural family (e.g., VGG16 + VGG19), our ensemble integrates three CNN backbones built on fundamentally different design paradigms — residual learning, compound scaling, and dense connectivity — which we hypothesize produces more decorrelated errors and thus stronger ensemble gain. We further replace plain image augmentation with ADASYN, which specifically targets borderline minority-class samples, and extend explainability beyond Grad-CAM by adding SHAP and LIME for quantitative feature attribution."

### Suggested journals (open-access, medical AI focus)

| Journal | Publisher | Impact factor (approx.) | Notes |
|---|---|---|---|
| **Diagnostics** (MDPI) | MDPI | ~3.6 | Same journal as reference paper — natural fit |
| **Computers in Biology and Medicine** | Elsevier | ~7.0 | Higher prestige, longer review |
| **IEEE Journal of Biomedical and Health Informatics** | IEEE | ~7.7 | Strong methodology bar |
| **Scientific Reports** | Nature | ~4.0 | Broad, fast turnaround |
| **PeerJ Computer Science** | PeerJ | ~3.5 | Open-access, beginner-friendly |
| **Frontiers in Aging Neuroscience** | Frontiers | ~4.8 | Clinical angle |

**Recommendation:** Submit to **Diagnostics** first — same venue as the reference paper, so reviewers will already be familiar with the comparison. Fallback to **Scientific Reports** or **PeerJ CS**.

### Checklist
- [ ] Create a LaTeX project (Overleaf) with the target journal's template
- [ ] Draft Introduction first, then Methodology, then Results
- [ ] Generate all figures with consistent style (matplotlib `seaborn` theme)
- [ ] Have advisor + 2 peers review before submission
- [ ] Run plagiarism check (Turnitin or iThenticate via your institution)

---

## Phase 8 — Tools & Environment

**Estimated time: 1–2 days setup**

### Why PyTorch (locked decision)

- More flexible for combining heterogeneous models in a single inference loop
- `timm` library gives one-line access to ResNet, EfficientNet, DenseNet (and Transformers if you later add them)
- Cleaner integration with `pytorch-grad-cam`, `shap`, `captum`
- Reference paper used TF/Keras — using PyTorch is also a small methodological differentiation

### Required Python libraries

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep learning framework |
| `timm` | Pretrained backbones (ResNet50, EfficientNetB3, DenseNet121) |
| `numpy`, `pandas` | Numerical arrays + tabular logging |
| `scikit-learn` | Train/test split, metrics, confusion matrix |
| `imbalanced-learn` (`imblearn`) | ADASYN implementation |
| `matplotlib`, `seaborn` | Plots, confusion matrices, ROC curves |
| `Pillow`, `opencv-python` | Image loading and preprocessing |
| `albumentations` | Fast image augmentation pipelines |
| `pytorch-grad-cam` | Grad-CAM visualization |
| `shap` | SHAP feature attribution |
| `lime` | LIME local explanations |
| `tqdm` | Progress bars |
| `tensorboard` or `wandb` | Training run tracking (wandb has a free tier and is more polished) |

### Compute setup (Kaggle)

- Create a Kaggle account, enable phone verification → unlocks GPU
- New notebook → Settings → Accelerator: **GPU T4 ×2** (or P100)
- Add the OASIS dataset to the notebook as an *input dataset*
- 30 hrs/week GPU quota — budget ~12 hrs for training (3 models × ~3 hrs each + ensemble + XAI)
- Save checkpoints to `/kaggle/working/` then download

### Project folder structure

```
alzheimer-ensemble/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # hyperparameters
├── data/
│   ├── raw/                      # original OASIS (gitignored)
│   ├── splits/                   # train/val/test indices
│   └── adasyn_cache/             # post-ADASYN training tensors
├── src/
│   ├── data.py                   # Dataset class, transforms
│   ├── adasyn_pipeline.py        # ADASYN application
│   ├── models.py                 # model factory (ResNet/EffNet/DenseNet)
│   ├── train.py                  # training loop
│   ├── ensemble.py               # weighted soft voting
│   ├── evaluate.py               # metrics, confusion matrix
│   └── xai/
│       ├── gradcam.py
│       ├── shap_explain.py
│       └── lime_explain.py
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_adasyn_sanity_check.ipynb
│   ├── 03_train_resnet50.ipynb
│   ├── 04_train_effnetb3.ipynb
│   ├── 05_train_densenet121.ipynb
│   ├── 06_ensemble_eval.ipynb
│   └── 07_xai_figures.ipynb
├── checkpoints/                  # saved model weights (gitignored)
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
└── paper/                        # LaTeX source (could be Overleaf-only)
```

### Checklist
- [ ] Create Kaggle account, verify phone, enable GPU
- [ ] Create one master Kaggle notebook with all dependencies installed in cell 1
- [ ] Set up local Git repo + push to GitHub (private until paper accepted)
- [ ] Pin all library versions in `requirements.txt`
- [ ] Set seeds everywhere (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`)

---

## Verification — How to know each phase worked

| Phase | Verification |
|---|---|
| 1 | Can write a 1-page gap statement from memory |
| 2 | All 3 models load via `timm.create_model(name, pretrained=True, num_classes=4)` without error |
| 3 | Class distribution post-ADASYN is roughly equal across 4 classes on training set only; test set retains original imbalance |
| 4 | All 3 models reach >85% val accuracy; ensemble val F1 > best single-model val F1 |
| 5 | Grad-CAM heatmaps consistently highlight medial temporal / hippocampal regions on dementia-positive cases (sanity check with a clinician) |
| 6 | Macro-F1 ≥ 0.93 and Moderate-class recall ≥ 0.80 on test set; ablation without ADASYN shows ≥3pp drop in Moderate recall |
| 7 | Paper passes a "stranger read" — give it to someone outside your field; they can summarize your three contributions in one sentence |
| 8 | Full pipeline runs end-to-end from `notebooks/01_*` through `notebooks/07_*` on a fresh Kaggle notebook |

## Total timeline estimate

| Phase | Time |
|---|---|
| 1. Foundation | 4–6 days |
| 2. Architecture | 2–3 days |
| 3. Data & preprocessing | 3–4 days |
| 4. Training | 1–2 weeks |
| 5. XAI | 4–6 days |
| 6. Evaluation | 3–4 days |
| 7. Paper writing | 3–5 weeks |
| 8. Tools setup | 1–2 days (front-loaded, in parallel with Phase 1) |
| **Total** | **~10–14 weeks part-time** |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reviewer attacks "cross-architecture" framing | Already mitigated — title says "heterogeneous-paradigm CNN"; acknowledge CNN-only limit in Discussion + Future Work |
| Moderate-class recall stays low even with ADASYN | Try class-weighted loss in addition to ADASYN; try focal loss; expand to OASIS-2 or ADNI if needed |
| Kaggle session timeouts during training | Save checkpoint every epoch; resume from last checkpoint; run shorter epochs initially |
| ADASYN on pixel space produces noisy synthetic samples | Switch to embedding-space ADASYN (Option B from Phase 3) |
| Three-model ensemble overfits 6400 images | Heavier dropout (0.5), stronger augmentation, k-fold cross-validation |
| SHAP runs out of memory on full test set | Sub-sample 20 representative test images for SHAP; full Grad-CAM coverage compensates |
