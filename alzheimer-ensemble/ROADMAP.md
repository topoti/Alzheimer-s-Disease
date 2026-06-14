# Execution Roadmap — Cross-Paradigm CNN Ensemble for Alzheimer's Classification

## Context

`RESEARCH_PLAN.md` is a complete 8-phase academic research plan
(improving on Mahmud et al. 2024 with ADASYN + a heterogeneous-paradigm CNN ensemble +
triple-XAI). The *thinking* is done; the *execution* is mostly not. This roadmap converts
the plan into an ordered, buildable engineering sequence given what already exists.

**Already complete** (do not rebuild):
- `RESEARCH_PLAN.md` + `learning/` (33-step curriculum) — Phases 1–2 foundation/design.
- `src/data.py` — patient-grouped stratified 80/10/10 split, transforms (gray→3ch, ImageNet
  norm, light aug), `OASISDataset`, `build_dataloaders`, `make_weighted_sampler`.
- `data/splits/{slice_index,subject_split}.csv` — split persisted (86,437 slices / 347
  subjects; Moderate = 2 subjects → 1 train / 1 test).
- `configs/default.yaml`, `requirements.txt`.

**Decisions for this roadmap:** compute is **Kaggle-only** — the local repo is
source-of-truth for `src/`, pushed to Kaggle as a utility dataset/notebook; all heavy compute
runs on Kaggle GPU.

**Two issues to fix early:**
1. `configs/default.yaml` sets `adasyn.apply_on: pixel_space`, but RESEARCH_PLAN makes
   **embedding-space ADASYN the main pipeline** (pixel-space is an explicitly *rejected*
   baseline). → Change default to `embedding_space`; keep `pixel_space` switchable for the
   ablation only.
2. The committed split CSVs hold **local absolute Windows filepaths**, unusable on Kaggle.
   → Rebuild the split on Kaggle (`python -m src.data --data-root /kaggle/input/...`); the
   subject→split assignment is deterministic (`seed=42`) so it reproduces the same partition.
   Store paths as **relative-to-data-root** if practical to make CSVs portable.

---

## Milestone 0 — Kaggle environment & repo wiring  *(Phase 8)*
- Create master Kaggle notebook; Accelerator = GPU T4×2 (or P100); attach
  `ninadaithal/imagesoasis` as input.
- Cell 1 installs pinned `requirements.txt`; set all seeds (`torch`, `numpy`, `random` = 42).
- Make `src/` importable on Kaggle (push repo as a Kaggle *dataset*, or `git clone`).
- **Verify:** `import src.data` works; `python -m src.data --data-root /kaggle/input/imagesoasis/Data`
  rebuilds the split and prints the subject-disjoint summary.

## Milestone 1 — Models & ADASYN pipeline  *(Phase 2 §head, Phase 3 §ADASYN)*
- `src/models.py` — factory `create_model(name, num_classes=4, pretrained=True)` via `timm`
  for `resnet50`, `efficientnet_b3`, `densenet121`; head = `GlobalAvgPool→Dropout(0.3)→Linear`.
  Helpers to freeze/unfreeze backbone and to expose a **penultimate-layer feature extractor**
  (for embedding-space ADASYN and SHAP/Grad-CAM target layers).
- `src/adasyn_pipeline.py` — **embedding-space** path (main): extract frozen-backbone
  embeddings for the **train split only**, run `imblearn.over_sampling.ADASYN` with a
  **capped** `sampling_strategy` (ablate 25%/50%/full), cache balanced embeddings to
  `data/adasyn_cache/`. Keep a `pixel_space` branch for the rejected-baseline ablation.
- **Verify:** all 3 models load via `timm.create_model(..., num_classes=4)`; print class
  distribution before/after ADASYN (cap honored, val/test untouched); nearest-real-image
  sanity check on a few synthetic embeddings.

## Milestone 2 — Training loop & 3 trained checkpoints  *(Phase 4)*
- `src/train.py` — one reusable `train_one_model(model_name, dataloaders, config)`:
  Stage 1 (freeze backbone, train head, ~3 ep) → Stage 2 (unfreeze last 2 blocks, fine-tune
  ~15–25 ep). AdamW, cosine LR, AMP, **class-balanced/focal loss**, early stopping on val
  **macro-F1**, per-epoch logging (loss/acc/F1/per-class recall), checkpoint best-by-val-F1.
  Pick **one** primary imbalance corrector (don't stack heavy weighted-loss + sampler +
  full oversampling). Checkpoint-resume to survive Kaggle session limits.
- Train ResNet50 → EfficientNetB3 (batch 16, 300×300) → DenseNet121; save to `checkpoints/`.
- **Verify:** each model >85% val accuracy; checkpoints reload and reproduce val metrics.

## Milestone 3 — Ensemble inference  *(Phase 4 §ensemble)*
- `src/ensemble.py` — load 3 checkpoints, per-sample softmax → **weighted soft voting**
  (weights = normalized val macro-F1) → argmax. Expose `predict(x)→probs` callable (reused by
  SHAP/LIME). Persist test predictions + probabilities for downstream reuse.
- **Verify:** ensemble val macro-F1 > best single model's val macro-F1.

## Milestone 4 — Evaluation & ablations  *(Phase 6)*
- `src/evaluate.py` — `evaluate(model_or_ensemble, test_loader)` → accuracy, macro-F1,
  per-class precision/recall, balanced accuracy, AUC-ROC + PR-AUC (OvR); confusion-matrix and
  ROC figures (`seaborn`). McNemar / paired bootstrap vs best single model.
- Run both **ablations**: (a) no-ADASYN (plain aug only), (b) same-family ensemble.
- **Verify:** macro-F1 ≥ 0.93 and Moderate recall ≥ 0.80 on test; no-ADASYN ablation shows
  ≥3pp Moderate-recall drop. (Treat Moderate as a single-held-out-patient probe, per plan.)

## Milestone 5 — Triple-XAI figures  *(Phase 5)*
- `src/xai/gradcam.py` (per-model + averaged ensemble heatmaps; target layers already in
  config), `shap_explain.py` (GradientExplainer on the ensemble `predict`), `lime_explain.py`
  (LimeImageExplainer). Produce 4-column figures: MRI | Grad-CAM | SHAP | LIME.
- **Verify:** Grad-CAM concentrates on medial-temporal / hippocampal / ventricular regions on
  dementia-positive cases; SHAP/LIME run on the ~20-image representative subset.

## Milestone 6 — Notebooks, results, paper  *(Phase 7)*
- Thin orchestration notebooks `01_dataset_eda` … `07_xai_figures` that call `src/` (logic
  lives in `src/`, notebooks only drive + visualize). Collect `results/{figures,tables,logs}`.
- Fill the comparison table vs Mahmud et al.; draft the paper per Phase 7 structure
  (Diagnostics target). Framing must say **"heterogeneous-paradigm / cross-paradigm CNN"**,
  never "cross-architecture"; acknowledge CNN-only + 2-subject-Moderate limitations.

---

## Critical files to create (mirror RESEARCH_PLAN structure)
`src/models.py`, `src/adasyn_pipeline.py`, `src/train.py`, `src/ensemble.py`,
`src/evaluate.py`, `src/xai/{gradcam,shap_explain,lime_explain}.py`, `notebooks/01..07`,
outputs in `checkpoints/`, `data/adasyn_cache/`, `results/`. Reuse the existing
`src/data.py` API (`load_splits`, `build_dataloaders`, `make_weighted_sampler`) everywhere —
do not re-implement splitting or transforms.

## Suggested dependency order
M0 → M1 → M2 → M3 → (M4 ∥ M5) → M6. M4 and M5 both consume M3's saved test predictions and
can proceed in parallel.

## End-to-end verification
A fresh Kaggle notebook runs `01_*`→`07_*` top-to-bottom: rebuilds the subject-disjoint split,
caches capped embedding-space ADASYN (train-only), trains/loads 3 checkpoints, ensembles,
emits the metrics suite + confusion/ROC figures + the 4-column XAI panels — matching the
Phase-by-phase verification table in RESEARCH_PLAN.md.
