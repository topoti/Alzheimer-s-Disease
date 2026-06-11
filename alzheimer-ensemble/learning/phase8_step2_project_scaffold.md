# Phase 8 · Step 2 — Project Scaffolding: Folders, Git, Requirements, Seeds

> **Phase 8 — Tools & Environment** (front-loaded; closes the curriculum)
> The goal of this step: set up a clean, reproducible repository so every later phase has a home and every run is repeatable.

The one question this step answers:

> **How do I structure the project so it stays organized and reproducible from day one?**

---

## 1. The folder structure (the home for everything you'll build)

```
alzheimer-ensemble/
├── README.md
├── requirements.txt
├── configs/  default.yaml            # hyperparameters
├── data/
│   ├── raw/                          # original OASIS (gitignored)
│   ├── splits/                       # train/val/test indices  (Phase 3 Step 3)
│   └── adasyn_cache/                 # post-ADASYN training tensors (Phase 3 Step 4)
├── src/
│   ├── data.py                       # Dataset + transforms (Phase 3 Steps 2–3)
│   ├── adasyn_pipeline.py            # ADASYN application (Phase 3 Step 4)
│   ├── models.py                     # model factory (Phase 2 Step 3)
│   ├── train.py                      # train_one_model (Phase 4 Step 2)
│   ├── ensemble.py                   # weighted soft voting (Phase 4 Step 4)
│   ├── evaluate.py                   # metrics (Phase 6 Step 1)
│   └── xai/  gradcam.py  shap_explain.py  lime_explain.py   # Phase 5
├── notebooks/  01_dataset_eda … 07_xai_figures.ipynb
├── checkpoints/                      # saved weights (gitignored)
├── results/  figures/  tables/  logs/
└── paper/                            # LaTeX source
```

Each `src/` file maps to a phase you've already studied — the curriculum and the codebase are the same skeleton.

---

## 2. Git + GitHub

- Initialize a Git repo; push to **GitHub, private** until the paper is accepted.
- A `.gitignore` for `data/raw/`, `checkpoints/`, `adasyn_cache/`, and large artifacts (never commit the dataset or weights).
- Commit per logical step; meaningful messages. (This repo already exists and is on `main` — just keep the discipline.)

---

## 3. Pin everything (reproducibility, part 1)

`requirements.txt` already pins exact versions (torch 2.3.1, timm 1.0.7, imbalanced-learn 0.12.3, grad-cam 1.5.0, shap 0.46.0, lime 0.2.0.1, …). Keep it authoritative — install from it on Kaggle in cell 1 so local and cloud match.

---

## 4. Seeds everywhere (reproducibility, part 2)

At the top of every entry point:

```
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# optionally: torch.backends.cudnn.deterministic = True
```

Plus `random_state=42` in the split (Phase 3 Step 3) and ADASYN (Phase 3 Step 4). Reproducibility is what lets a reviewer (and future-you) trust the numbers.

### Resources (a few hours total)
- A `.gitignore` template for Python/ML projects.
- PyTorch "reproducibility" docs (seeds + cuDNN determinism caveats).
- YAML config patterns (`configs/default.yaml` already exists — keep hyperparameters there, not hardcoded).

---

## 5. Your task (verifies Step 2 — and closes the curriculum)

- [ ] **1.** Confirm the folder structure exists (create any missing dirs); each `src/` stub maps to its phase.
- [ ] **2.** Verify `.gitignore` excludes `data/raw/`, `checkpoints/`, `adasyn_cache/`.
- [ ] **3.** Add a `set_seeds()` helper and call it at the top of every notebook/script.
- [ ] **4.** Confirm `requirements.txt` installs cleanly both locally and on Kaggle.

✅ Phase 8 verification (plan): the full pipeline can run end-to-end from `notebooks/01_*` through `07_*` on a fresh Kaggle notebook. When the scaffold supports that, your environment is ready — and you've completed the whole curriculum.

## Curriculum complete

You've walked all 8 phases. Loop back to **`phase1_step1_problem_framing.md`** any time you need to re-anchor on *why* the project is built the way it is — every decision traces back to the ~138× class imbalance.
