# Phase 8 · Step 1 — Kaggle Setup

> **Phase 8 — Tools & Environment** (front-loaded — do this in parallel with Phase 1, not at the end)
> The goal of this step: get a working GPU environment with the OASIS dataset attached, so you can train when you reach Phase 4.

The one question this step answers:

> **Where does my code run, and how do I get free GPU + the dataset in one place?**

---

## 1. Why Kaggle (locked decision)

- **Free GPU** (T4 ×2 or P100) — enough for this project's three models.
- **OASIS is already hosted** there — no large download/upload.
- 30 GPU-hrs/week quota — plenty for the ~12-hr training budget.

---

## 2. The setup steps

1. **Create a Kaggle account**, then **verify your phone number** — this unlocks GPU access (a common gotcha; no verification → no GPU).
2. **New Notebook → Settings → Accelerator → GPU T4 ×2** (or P100).
3. **Add the OASIS dataset** as an *input dataset* (search the "Alzheimer MRI Disease Classification Dataset"). It mounts read-only under `/kaggle/input/`.
4. **Save outputs** (checkpoints, figures) to `/kaggle/working/`, then download or version them — `/kaggle/input/` is read-only.
5. **Install dependencies in cell 1** from your `requirements.txt` (pin versions so runs are reproducible).

---

## 3. Budget the GPU quota

~**12 GPU-hrs** total: ≈3 hrs each for ResNet50 / EfficientNetB3 / DenseNet121, plus ensemble inference + XAI. EffNetB3 (300×300) is the heaviest — give it headroom. Survive timeouts by **checkpointing every epoch** and supporting **resume** (Phase 4 Step 3).

> Practical tip: develop on a tiny subset with the GPU *off* (saves quota), switch the GPU *on* only for real training runs.

### Resources (a few hours total)
- Kaggle Notebooks docs: accelerators, input datasets, output persistence, weekly quota.
- The OASIS dataset's Kaggle page (note its license/citation — cross-check with Phase 1 Step 5).

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Create + phone-verify a Kaggle account; confirm GPU is selectable.
- [ ] **2.** Spin up a notebook with **GPU on** and run `torch.cuda.is_available()` → `True`.
- [ ] **3.** Attach the OASIS dataset; confirm it appears under `/kaggle/input/` with the expected class folders.
- [ ] **4.** Install `requirements.txt` in cell 1 and import `torch`, `timm`, `imblearn` without error.

✅ When a GPU notebook sees both CUDA and the dataset, move on to scaffold the project.

## When you're ready

> **Phase 8 · Step 2 — Project scaffolding: folders, Git, pinned requirements, seeds.**
