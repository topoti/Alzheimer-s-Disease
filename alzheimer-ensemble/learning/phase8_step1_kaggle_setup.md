# Phase 8 · Step 1 — Kaggle Setup

> **Phase 8 — Tools & Environment** (front-loaded — do this in parallel with Phase 1, not at the end)
> The goal of this step: get a working GPU environment with the `ninadaithal/imagesoasis` OASIS dataset attached, so you can train when you reach Phase 4.

The one question this step answers:

> **Where does my code run, and how do I get free GPU + the dataset in one place?**

---

## 1. Why Kaggle (locked decision)

- **Free GPU** (T4 ×2 or P100) — enough for this project's three models.
- **OASIS is already hosted** there (`ninadaithal/imagesoasis`, ~86k images) — no large download/upload.
- 30 GPU-hrs/week quota — workable, but this dataset is large, so budget carefully (see §3).

---

## 2. The setup steps

1. **Create a Kaggle account**, then **verify your phone number** — this unlocks GPU access (a common gotcha; no verification → no GPU).
2. **New Notebook → Settings → Accelerator → GPU T4 ×2** (or P100).
3. **Add the OASIS dataset** as an *input dataset* (search **`ninadaithal/imagesoasis`** / "OASIS Alzheimer's Detection"). It mounts read-only under `/kaggle/input/`.
4. **Save outputs** (checkpoints, figures) to `/kaggle/working/`, then download or version them — `/kaggle/input/` is read-only.
5. **Install dependencies in cell 1** from your `requirements.txt` (pin versions so runs are reproducible).

---

## 3. Budget the GPU quota (this dataset is large)

With ~86k images — and a **balanced training set that can reach ~200k** after ADASYN (Phase 3 Step 4) — epochs are far heavier than on the reference paper's 6.4k-image set, and the old "~3 hrs per model" estimate no longer holds. Plan to stay inside 30 GPU-hrs/week with these levers:

- **Two-stage transfer learning** (Phase 4 Step 1): the frozen-backbone stage is cheap; only the short fine-tune stage is expensive.
- **Fewer epochs + early stopping** on val macro-F1 — don't over-train.
- **Cap the ADASYN target** (`sampling_strategy`) or train on a **stratified subset** rather than balancing all the way to the ~53,800 majority (Phase 3 Step 4 scale note).
- **Checkpoint every epoch + resume** so a session timeout never loses progress, and training can span multiple sessions/weeks if needed (Phase 4 Step 3).
- EffNetB3 (300×300) is the heaviest — give it the most headroom.

> Practical tip: develop on a tiny subset with the GPU *off* (saves quota), switch the GPU *on* only for real training runs.

### Resources (a few hours total)
- Kaggle Notebooks docs: accelerators, input datasets, output persistence, weekly quota.
- The `ninadaithal/imagesoasis` Kaggle page (note its license/citation, and cite OASIS-1 itself — cross-check with Phase 1 Step 5).

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Create + phone-verify a Kaggle account; confirm GPU is selectable.
- [ ] **2.** Spin up a notebook with **GPU on** and run `torch.cuda.is_available()` → `True`.
- [ ] **3.** Attach `ninadaithal/imagesoasis`; confirm it appears under `/kaggle/input/` with the four class folders (Non Demented, Very mild Dementia, Mild Dementia, Moderate Dementia).
- [ ] **4.** Install `requirements.txt` in cell 1 and import `torch`, `timm`, `imblearn` without error.

✅ When a GPU notebook sees both CUDA and the dataset, move on to scaffold the project.

## When you're ready

> **Phase 8 · Step 2 — Project scaffolding: folders, Git, pinned requirements, seeds.**
