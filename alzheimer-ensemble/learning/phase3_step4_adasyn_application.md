# Phase 3 · Step 4 — Applying ADASYN (Training Set Only)

> **Phase 3 — Dataset & Preprocessing**
> The goal of this step: actually run ADASYN — on the right data, in the right space, at the right moment — to balance the training set.

The one question this step answers:

> **How and where do I apply ADASYN so it helps the minority class without leaking into evaluation?**

---

## 1. The rule, restated (it's that important)

> **ADASYN runs AFTER the split (Step 3), on the TRAINING SET ONLY.** Val and test stay untouched and imbalanced.

You learned *why* in Phase 1 Step 2 (leakage inflates test scores → reviewer rejection). Now you enforce it: feed ADASYN only the training indices saved in `data/splits/`.

---

## 2. On *what* do you run ADASYN? (images aren't vectors)

ADASYN needs feature vectors. Two placements (Phase 1 Step 2 preview, now decided):

- **Option A — pixel space (main pipeline).** Resize training images small (e.g. **64×64×1 = 4,096** features), flatten to vectors, run `imblearn.over_sampling.ADASYN`, reshape synthetic vectors back to images. Fast, simple, works. Synthetic samples may look slightly blurry — acceptable.
- **Option B — embedding space (ablation).** Push training images through a pretrained CNN, take the penultimate embedding (e.g. 2,048-dim), run ADASYN there, train a head on the balanced embeddings. Cleaner but needs a separate extraction pass.

**Decision: Option A for the main pipeline; Option B reported as an ablation** in Phase 6.

> Why flatten to 64×64 and not full resolution? ADASYN's nearest-neighbor search is expensive in very high dimensions, and you only need the synthetic *content*; you reshape and the training transforms (resize to 224/300) take over from there.

---

## 3. The target distribution

ADASYN's default target is "match the majority class." Starting from the ~train counts in Step 3:

| Class | Before ADASYN | After ADASYN (target ≈ majority) |
|---|---|---|
| Non-Dementia | ~2,560 | ~2,560 |
| Very Mild | ~1,792 | ~2,560 |
| Mild | ~717 | ~2,560 |
| Moderate | ~51 | ~2,560 |
| **Total** | ~5,120 | **~10,240** |

All four classes land near ~2,560 → ~10,240 balanced training samples. **Test set stays imbalanced** (it must reflect reality).

> ADASYN won't always hit *exactly* equal counts (it depends on the difficulty weighting `rᵢ`) — "roughly balanced" is expected and fine. You can also pass `sampling_strategy` to control targets.

---

## 4. Where it lives in code

`src/adasyn_pipeline.py`:
1. Load training images (per saved indices), resize→flatten to vectors.
2. `X_res, y_res = ADASYN(random_state=42).fit_resample(X, y)`.
3. Reshape `X_res` back to images; cache to `data/adasyn_cache/` so you don't recompute every run.
4. Downstream training reads the **balanced cache** for train; val/test read the original images.

A subtle gotcha: `Moderate ~51` samples means ADASYN's `n_neighbors` (default 5) must be ≤ available minority neighbors. If it errors on the tiny class, lower `n_neighbors`.

### Resources (a few hours total)
- `imblearn.over_sampling.ADASYN` docs — `sampling_strategy`, `n_neighbors`, `random_state`.
- The toy 2D `imblearn` plotting example — confirm your mental model of "more samples near hard points."

---

## 5. Your task (verifies Step 4 — do it before moving on)

- [ ] **1.** Implement Option A: flatten **training-only** images → `ADASYN.fit_resample` → reshape → cache.
- [ ] **2.** Handle the tiny-Moderate edge case (adjust `n_neighbors` if needed) and set `random_state=42`.
- [ ] **3.** Confirm **val/test were not passed to ADASYN** (re-read your code — this is the trap).
- [ ] **4.** Leave a stub/plan for Option B (embedding-space) as the ablation.

✅ When ADASYN runs on train only and caches balanced images, move on to verify it.

## When you're ready

> **Phase 3 · Step 5 — Verify the balance and eyeball the synthetic samples.**
