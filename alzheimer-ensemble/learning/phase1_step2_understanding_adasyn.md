# Phase 1 · Step 2 — Understand ADASYN (Pillar 1)

> **Phase 1 — Research Foundation** (still no code — pure understanding)
> The goal of this step: understand *why* ADASYN beats plain augmentation, deeply enough to defend it to a reviewer — because ADASYN is the first of your three contributions.

The one question this step answers:

> **How do we fix the 50× class imbalance from Step 1, and why is ADASYN smarter than the alternatives?**

---

## 1. Three ways to fix imbalance — a ladder from dumb to smart

In Step 1 you learned the villain: one class (Moderate, 64 images) is drowned out by another (Non-Dementia, 3,200). There are three families of fixes; they form a ladder.

**Rung 1 — Plain augmentation (what the reference paper does).**
Take the existing minority images and make copies that are rotated, flipped, zoomed, brightness-shifted. Now you have more minority images.
- ✅ Easy, keeps images looking real.
- ❌ The copies carry the **same information** as the originals. If the model already struggles with a confusing Moderate image, you've just handed it 10 rotated copies of the *same* confusion. You changed the *count*, not the *coverage*.

**Rung 2 — SMOTE (Synthetic Minority Oversampling).**
Instead of copying, SMOTE *invents* new minority samples by **interpolating**: pick a minority sample, pick one of its k nearest minority neighbors, and create a new point somewhere on the line between them.
- ✅ Genuinely new points → fills in the gaps *between* real minority samples.
- ❌ Treats every minority sample **equally**. It spends as much effort manufacturing "easy" Moderate samples (deep in Moderate territory) as "hard" ones (near the border with Mild).

**Rung 3 — ADASYN (Adaptive Synthetic Sampling) — your choice.**
ADASYN is SMOTE with a **brain**. It looks at each minority sample and asks *"how surrounded by the enemy is this one?"* The more majority-class neighbors a minority point has, the **harder** it is to classify — and the **more** synthetic samples ADASYN generates near it.

> **Memorize this analogy:** Plain augmentation re-reads the same flashcards louder. SMOTE makes new flashcards uniformly. ADASYN makes *extra* flashcards specifically for the hamster photos that look like rabbits — the borderline cases the model keeps getting wrong.

---

## 2. ADASYN's algorithm, in plain words (He et al., 2008 — Algorithm 1)

You do **not** need the heavy math, but you must be able to narrate these five moves:

1. **Measure the imbalance.** Count majority (`m_maj`) vs minority (`m_min`). Decide how many synthetic samples `G` you need to (roughly) balance them.
2. **Score each minority sample's difficulty.** For minority sample `xᵢ`, look at its `k` nearest neighbors (across *all* classes). Let `rᵢ = (# of those neighbors that are majority) / k`. A point fully surrounded by the enemy gets `rᵢ = 1` (very hard); a point among its own kind gets `rᵢ ≈ 0` (easy).
3. **Turn difficulty into a budget.** Normalize the `rᵢ` so they sum to 1, then multiply by `G`. Hard points get a **bigger share** of the synthetic budget. *This weighting step is the whole point of ADASYN.*
4. **Generate.** For each minority point, create its allotted number of synthetic samples by interpolating toward a randomly chosen *minority* neighbor (the SMOTE move).
5. **Result.** A training set where the minority class is not just bigger, but **denser exactly where the decision boundary is contested.**

The one-line difference vs SMOTE: **SMOTE's per-sample budget is uniform; ADASYN's is proportional to local difficulty (`rᵢ`).**

---

## 3. The catch for *images* (don't skip — reviewers will ask)

ADASYN was designed for **feature vectors** (rows in a spreadsheet), not 2D pictures. Interpolating "halfway between two MRIs" is not obviously meaningful. So *where* you apply it matters. There are two placements (you'll decide concretely in Phase 3):

- **Option A — pixel space:** flatten each image to a long vector (e.g. 64×64 = 4,096 numbers), run ADASYN, reshape back to an image. Simple and fast; synthetic samples can look a little blurry.
- **Option B — embedding space:** push images through a pretrained CNN, grab the penultimate feature vector (e.g. 2,048-dim for ResNet50), run ADASYN there. Cleaner (interpolating *features* is more meaningful than interpolating raw pixels) but needs an extra extraction pass.

You'll use **Option A for the main pipeline, Option B as an ablation.** For now just hold the idea: *ADASYN needs vectors, and an image has to be turned into a vector first.*

---

## 4. The one rule you must never break (preview of Phase 3)

> **ADASYN is applied AFTER the train/val/test split, to the TRAINING SET ONLY.**

If you balance first and split later, synthetic samples built from a test image's neighbors leak into training. Your test score inflates, and a reviewer who spots it rejects the paper. We'll enforce this in Phase 3 — flag it now so it's burned in.

### Resources (a few hours total)
- **He, Bai, Garcia, Li (2008)** — *"ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning."* Read the intro + **Algorithm 1**. That's the primary source you'll cite.
- **`imbalanced-learn` docs** → "ADASYN" and "SMOTE" pages — see the toy 2D scatter plots; they make the difference click instantly.
- **StatQuest / any SMOTE explainer** (YouTube) for visual interpolation intuition.

---

## 5. Your task (verifies Step 2 — do it before moving on)

Write these down (same notebook as Step 1 — raw material for your Methodology section):

- [ ] **1. One paragraph, your own words:** explain the difference between augmentation, SMOTE, and ADASYN using a *non-medical* analogy of your own.
- [ ] **2. Define `rᵢ`** in a single sentence and say what `rᵢ = 1` vs `rᵢ = 0` means for how many synthetic samples that point gets.
- [ ] **3. The leakage question:** in two sentences, explain *why* applying ADASYN before splitting inflates the test score. (If you can answer this, you understand the #1 reviewer trap.)
- [ ] **4.** State which placement (pixel vs embedding space) is the main pipeline and which is the ablation.

✅ When #3 rolls off your tongue, you've internalized both *what* ADASYN does and *how not to misuse it.*

## When you're ready

> **Phase 1 · Step 3 — Why a heterogeneous-paradigm ensemble beats a same-family one (Pillar 2).**
