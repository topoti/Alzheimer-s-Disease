# Phase 6 · Step 3 — Ablation Studies

> **Phase 6 — Evaluation**
> The goal of this step: run controlled experiments that *isolate the contribution* of each of your two methodological pillars (ADASYN, heterogeneity).

The one question this step answers:

> **How do I prove that ADASYN and the heterogeneous ensemble each actually helped — not just that the final model is good?**

---

## 1. What an ablation is and why reviewers demand it

An **ablation** removes one component and re-measures, so any drop in performance is *attributable to that component*. Without ablations, a reviewer can say "nice result, but maybe it's all from the modern backbones, not your ADASYN or your heterogeneity." Ablations close that door.

You run **two**, each changing exactly one thing while holding everything else fixed (same splits, same training recipe, same seed):

---

## 2. Ablation 1 — Without ADASYN

Re-run the **whole pipeline using plain augmentation instead of ADASYN** (i.e., the reference paper's imbalance handling). Compare to your ADASYN pipeline.

- **What it isolates:** the value of ADASYN.
- **Where you expect the difference:** **Moderate-class recall** (and macro-F1). The plan's success bar: *ablation without ADASYN shows ≥3pp drop in Moderate recall.* If removing ADASYN barely changes Moderate recall, your Pillar 1 claim is weak — important to know honestly.

---

## 3. Ablation 2 — Same-family ensemble

Replace the heterogeneous trio with a **same-family ensemble** (e.g., three ResNets of different depths, or three DenseNets) and re-evaluate.

- **What it isolates:** the value of paradigm *heterogeneity* (Phase 1 Step 3's uncorrelated-error claim).
- **What you expect:** the heterogeneous ensemble shows a larger gain over its best member than the same-family ensemble does — evidence that decorrelated errors, not just "having 3 models," drive the improvement.

> Tip: also measure **error correlation** between members (e.g., agreement on wrong predictions) for both ensembles — a direct, quantitative version of the heterogeneity argument that's very persuasive in Discussion.

---

## 4. Keep everything else fixed

The only thing that changes between an ablation and the main run is the **one variable under test**. Same data splits (`data/splits/`), same `train_one_model` recipe, same seeds, same `evaluate()`. Otherwise you can't attribute the difference. Log each ablation's full metric dict for the comparison table (next step).

### Resources (a few hours total)
- Reuse your own `src/adasyn_pipeline.py` (toggle to plain augmentation) and `src/models.py` (swap to same-family trio).
- Any short "ablation study best practices" note — the "change one variable" principle.

---

## 5. Your task (verifies Step 3 — do it before moving on)

- [ ] **1.** Run the **no-ADASYN** ablation (plain augmentation); record full metrics, especially Moderate recall.
- [ ] **2.** Confirm the ADASYN→no-ADASYN **Moderate-recall drop is ≥3pp** (or honestly report if not).
- [ ] **3.** Run the **same-family ensemble** ablation; compare its ensemble gain to the heterogeneous one.
- [ ] **4.** (Optional but strong) Measure error correlation between members for both ensembles.

✅ When both ablations are quantified and each pillar's contribution is isolated, move on.

## When you're ready

> **Phase 6 · Step 4 — The comparison table and statistical significance testing.**
