# Phase 7 · Step 1 — Paper Structure & the Novelty Paragraph

> **Phase 7 — Paper Writing** (3–5 weeks — the longest calendar phase)
> The goal of this step: lock the section-by-section skeleton and write the one paragraph that positions your novelty correctly.

The one question this step answers:

> **What is the shape of the paper, and how do I state my contribution so a reviewer accepts the novelty?**

---

## 1. The standard structure (8 sections)

1. **Abstract** (~250 words) — problem, dataset, your 3 pillars, headline result, one clinical-implication sentence.
2. **Introduction** (1.5–2 pp) — epidemiology → why early diagnosis/MRI → why DL + open challenges → **your 3 contributions (bulleted)** → roadmap.
3. **Related Work** (2–3 pp) — DL for Alzheimer's MRI; class imbalance (SMOTE/ADASYN/aug); ensembles in medical imaging; XAI in medical imaging; end with the **gap table** (Phase 1 Step 5).
4. **Methodology** (3–4 pp) — dataset, preprocessing, ADASYN, three backbones, ensemble fusion, XAI, training protocol.
5. **Results** (3–4 pp) — single models, ensemble, ablations, confusion/ROC, XAI figure, comparison with SOTA.
6. **Discussion** (1.5–2 pp) — why heterogeneity helps, clinical implications of XAI, limitations, threats to validity.
7. **Conclusion + Future Work** (~0.5 p).
8. **References** (35–55).

**Writing order (not document order):** draft **Methodology → Results → Introduction → Related Work → Discussion → Abstract**. You write what you *did* first (easy, factual), and the framing last (once you know the story).

---

## 2. The novelty-positioning paragraph (the make-or-break)

This appears in both Introduction and Discussion. Reuse your Phase 1 Step 5 draft:

> *"Unlike Mahmud et al. (2024), who combined models from the same architectural family (e.g., VGG16 + VGG19), our ensemble integrates three CNN backbones built on fundamentally different design paradigms — residual learning, compound scaling, and dense connectivity — which we hypothesize produces more decorrelated errors and thus stronger ensemble gain. We further replace plain image augmentation with ADASYN, which specifically targets borderline minority-class samples, and extend explainability beyond Grad-CAM by adding SHAP and LIME for quantitative feature attribution."*

The three guardrails baked into it:
- Say **"cross-paradigm" / "heterogeneous CNN"** — **never "cross-architecture"** (Phase 1 Step 3 rule).
- Name the baseline explicitly so the contribution is *relative and concrete*.
- Hedge claims ("we hypothesize") — overclaiming invites rejection.

---

## 3. The three contributions, bulleted (for the Intro)

State them crisply, in this order:
1. **ADASYN** replaces plain augmentation to target hard borderline minority samples.
2. **Heterogeneous-paradigm CNN ensemble** (residual + compound-scaled + densely-connected) for decorrelated errors.
3. **Triple-XAI** (Grad-CAM + SHAP + LIME) applied to the ensemble for visual *and* quantitative explanation.

### Resources (a few hours total)
- The **target journal's author template** (start with *Diagnostics*, MDPI).
- 2–3 recently accepted papers in the same venue — mimic their section rhythm and figure style.

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Create the LaTeX project (Overleaf) with the target journal template and the 8 section headers as stubs.
- [ ] **2.** Write the bulleted 3-contribution list for the Introduction.
- [ ] **3.** Finalize the novelty-positioning paragraph (paste into both Intro and Discussion stubs).
- [ ] **4.** Drop the Phase 1 Step 5 gap table into the Related Work stub.

✅ When the skeleton exists and the novelty paragraph is locked, start drafting.

## When you're ready

> **Phase 7 · Step 2 — Draft the Introduction and Related Work.**
