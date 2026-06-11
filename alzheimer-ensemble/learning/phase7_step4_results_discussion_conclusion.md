# Phase 7 · Step 4 — Draft Results, Discussion & Conclusion

> **Phase 7 — Paper Writing**
> The goal of this step: report the numbers objectively (Results), interpret them honestly (Discussion), and close (Conclusion).

The one question this step answers:

> **How do I present my findings convincingly *and* honestly — including the limitations?**

---

## 1. Results — report, don't interpret (yet)

Six subsections, each pointing at a figure/table you already built in Phase 6:
5.1 **Individual model results** (table) — the three backbones on test.
5.2 **Ensemble results** (table) — the proposed model; note the gain over the best single model + the **significance test** (Phase 6 Step 4).
5.3 **Ablation studies** — without-ADASYN and same-family; quantify each pillar's contribution.
5.4 **Confusion matrices & ROC curves** (figures) — Phase 6 Step 2.
5.5 **XAI visualizations** — the 4-column figure (MRI | Grad-CAM | SHAP | LIME), Phase 5 Step 4.
5.6 **Comparison with Mahmud et al. and SOTA** — the comparison table, Phase 6 Step 4.

In Results, state *what* the numbers are. Save *why* for Discussion. Report accuracy but **lead with macro-F1 and per-class recall**.

---

## 2. Discussion — interpret, and be honest about limits

- **Why heterogeneity helped** — tie the ensemble gain to the uncorrelated-error theory (Phase 1 Step 3); cite your error-correlation ablation if you ran it.
- **Clinical implications of the XAI** — the methods converging on hippocampal/medial-temporal regions builds clinical trust; the clinician's sanity check (Phase 5 Step 4).
- **Limitations (state them openly — it strengthens the paper):**
  - **All three backbones are CNNs** — this is "cross-paradigm," not "cross-architecture" (the honesty clause). Frame CNN+Transformer as future work.
  - **2D slices, not 3D volumes** — loses inter-slice context.
  - **Single dataset (OASIS)** — generalization unproven without multi-site validation.
- **Threats to validity** — slice-level (not patient-level) splitting risk; pixel-space ADASYN producing blurry synthetics; small Moderate class even after balancing.

> Reviewers reward openly-acknowledged limitations and punish hidden ones. The CNN-only admission is the single most important sentence for acceptance.

---

## 3. Conclusion + Future Work (~0.5 page)

Summarize the three contributions and headline result in a few sentences, then list future work: **CNN + Transformer ensemble** (true cross-architecture), **3D volumetric input**, **multi-site validation (OASIS-2/ADNI)**, **prospective clinical study**.

### Resources (a few hours total)
- All Phase 6 tables/figures and the Phase 5 XAI figure — Results is assembly.
- A "how to write a limitations section" guide — frame each limit + its mitigation/future direction.

---

## 4. Your task (verifies Step 4 — do it before moving on)

- [ ] **1.** Draft Results 5.1–5.6, each referencing its figure/table; lead with macro-F1 + per-class recall.
- [ ] **2.** Draft Discussion: heterogeneity rationale, clinical XAI implications, and the **3 limitations** (CNN-only first).
- [ ] **3.** Draft Conclusion + Future Work with the four future directions.
- [ ] **4.** Verify every claim in Discussion is backed by a number/figure in Results (no unsupported assertions).

✅ When Results state the facts and Discussion interprets them honestly (limits included), move on to polish.

## When you're ready

> **Phase 7 · Step 5 — Figures, references, review, and submission.**
