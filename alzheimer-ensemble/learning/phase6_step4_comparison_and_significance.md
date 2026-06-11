# Phase 6 · Step 4 — Comparison Table & Statistical Significance

> **Phase 6 — Evaluation** (closes Phase 6)
> The goal of this step: assemble the headline comparison table and prove your improvement is *statistically real*, not noise.

The one question this step answers:

> **Is my ensemble actually better than the baselines and the single models — and can I prove it's not luck?**

---

## 1. The comparison table (your paper's Table 4-ish)

Fill in every row with the metrics from `evaluate()`:

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

Add the two **ablation rows** (no-ADASYN, same-family) from Step 3 so the table tells the whole story: each pillar's contribution is visible in one place.

---

## 2. Statistical significance — don't claim "better" without it

A higher number isn't enough; you must show the improvement is **unlikely to be chance**. Two standard tools for classifiers on a fixed test set:

- **McNemar's test** — compares two models' *paired* predictions on the same test items (where one is right and the other wrong). Ideal for "ensemble vs best single model." Reports a p-value.
- **Paired bootstrap** — resample the test set many times, recompute the metric gap each time, and report a confidence interval for the difference. More flexible (works for macro-F1, AUC, etc.).

Use one (or both) to back the claim *"the ensemble significantly outperforms the best single model"* and ideally *"the ADASYN pipeline significantly improves Moderate recall over the no-ADASYN ablation."*

> The honest framing: report the p-value / CI right next to the claim. "97% vs 96%" means little; "macro-F1 improvement, McNemar p<0.05" means a lot.

---

## 3. The bar to clear (plan's Phase 6 verification)

- **Macro-F1 ≥ 0.93** and **Moderate-class recall ≥ 0.80** on test.
- No-ADASYN ablation shows **≥3pp drop** in Moderate recall.

If you fall short, the risk-mitigation levers (from the plan): class-weighted or focal loss on top of ADASYN; embedding-space ADASYN (Option B); or expanding data (OASIS-2/ADNI).

### Resources (a few hours total)
- `statsmodels.stats.contingency_tables.mcnemar` for McNemar's test.
- Any "bootstrap confidence interval for model comparison" tutorial.

---

## 4. Your task (verifies Step 4 — and closes Phase 6)

- [ ] **1.** Fill the full comparison table (4 single/ensemble rows + 2 ablation rows + Mahmud et al.).
- [ ] **2.** Run **McNemar's test** (or paired bootstrap) for ensemble vs best single model; record the p-value/CI.
- [ ] **3.** Confirm the bar: **macro-F1 ≥ 0.93** and **Moderate recall ≥ 0.80**; note any shortfall + mitigation.
- [ ] **4.** Save the table to `results/tables/` ready for the paper.

✅ Phase 6 verification (plan): macro-F1 ≥ 0.93, Moderate recall ≥ 0.80, and the no-ADASYN ablation drop ≥ 3pp. When the numbers clear (and are significant), you're ready to write.

## When you're ready

> **Phase 7 · Step 1 — Paper structure and the novelty-positioning paragraph.**
