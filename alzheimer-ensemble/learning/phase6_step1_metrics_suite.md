# Phase 6 · Step 1 — The Metrics Suite

> **Phase 6 — Evaluation**
> The goal of this step: build one `evaluate()` function that reports the *right* metrics for imbalanced medical data — and understand why accuracy is the wrong headline.

The one question this step answers:

> **Which numbers actually tell me my model works, and why isn't accuracy one of them?**

---

## 1. The five metrics and why each matters here

| Metric | What it measures | Why it matters in this project |
|---|---|---|
| **Accuracy** | % correctly classified | **Misleading** on imbalanced data — always predicting "Non-Dementia" scores ~50% (Step 1's dumb baseline). Report it, but de-emphasize. |
| **Precision (per class)** | Of all "Mild" predictions, how many were truly Mild? | Low precision = false alarms → patient anxiety, wasted follow-ups |
| **Recall / Sensitivity (per class)** | Of all true Mild patients, how many did we catch? | **The most critical metric.** A missed dementia case (false negative) is far worse than a false alarm |
| **F1-Score (per class + macro)** | Harmonic mean of precision & recall | One number balancing both. **Macro-F1** (unweighted average over the 4 classes) is the right *headline* for imbalanced medical data |
| **AUC-ROC (one-vs-rest)** | How well the model *ranks* positives above negatives, threshold-free | Doctors may shift the decision threshold (screening vs diagnosis); AUC captures performance across all thresholds |

---

## 2. Why macro-F1 is the headline

**Macro**-F1 averages the per-class F1 scores *without* weighting by class size — so the tiny Moderate class counts **just as much** as the giant Non-Dementia class. That's exactly what you want: a model that ignores Moderate gets punished hard in macro-F1 but barely dented in accuracy. This is the metric that exposes whether ADASYN actually worked.

> **Headline reporting (plan):** Macro-F1 + per-class Recall + Confusion Matrix. Accuracy reported but de-emphasized.

---

## 3. The `evaluate()` function

Build one `evaluate(model_or_ensemble, test_loader) → dict` in `src/evaluate.py` that returns **all** metrics at once: accuracy, per-class precision/recall/F1, macro-F1, and one-vs-rest AUC-ROC. Reuse it for:
- per-epoch validation during training (Phase 4 Step 2),
- each single model on test,
- the ensemble on test,
- the ablations (next steps).

`sklearn.metrics` gives you everything: `classification_report`, `f1_score(average='macro')`, `recall_score(average=None)`, `roc_auc_score(multi_class='ovr')`. Standardizing on one function guarantees every row of your comparison table is computed identically.

### Resources (a few hours total)
- `sklearn.metrics`: `classification_report`, `f1_score`, `recall_score`, `precision_score`, `roc_auc_score`, `confusion_matrix`.
- Any "why accuracy is misleading on imbalanced data" explainer (ties back to Phase 1 Step 1).

---

## 4. Your task (verifies Step 1 — do it before moving on)

- [ ] **1.** Explain in two sentences why macro-F1, not accuracy, is your headline metric.
- [ ] **2.** State why **recall** is the metric you most care about clinically, with an example of the cost of a false negative.
- [ ] **3.** Implement `evaluate(...)` returning the full metric dict; run it on the ensemble's saved test predictions.
- [ ] **4.** Print the per-class recall — note Moderate's value (you'll compare it to the no-ADASYN ablation).

✅ When `evaluate()` returns every metric and you can defend macro-F1 as the headline, move on.

## When you're ready

> **Phase 6 · Step 2 — Confusion matrices and ROC curves.**
