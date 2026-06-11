# Phase 2 · Step 2 — Ensemble Combination: Weighted Soft Voting

> **Phase 2 — Architecture Selection**
> The goal of this step: decide *how* the three models' outputs become one prediction, and be able to justify the choice against the alternatives.

The one question this step answers:

> **Given three probability vectors, how do I fuse them into a single answer — and why this way?**

---

## 1. Three ways to combine model votes

| Strategy | How it works | Trade-off |
|---|---|---|
| **Hard voting** | Each model picks one class; majority wins | Simple, but **throws away confidence** — a 0.99 vote counts the same as a 0.34 vote |
| **Soft voting (averaging)** | Average the three 4-class **probability** vectors, take argmax | Keeps confidence info; standard; simple; works well |
| **Stacking** | Train a *meta-classifier* on the three models' outputs | Most powerful, but adds complexity and **risks leakage** if not validated carefully |

---

## 2. The decision: **weighted soft voting**

Average the probability vectors, but let a better model speak louder. Weights are **proportional to each model's validation-set macro-F1**, normalized to sum to 1.

```
p_ensemble = w1·softmax(ResNet50(x)) + w2·softmax(EffNetB3(x)) + w3·softmax(DenseNet121(x))
prediction = argmax(p_ensemble)          # w1 + w2 + w3 = 1
```

Why this is the sweet spot:
- **vs hard voting:** keeps the confidence signal (the whole reason you used softmax in Step 1).
- **vs stacking:** no extra model to train, no leakage risk, far easier to explain in a paper.
- **Weighting by val-F1** (not val-accuracy) is deliberate — F1 respects the minority class, so a model that's genuinely better on rare Moderate cases gets more say. (Why F1 not accuracy is Phase 6's lesson; you're using it early here.)

> Keep **stacking as an ablation** — if a reviewer asks "did you try stacking?", you can say yes and report it, without it being your main method.

---

## 3. The one rule for the weights (no leakage)

> **Weights come from the VALIDATION set, never the test set.**

Compute each model's macro-F1 on validation, derive `w1, w2, w3`, *freeze them*, then evaluate the ensemble once on test. If you tune weights on test, the test score is no longer honest. Same discipline as the ADASYN rule in Phase 1 Step 2.

---

## 4. Pseudocode to internalize (you'll implement in `src/ensemble.py`)

```
load three trained checkpoints
for each batch x in test_loader:
    p1 = softmax(resnet(x));  p2 = softmax(effnet(x));  p3 = softmax(densenet(x))
    p  = w1*p1 + w2*p2 + w3*p3
    preds = argmax(p, dim=1)
collect preds → feed to evaluate()
```

Mind the input sizes from Step 1: ResNet/DenseNet get the 224 tensor, EffNet gets the 300 tensor — the ensemble loop must route each model its correct-resolution input.

### Resources (a few hours total)
- `sklearn.ensemble.VotingClassifier` docs — read the `voting='soft'` section for the concept (you'll hand-roll it in PyTorch, but the idea is identical).
- Any short "soft voting vs hard voting" explainer for the confidence-loss intuition.

---

## 5. Your task (verifies Step 2 — do it before moving on)

- [ ] **1.** Write the weighted soft-voting formula from memory, including the constraint on the weights.
- [ ] **2.** In one sentence each, say why soft voting beats hard voting, and why you chose it over stacking.
- [ ] **3.** State *which dataset split* the weights are computed on and why it must not be test.
- [ ] **4.** Note one detail the ensemble loop must respect because of mixed input sizes.

✅ When you can defend "weighted soft voting by val-F1" in one breath, move on.

## When you're ready

> **Phase 2 · Step 3 — The classifier head and sanity-loading all three from `timm`.**
