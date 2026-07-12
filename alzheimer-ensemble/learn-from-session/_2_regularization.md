# Fighting Overfitting & Underfitting — The Complete Toolkit

*A top-to-bottom guide. Read it in order — each section builds on the previous one.*

---

## 1. The Core Problem: Fitting

Every ML model tries to learn a pattern from training data, then apply it to new, unseen data.
The ability to perform well on unseen data is called **generalization** — it is the entire
point of machine learning. A model that only works on data it has already seen is just a
lookup table.

Two things can go wrong:

| | Underfitting | Overfitting |
|---|---|---|
| **What happens** | Model is too simple to learn the pattern | Model memorizes training data, including its noise |
| **Training accuracy** | Low | Very high (often ~100%) |
| **Validation accuracy** | Low | Much lower than training |
| **Also called** | High **bias** | High **variance** |
| **Analogy** | Student who didn't study | Student who memorized past exams word-for-word |

**The key diagnostic:** the *gap* between training and validation performance.

- Both low → underfitting (high bias)
- Train high, validation low → overfitting (high variance)
- Both high, small gap → just right ✅

### The bias–variance tradeoff (the idea behind everything)

- **Bias** = error from wrong assumptions. A straight line trying to fit a curve will
  *always* miss, no matter how much data you give it. Rigid model → high bias.
- **Variance** = error from sensitivity to the particular training sample you happened to
  get. Retrain a very flexible model on a slightly different sample and you get a wildly
  different model. Flexible model → high variance.

Picture fitting points that roughly follow a curve:
- A straight line (degree-1 polynomial): misses the curve entirely → **underfit**.
- A degree-3 polynomial: follows the trend, ignores the jitter → **just right**.
- A degree-30 polynomial: passes through *every single point*, wiggling violently between
  them → **overfit**. Ask it to predict between two points and it gives nonsense.

You can't minimize both bias and variance for free — reducing one tends to increase the
other. Every technique in this file is a way of trading between them intelligently.

---

## 2. First, Diagnose — Don't Guess

Before applying any fix, **plot learning curves**: training loss and validation loss over
epochs (or over training-set size).

```
loss
 │ \
 │  \  ___________________  ← validation loss flattens, then rises
 │   \/
 │    \_______
 │            \___________  ← training loss keeps falling
 └────────────────────────→ epochs
        ↑ overfitting starts where the curves diverge
```

How to read them:

- Validation loss goes down, then turns back **up** while training loss keeps falling →
  overfitting has begun at that turning point.
- Both losses plateau at a **high** value → underfitting.
- Both still falling → not converged; just train longer.
- Validation loss is *noisy/jumpy* → validation set may be too small, or learning rate too high.

**Rule of thumb: fix underfitting first.** Get a model that can at least learn the pattern,
*then* constrain it. A classic sanity check: your model should be able to reach ~100%
accuracy on a tiny sample (say, 10 examples). If it can't even overfit 10 examples, it has
a bug or is far too weak — no regularization discussion applies yet.

---

## 3. Fixes for UNDERFITTING (make the model stronger)

Underfitting is usually the easier problem — the fixes are direct:

1. **Bigger / deeper model** — more layers, more neurons, deeper trees. More capacity to
   represent complex patterns.
2. **Train longer** — maybe it just hasn't converged yet. Check the curves.
3. **Better features** — feature engineering, or richer inputs (for images: higher
   resolution; for tabular data: interaction terms, domain-specific ratios).
4. **Reduce regularization** — yes, everything in Section 4 can *cause* underfitting if
   applied too strongly. Regularization is a dial, not a switch. If you inherit a model
   with dropout 0.7 and huge weight decay and it's underperforming, turn those *down* first.
5. **Tune the learning rate** — too high: loss bounces around and never settles;
   too low: loss creeps down so slowly it looks like a plateau.

The rest of this file is about the harder problem.

---

## 4. Fixes for OVERFITTING — The Big Six

A useful mental model before diving in: an overfit model has found it *easier to memorize
than to generalize*. Every technique below changes the economics — it makes memorization
expensive and generalization cheap.

### 4.1 More Data (the king)

Nothing beats more real data. Overfitting = memorizing noise; with more examples, the noise
averages out and only the true pattern survives. A degree-30 polynomial fit to 10 points is
a disaster; fit to 10,000 points, it's forced to behave.

When you can't collect more, *manufacture* more:

**Data augmentation** (you know this one!) — create modified copies of training samples:

- **Images:** flips, rotations, crops, brightness/contrast shifts, small translations,
  cutout (mask a random patch), mixup (blend two images and their labels).
- **Text:** synonym replacement, back-translation (English → French → English gives a
  paraphrase), random word deletion.
- **Audio:** pitch shift, time stretch, background noise overlay.
- **Tabular:** harder — SMOTE for imbalanced classes, small Gaussian noise on numeric features.

Why it works: it teaches the model *invariances* — "a rotated brain scan is still the same
brain." The model can't memorize a sample it never sees the same way twice.

```python
# PyTorch example — augmentation lives in the training transform only
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
val_tf = transforms.Compose([transforms.ToTensor()])   # ← no augmentation!
```

⚠️ Two rules:
1. Only augment the **training set**, never validation/test — you want to *evaluate* on
   realistic, unmodified data.
2. Keep augmentations **label-preserving**. Don't flip a digit '6' vertically (it becomes
   a '9'), and think carefully with medical images — anatomy has a real left and right.

### 4.2 Simpler Model

The mirror image of Section 3, item 1. Fewer parameters = less capacity to memorize.

- Neural nets: fewer layers, fewer neurons per layer.
- Trees: limit `max_depth`, raise `min_samples_leaf`.
- Linear models with many features: drop features (see Section 5).

This is **Occam's razor** as a strategy: among models that fit the data comparably well,
prefer the simplest. Simple models have fewer ways to be wrong on new data.

### 4.3 L1 / L2 Regularization (weight penalties)

Instead of shrinking the architecture, keep it big but *penalize* it for using its full
capacity. Add a penalty term to the loss function:

```
Loss_total = Loss_data + λ · penalty(weights)
```

The model now has two competing goals: fit the data (first term) AND keep weights small
(second term). It only "spends" large weights when the data truly justifies them.

- **L2 (Ridge / weight decay):** penalty = sum of *squared* weights (Σw²). Shrinks all
  weights toward zero smoothly and proportionally — big weights get punished hardest.
  The default choice; in deep learning it appears as the `weight_decay` optimizer argument.
- **L1 (Lasso):** penalty = sum of *absolute* weights (Σ|w|). Its geometry pushes some
  weights to *exactly* zero, effectively deleting those features → automatic feature
  selection. Useful when you suspect many features are irrelevant.
- **Elastic Net:** a weighted mix of both — L1's sparsity with L2's stability.

**Intuition:** a model with huge weights produces wildly curvy decision boundaries — it's
contorting itself to pass through every noisy training point (that degree-30 polynomial
again: its coefficients are enormous). Small weights = smooth, gentle functions = better
generalization.

**λ (lambda)** controls the strength — it's the dial:
- λ = 0 → no regularization, back to plain overfitting risk.
- λ too high → the model cares only about small weights, ignores the data → underfitting.
- Tune it on the validation set, typically over a log scale: 0.0001, 0.001, 0.01, 0.1, 1.

```python
# scikit-learn: same linear model, three flavors
from sklearn.linear_model import Ridge, Lasso, ElasticNet
Ridge(alpha=1.0)        # L2  (alpha is sklearn's name for λ)
Lasso(alpha=0.1)        # L1
ElasticNet(alpha=0.1, l1_ratio=0.5)

# PyTorch: L2 via the optimizer
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### 4.4 Dropout (neural networks)

During training, randomly "turn off" a fraction of neurons (commonly 20–50%) on **every
forward pass**. Each batch effectively trains a different randomly-thinned sub-network.

**Why it works — two views:**

1. **No fragile co-dependencies.** Without dropout, a neuron can learn to rely on one
   specific partner neuron to fix its mistakes — a brittle memorization circuit. With
   dropout, any partner might vanish at any moment, so every neuron must learn features
   that are useful *on their own*. The result is redundant, robust representations.
2. **A free ensemble.** Training with dropout ≈ training exponentially many thinned
   networks that share weights; inference approximates averaging all of them. Ensembles
   reduce variance (Section 4.6) — dropout gives you that for free inside one model.

At **inference time** dropout is turned off — all neurons active, outputs scaled so the
expected magnitude matches training. Frameworks handle this automatically, *but only if you
tell them*:

```python
model.train()   # dropout ON  — training
model.eval()    # dropout OFF — validation/inference
# Forgetting model.eval() before validating is one of the most common PyTorch bugs:
# your validation scores become noisy and pessimistic for no visible reason.
```

Placement tips: dropout goes between fully-connected layers most usefully; in modern
convnets it's often replaced or reduced because batch norm and augmentation already
regularize. Start with p=0.3–0.5 on dense layers and tune.

### 4.5 Early Stopping

Remember the learning-curve turning point from Section 2? Just **stop training there**.

The insight: *training time is itself a capacity dial*. Early in training, gradient descent
captures the broad, strong patterns (signal); only later does it descend into fitting
sample-specific quirks (noise). Stopping early keeps the signal, skips the noise.

The standard recipe:
1. After every epoch, evaluate on the validation set.
2. Keep a checkpoint of the best validation score seen so far.
3. If validation hasn't improved for **N consecutive epochs** (N = **patience**,
   typically 5–20), stop.
4. Restore the best checkpoint — not the last one.

```python
best_loss, patience, bad_epochs = float("inf"), 10, 0
for epoch in range(max_epochs):
    train_one_epoch(model)
    val_loss = evaluate(model)
    if val_loss < best_loss:
        best_loss, bad_epochs = val_loss, 0
        save_checkpoint(model)          # remember the best model
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break                        # stop; load_checkpoint() afterwards
```

Why patience matters: validation loss is noisy — a single bad epoch doesn't mean
overfitting has begun. Patience waits for a *sustained* failure to improve.

Early stopping is simple, costs nothing, combines with every other technique, and even
applies beyond neural nets (e.g., boosting rounds in XGBoost: `early_stopping_rounds`).

### 4.6 Ensembles

Combine several models and aggregate their predictions (this is what your project does!).

**Why averaging helps:** each model makes errors, but if the models are *diverse*, their
errors are partly independent — some guess high where others guess low — and averaging
cancels them out. The signal (which all models agree on) survives; the noise (which each
model got wrong differently) is diluted. Mathematically, averaging k independent models
divides the variance by up to k while leaving bias unchanged. Ensembles are a direct
attack on variance — i.e., on overfitting.

Three main flavors:

- **Bagging** (Bootstrap AGGregatING — e.g., Random Forest): train many copies of the same
  model, each on a random resample of the training data (and, in Random Forest, a random
  subset of features per split). Each tree overfits *its own* sample differently; the
  forest's average is stable. Bagging = variance reduction, pure and simple.
- **Boosting** (e.g., XGBoost, LightGBM): train models *sequentially*, each one focusing on
  the examples the previous ones got wrong. Boosting primarily reduces **bias** (it turns
  weak learners into a strong one) — and it *can* overfit if run too long, which is why it
  pairs with early stopping and shallow trees.
- **Stacking:** train several different model types (say, a CNN, a random forest, and a
  logistic regression), then train a small **meta-model** that takes their predictions as
  inputs and learns the best way to combine them. Crucial detail: the meta-model must be
  trained on *out-of-fold* predictions (predictions each base model made on data it didn't
  train on), otherwise it learns to trust overfit base-model outputs — a form of leakage.

Diversity is the fuel: ensembling five copies of the *same* model trained the *same* way
gains almost nothing. Vary the data (bagging), the features, the architecture, or the
random seed.

---

## 5. Honorable Mentions (worth knowing)

- **Batch Normalization:** normalizes each layer's activations across the batch. Primarily
  it stabilizes and speeds up training, but the batch-to-batch statistical noise acts as a
  mild regularizer too. Like dropout, it behaves differently in train vs. eval mode.
- **Label smoothing:** train against soft targets (e.g., 0.9/0.1 instead of 1.0/0.0) so the
  model is never rewarded for pushing predictions to absurd overconfidence.
- **Noise injection:** add small random noise to inputs, weights, or gradients during
  training — same spirit as augmentation: never show the model the exact same problem twice.
- **Transfer learning:** start from a model pre-trained on a huge dataset (ImageNet, etc.).
  Its early layers already encode general-purpose features that generalize well, so your
  small dataset fine-tunes rather than teaches from scratch — far less room to memorize.
  Especially powerful for medical imaging, where labeled data is scarce.
- **Feature selection / dimensionality reduction (PCA):** fewer input features = fewer ways
  to memorize noise. Matters most when features ≫ samples (e.g., 10,000 gene measurements,
  200 patients) — in that regime a model can "explain" anything by chance.
- **Gradient clipping & learning-rate schedules:** not regularizers strictly, but they keep
  training stable, which keeps your learning curves readable and your diagnosis honest.

---

## 6. The Golden Rule Behind Everything: Proper Validation

None of the above helps if you *measure* generalization wrongly. Regularization tunes the
model against the validation score — so the validation score must be trustworthy.

- **Train / Validation / Test split** (e.g., 70/15/15):
  - **Train** — fit the weights.
  - **Validation** — tune hyperparameters: λ, dropout rate, patience, model size, which
    augmentations to use.
  - **Test** — touched *exactly once*, at the very end, to report final performance.
  - Why keep them separate? Because tuning on a set is a slow, indirect way of fitting to
    it. Tune long enough against validation and you've overfit the validation set — the
    untouched test set is your insurance.

- **k-Fold Cross-Validation:** with small datasets, a single split wastes data and gives a
  noisy estimate. Instead, split the data into k folds (k=5 or 10), train k times, each
  time holding out a different fold for validation, and average the k scores. Every sample
  gets used for both training and validation (never simultaneously), and the averaged score
  is far more reliable.

- **Data leakage** — the silent killer: information from validation/test sneaking into
  training. It makes an overfit model *look* great, right up until real-world deployment.
  Classic traps:
  - **Preprocessing before splitting:** computing normalization statistics (mean/std),
    or fitting PCA/feature selection, on the *whole* dataset before splitting. The training
    pipeline has now "seen" the test data. Always: split first, fit preprocessing on train
    only, apply to the rest.
  - **Grouped data split naively:** the same patient's multiple scans landing in both train
    and test. The model recognizes the *patient*, not the disease. Use group-aware splits
    (`GroupKFold` with patient ID) — directly relevant to an Alzheimer's dataset.
  - **Time-series shuffled randomly:** training on the future to predict the past. Use
    chronological splits.
  - **Target leakage:** a feature that is a proxy for the label and won't exist at
    prediction time (e.g., "treatment prescribed" as a feature for predicting diagnosis).

If your validation or test score ever looks *too good*, suspect leakage before celebrating.

---

## 7. Cheat Sheet

| Symptom | Try, in order |
|---|---|
| Train low, val low (underfit) | Bigger model → train longer → better features → *less* regularization → tune learning rate |
| Train high, val low (overfit) | More data / augmentation → L2 or dropout → early stopping → simpler model → ensemble |
| Val loss noisy/jumpy | Bigger validation set, lower learning rate |
| Val score suspiciously perfect | Check for data leakage! |
| Boosting model degrades late | `early_stopping_rounds`, shallower trees, lower learning rate |

**A practical workflow:**
1. Build a model big enough to overfit a small sample — proves it *can* learn.
2. Train on the full data; watch learning curves every run.
3. Add regularization **one technique at a time**, re-checking the curves after each — if
   you add five things at once, you won't know which helped or hurt.
4. Tune strengths (λ, dropout rate, patience) on the validation set, over a log scale.
5. When happy, evaluate on the test set exactly once and report that number.

**Sensible starting points** (tune from here, don't trust them blindly):
weight decay 1e-4 · dropout 0.3–0.5 on dense layers · patience 10 ·
Random Forest `max_depth` 5–15 · standard flips/rotations for natural images.

---

## 8. One-Paragraph Summary

Underfitting means the model is too weak — give it more capacity, better features, or more
training. Overfitting means it memorized noise — constrain it. Every anti-overfitting
technique is a different way of applying the same pressure: **make memorization harder than
generalization.** More data and augmentation dilute the noise and teach invariances; L1/L2
penalties make contorted, large-weight solutions expensive; dropout forbids fragile neuron
co-dependencies and averages a free ensemble; early stopping caps how long the model has to
descend into noise; simpler models remove the capacity to memorize; ensembles average away
each member's individual variance. Diagnose with learning curves before reaching for any of
them, tune one dial at a time against a clean validation set, guard fiercely against data
leakage, and remember: the whole game is played on the bias–variance tradeoff, and the only
score that counts is on data the model has never seen.
