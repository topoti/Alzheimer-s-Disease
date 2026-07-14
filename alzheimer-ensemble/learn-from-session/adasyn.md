# ADASYN — Adaptive Synthetic Sampling (A Learner's Guide)

## 1. The Problem ADASYN Solves: Imbalanced Data

Imagine your Alzheimer's dataset has:

- **900 "Healthy"** patients (majority class)
- **100 "Alzheimer's"** patients (minority class)

A machine learning model that simply predicts *"Healthy"* for **everyone**
would be **90% accurate** — and completely useless. It never catches a single
sick patient. This is the **class imbalance problem**.

The model has almost no examples of the class you care most about (the sick
patients), so it never learns what they look like. It becomes lazy and biased
toward the majority.

**Goal:** give the model *more* minority examples so it pays attention to them.

There are three broad ways to fix imbalance:

1. **Undersampling** — throw away majority examples (you lose data).
2. **Oversampling** — duplicate/create more minority examples.
3. **Cost-sensitive learning** — punish the model more for missing the rare class.

ADASYN belongs to family **#2 — oversampling**. But it does it *intelligently*.

---

## 2. The Naive Approach and Why It Fails

**Random Oversampling** just copies existing minority patients until the counts
match. Problem: exact duplicates teach the model nothing new — it just memorizes
the same points and **overfits**.

**SMOTE** (the famous predecessor, 2002) improved this. Instead of copying, it
*creates new synthetic patients* by interpolating between a minority patient and
its nearest minority neighbor:

```
new_point = existing_point + random(0,1) × (neighbor − existing_point)
```

Think of it as drawing a line between two sick patients and placing a brand-new
"imaginary patient" somewhere on that line. This fills the space and reduces
overfitting.

But SMOTE has a blind spot: **it treats every minority point equally.** It makes
the same number of synthetic points everywhere — even in easy regions the model
already understands.

---

## 3. The Core Idea of ADASYN

ADASYN (He et al., 2008) asks one clever question:

> **"Which minority patients are HARD to learn — and can we make MORE synthetic
> data specifically around those hard cases?"**

The insight: a minority point surrounded by lots of **majority** neighbors sits
near the **decision boundary**. It's confusing and easy to misclassify. These
are exactly the points the model struggles with.

**ADASYN adaptively generates more synthetic samples in the difficult,
boundary regions and fewer in the safe, easy regions.**

That word **adaptive** is the whole name — the amount of synthesis *adapts* to
the local difficulty of each minority point.

| | SMOTE | ADASYN |
|---------------|--------------------------|-----------------------------------|
| How many new points per minority sample? | Same for all | More where learning is hard |
| Focus | Uniform coverage | Decision boundary / hard cases |
| Driven by | Just minority neighbors | Ratio of majority neighbors |

---

## 4. The Steps of ADASYN (Conceptually + Math)

Let:
- `m_minority` = number of minority samples
- `m_majority` = number of majority samples

### Step 1 — Decide how many samples to create in total
```
G = (m_majority − m_minority) × β
```
`β` (beta) is between 0 and 1. `β = 1` means "fully balance the classes."
`G` is the total number of synthetic points we will generate.

### Step 2 — Measure difficulty for each minority point
For **each** minority patient `xᵢ`:
- Find its **k nearest neighbors** (e.g. k = 5) among *all* patients.
- Count how many of those neighbors belong to the **majority** class → `Δᵢ`.
- Compute the **difficulty ratio**:
```
rᵢ = Δᵢ / k
```
A high `rᵢ` (e.g. 5/5 = 1.0) means "this sick patient is drowning in healthy
neighbors" → very hard, near the boundary. A low `rᵢ` (0/5 = 0) means "safe,
deep inside minority territory."

### Step 3 — Normalize the ratios into a distribution
```
r̂ᵢ = rᵢ / Σ rᵢ        (so all r̂ᵢ add up to 1)
```
This turns difficulty into a **probability weight**.

### Step 4 — Allocate synthetic points by difficulty
```
gᵢ = r̂ᵢ × G
```
Hard points get a **big** `gᵢ` (many new neighbors created around them);
easy points get a small `gᵢ`. This is the adaptive magic.

### Step 5 — Generate the synthetic samples (SMOTE-style)
For each minority point `xᵢ`, create `gᵢ` new points by interpolating toward a
randomly chosen **minority** neighbor:
```
x_new = xᵢ + random(0,1) × (x_neighbor − xᵢ)
```

**Result:** a balanced dataset, with the *new* patients concentrated in the
tricky boundary zones where the model needs the most help.

---

## 5. A Real-Life Analogy

Think of a **teacher preparing students for an exam.**

- **Random oversampling** = photocopy the same practice question 10 times.
  Students just memorize it. (Overfitting.)
- **SMOTE** = create *new* practice questions evenly across every topic,
  including topics students already ace. (Wasted effort.)
- **ADASYN** = the teacher notices which topics students keep failing and
  **writes extra practice questions specifically for those hard topics.**
  Effort goes where it's needed most. (Adaptive.)

Another example — **fraud detection**: 99.9% of transactions are legitimate.
ADASYN generates extra synthetic *fraud* examples, focusing on the ambiguous
transactions that sit right on the edge between "normal" and "fraud" — the ones
the model keeps getting wrong.

For **your Alzheimer's project**: ADASYN would create the most synthetic
"Alzheimer's" patients around those borderline cases whose feature values
(brain volume, cognitive scores, biomarkers) look confusingly similar to
healthy patients — the diagnostically hardest cases.

---

## 6. How to Apply It in Python

ADASYN lives in the `imbalanced-learn` library (works with scikit-learn).

```bash
pip install imbalanced-learn
```

```python
from imblearn.over_sampling import ADASYN
from collections import Counter

# X = your features, y = your labels (0 = Healthy, 1 = Alzheimer's)
print("Before:", Counter(y))   # e.g. {0: 900, 1: 100}

adasyn = ADASYN(
    sampling_strategy='auto',  # balance minority up to majority (β = 1)
    n_neighbors=5,             # the 'k' in k-nearest-neighbors
    random_state=42
)

X_resampled, y_resampled = adasyn.fit_resample(X, y)

print("After:", Counter(y_resampled))  # e.g. {0: 900, 1: ~900}
```

### CRITICAL RULES (do not skip)

1. **Only resample the TRAINING set — never the test/validation set.**
   Split first, then apply ADASYN only to training data. Synthetic points in
   your test set would give a fake, inflated accuracy.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42)
   X_train, y_train = adasyn.fit_resample(X_train, y_train)  # train only!
   ```

2. **Scale your numeric features first** (e.g. `StandardScaler`). ADASYN uses
   distance (nearest neighbors), so features on huge scales dominate unfairly.

3. **Encode categorical variables** — plain ADASYN assumes continuous numeric
   features. For mixed data, use `SMOTENC` instead.

---

## 7. Limitations of ADASYN (Know the Downsides)

- **Sensitive to outliers / noise.** A mislabeled minority point sitting deep in
  majority territory looks "very hard," so ADASYN pours synthetic points around
  a *noisy* location — amplifying the error.
- **Can blur the decision boundary.** By packing synthetic points right on the
  boundary, classes can start to overlap.
- **Struggles in high dimensions** — nearest-neighbor distances become less
  meaningful when you have very many features (the "curse of dimensionality").
- **Only for numeric features** in its basic form.

---

## 8. Alternatives to ADASYN

| Method | One-line idea | When to prefer it |
|--------------------------|----------------------------------------------------------|-------------------|
| **Random Oversampling** | Duplicate minority points | Tiny/simple datasets, quick baseline |
| **Random Undersampling** | Drop majority points | You have LOTS of data to spare |
| **SMOTE** | Interpolate evenly between minority neighbors | Clean, well-separated classes |
| **Borderline-SMOTE** | Like SMOTE but only near the boundary | Focus on hard cases, less noise-prone than ADASYN |
| **SMOTENC / SMOTE-N** | SMOTE for mixed categorical + numeric data | You have categorical features |
| **SMOTE + Tomek Links** | Oversample, then clean overlapping pairs | Reduce boundary overlap |
| **SMOTE + ENN** | Oversample, then remove misclassified noise | Noisy datasets |
| **Class weights** | Tell the model to penalize minority errors more | No synthetic data at all (`class_weight='balanced'`) |

**Rule of thumb:** Try **SMOTE** as your baseline. If your minority class is
hard and boundary-heavy, try **ADASYN** or **Borderline-SMOTE**. Always compare
against simple **class weighting** — sometimes it wins with far less complexity.

---

## 9. Measuring Success (Don't Use Accuracy!)

After resampling, **never** judge with plain accuracy. Use metrics that respect
the rare class:

- **Recall / Sensitivity** — of all true Alzheimer's patients, how many did we catch?
- **Precision** — of those we flagged, how many were truly sick?
- **F1-score** — harmonic mean of precision and recall.
- **ROC-AUC / PR-AUC** — threshold-independent quality; PR-AUC is best for imbalance.
- **Confusion matrix** — always look at it directly.

---

## 10. One-Paragraph Summary

**ADASYN is an intelligent oversampling technique that fixes class imbalance by
creating synthetic minority examples — but instead of spreading them evenly like
SMOTE, it concentrates them around the *hard-to-learn* minority points that sit
near the decision boundary (measured by how many majority neighbors surround
each minority point). This forces the model to pay extra attention exactly where
it tends to fail. Apply it only to your training set, scale your features first,
and always evaluate with recall/F1/PR-AUC rather than accuracy.**
```
