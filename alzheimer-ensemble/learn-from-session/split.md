# Splitting Data — Train / Validation / Test (A Learner's Guide)

> Read this **before** ADASYN. Splitting is the very first thing you do to your
> data, and if you get it wrong every number you report afterwards is fiction —
> including any ADASYN result. This guide is grounded in *your* dataset (the
> OASIS MRI 4-class task), with the real subject counts baked in.

---

## 0. The One Sentence to Remember

> **Split first, split by patient, split once — and never let evaluation data
> (or even the knowledge of which patient it came from) touch training.**

Everything below is just that sentence, unpacked and justified.

---

## 1. Why Split At All? The Exam Analogy

You are a teacher. You want to know if a student *understands* the material —
not whether they *memorized the answer key*.

- If you test students on the **exact questions you handed out as homework**,
  a student who memorized the homework scores 100% and learns nothing. The score
  is a lie.
- To measure real understanding, you test on **fresh questions they have never
  seen**.

Machine learning has the identical problem. A model with millions of parameters
can *memorize* its training images. If you then evaluate on those same images,
you measure memorization, not learning. So you **hold out** data the model never
trains on, and judge it there. That held-out score is your honest estimate of
how the model behaves on the next, unseen patient — which is the only thing that
matters clinically.

This is the whole reason splitting exists: **to get an honest estimate of
generalization.**

---

## 2. The Three Splits and Their Jobs

We carve the data into three disjoint piles, each with a *different job*:

| Split | Job | How often you touch it |
|---|---|---|
| **Train** | The model learns its weights from these. | Constantly — every epoch. |
| **Validation** | You *tune* on these: early stopping, learning rate, choosing the best epoch, and (in your project) computing **ensemble weights**. You never train weights on them. | Often — after each epoch. |
| **Test** | The final, honest report card. | **Exactly once**, at the very end. |

**Why not just two piles (train + test)?**

Because the moment you use a pile to *make decisions* — "stop training here",
"this learning rate is best", "weight ResNet 0.4 and DenseNet 0.6" — that pile
has *influenced* your model. It's no longer a neutral judge. Information from it
has leaked into your choices.

So:
- **Validation** is the pile you're *allowed* to peek at repeatedly to make
  decisions. It gets "used up" — you slowly overfit to it through your choices.
- **Test** is sacred. You open it **once**, take the number, and report it. If
  you look at the test score, then go "hmm let me tweak the model", you've just
  converted your test set into a validation set, and you no longer have an honest
  final number. This discipline is not bureaucracy; it's the difference between
  science and self-deception.

A common split ratio — and the one locked for your project — is **80 / 10 / 10**.

---

## 3. The Two Enemies of a Good Split

A split can fail in exactly two ways. Almost everything else in this guide is a
tool against one of these two enemies.

### Enemy #1 — The Unrepresentative Split
The piles don't look like each other (or like reality). Example: by bad luck,
almost all the rare "Moderate" cases land in train and almost none in test. Now
your test set barely measures the class you care most about, and its score is
noisy and misleading. → **Fixed by *stratification* (Section 4).**

### Enemy #2 — Leakage
Information secretly crosses from an evaluation pile into training, inflating
your score. The score looks great and is completely fake. Leakage is the silent
killer of medical ML papers. → **Fixed by *grouping by patient* (Section 5) and
by *ordering operations correctly* (Section 7).**

---

## 4. Stratification — Keep the Class Ratios Identical

**Stratified** splitting means each pile keeps the *same class proportions* as
the full dataset.

Your dataset is wildly imbalanced:

| Class | Slices | Share |
|---|---|---|
| Non Demented | 67,222 | 77.8% |
| Very mild Dementia | 13,725 | 15.9% |
| Mild Dementia | 5,002 | 5.8% |
| Moderate Dementia | 488 | 0.56% |

That's a **~138× imbalance** between the biggest and smallest class.

Now imagine a *naive random* split. Moderate has only 488 slices. Pure chance
could dump 480 of them into train and leave 8 in test. Your test "Moderate
recall" would then be computed on a handful of images — statistically
meaningless, and it swings violently with a single misclassification.

Stratification says: *"Whatever fraction each class is overall, force that same
fraction into train, into val, and into test."* So test gets ~0.56% Moderate,
just like the whole dataset. The split becomes **representative** (defeats Enemy
#1).

> Note: stratifying keeps the split *representative of reality* — it does **not**
> balance the classes. Your test set stays imbalanced **on purpose**, because
> the real world is imbalanced and your report card must reflect real conditions.
> Balancing is a *training-time* trick (that's ADASYN's job, Section 7), never a
> test-time one.

---

## 5. Grouping by Patient — The Leakage That Kills Medical ML

This is the single most important, most dataset-specific idea in this guide.
Read it twice.

### The trap hiding in your filenames

Look at three real files from your `Non Demented` folder:

```
OAS1_0001_MR1_mpr-1_100.jpg
OAS1_0001_MR1_mpr-1_101.jpg
OAS1_0001_MR1_mpr-1_102.jpg
```

These are **not three different people.** They are three 2D slices of the
**same brain** — patient `OAS1_0001`. Your dataset isn't 86,437 independent
patients; it's a few hundred patients, each sliced into hundreds of nearly
identical images.

Here are the true counts for your data (slices vs *distinct patients*):

| Class | Slices | **Distinct patients (subjects)** |
|---|---|---|
| Non Demented | 67,222 | **266** |
| Very mild Dementia | 13,725 | **58** |
| Mild Dementia | 5,002 | **21** |
| Moderate Dementia | 488 | **2** |
| **Total** | **86,437** | **347** |

So the "big" dataset is really **347 people.**

### Why slice-level splitting silently cheats

Suppose you split by *slice* (the naive way). Slice #100 of patient `OAS1_0001`
lands in train; slice #101 of the *same patient* lands in test. Those two images
are almost identical — same skull shape, same ventricle size, same scanner
artifacts, taken 5mm apart.

The model doesn't need to learn "what does dementia look like." It can learn
"I've seen this exact brain before, and it was Non Demented." At test time it
recognizes the brain, not the disease. Your test accuracy rockets to 99% — and
means **nothing**, because in the clinic you will only ever see *brand-new*
patients whose brains the model has never seen.

This is **data leakage via patient identity.** It is the classic way medical
imaging papers accidentally report fake, unpublishably-high numbers.

### The fix: group by subject

> **Split over *patients*, not over *slices*. Every slice of a patient moves
> into one pile, together, as an unbreakable unit.**

Because in OASIS each subject has a single diagnosis (verified: the 347 subjects
partition cleanly across the four classes with zero overlap), a patient maps to
exactly one class. So you can *group by patient* and *stratify by class* at the
same time.

Concretely: extract the `OAS1_XXXX` token from each filename (your
`src/data.py` already has `parse_subject_id` for exactly this), build the list of
347 unique subjects with their class, split the **subjects**, then expand each
subject back into all its slices.

---

## 6. The Collision — When "Stratify" and "Group" Fight, and Your Dataset Loses

Here is where theory meets your specific, unforgiving data. This section is the
reason the guide is grounded in real numbers instead of a tidy textbook example.

You want two things at once for every class:
1. **Grouping:** all of a patient's slices stay together in one pile.
2. **Three-way coverage:** the class appears in train *and* val *and* test.

To have a class present in all three piles under grouping, you need **at least 3
distinct patients** in that class (one can go to each pile). Ideally many more,
so an 80/10/10 patient split is even meaningful.

Now look at your patient counts again:

| Class | Patients | Can it cover train/val/test by patient? |
|---|---|---|
| Non Demented | 266 | ✅ Easily — 80/10/10 ≈ 213 / 27 / 26 patients |
| Very mild Dementia | 58 | ✅ Yes — ≈ 46 / 6 / 6 patients |
| Mild Dementia | 21 | ⚠️ Barely — ≈ 17 / 2 / 2 patients (val/test tiny) |
| **Moderate Dementia** | **2** | ❌ **Impossible** |

**Moderate Dementia has only 2 patients** (`OAS1_0308` and `OAS1_0351`, 244
slices each). You cannot place 2 people into 3 piles while keeping each person
whole. One of train/val/test *must* get zero Moderate patients. This is not a
coding problem you can clever your way out of — it's a hard mathematical fact
about the data you were given.

**You cannot simultaneously have all three of:** (a) grouping by patient,
(b) Moderate present in every split, and (c) no patient in two splits. Pick which
one to relax, and *document it honestly*.

### The honest options (choose and write it down)

There is no free lunch. Each option trades something:

1. **Grouped, accept Moderate is absent from one split (recommended default).**
   Put the 2 Moderate patients into train and test (e.g. `OAS1_0308` → train,
   `OAS1_0351` → test), and accept that **validation has no Moderate**. Keep
   grouping intact for *every* class. You report: "Moderate could not be
   represented in validation due to having only 2 subjects." This preserves the
   thing that matters most — **no leakage** — and is the most defensible choice.

2. **Slice-level split for Moderate only.** Split Moderate's 488 slices
   80/10/10 by slice, group all other classes by patient. This gives Moderate
   coverage in all three piles **but reintroduces patient leakage inside the
   Moderate class** (adjacent slices of the same brain in train and test). If you
   do this, you must say so plainly — you've knowingly leaked one class for
   coverage.

3. **Drop or merge Moderate.** Collapse the task to 3 classes, or merge Moderate
   into Mild ("Mild+"). Cleanest statistically, but you lose the most clinically
   severe class — usually undesirable for an Alzheimer's staging paper.

4. **Report it as a known limitation regardless.** Whatever you pick, Moderate's
   numbers rest on 2 people. Even a perfect split can't make 2 patients
   statistically robust. This *must* appear in your limitations section.

> This is exactly the "rare-subject caveat" your Phase 3 notes flagged. The
> point of a good scientist isn't to make the problem disappear — it's to make
> the compromise **visible, principled, and reproducible.**

---

## 7. Ordering — Why the Split Comes *Before* Everything Else

The sequence of operations is itself a correctness question. Get the order wrong
and you leak, even with a perfect patient-grouped split.

```
  1. SPLIT           ← by patient, stratified, once. (this guide)
  2. Fit preprocessing on TRAIN ONLY   (e.g. any scaler/statistics)
  3. ADASYN on TRAIN ONLY              (balance the training set)
  4. Train the model
  5. Tune / early-stop / ensemble-weight on VAL
  6. Open TEST once, report
```

Two rules people violate constantly:

### Rule A — ADASYN (and all resampling) happens *after* the split, on *train
only.*

ADASYN invents *synthetic* minority samples. If you resample **before**
splitting, synthetic points — which are built by interpolating between real
points — can land in your test set, or a synthetic point in test can be derived
from a real point in train. Your test set is now polluted with fabricated data
derived from training data. The score inflates and lies.

> **Never oversample the validation or test set.** They must stay real and
> imbalanced, reflecting reality. Only the *training* pile gets balanced, because
> balancing is a *learning aid*, not a measurement.

### Rule B — Fit any statistic (scalers, PCA, normalization constants) on train
only.

If you compute, say, a mean/std for normalization over the *whole* dataset
before splitting, the test set's values have influenced a number the model uses —
subtle leakage. Compute such statistics on **train**, then *apply* them to
val/test. (Your image pipeline uses fixed ImageNet stats, so this is less of an
issue here, but the principle is universal and will bite you the moment you add
any data-derived preprocessing.)

**The unifying principle:** *the split is a one-way information barrier. Nothing
learned from — or synthesized from — train may be measured on val/test, and
nothing from val/test may flow back into train.*

---

## 8. Split Once, Save to Disk, Reuse Everywhere

You have **seven notebooks** (EDA, ADASYN check, three training notebooks,
ensemble eval, XAI). If each one re-runs the split, even with the same
`random_state`, one subtle difference (a different library version, a reordered
file list) produces a *different* split — and now your ResNet trained on one
partition while your DenseNet tested on another. That's silent leakage across
your own ensemble.

**Do it once. Persist it. Everyone reads the same file.**

- Split at one place (`src/split.py` → save to `data/splits/`).
- Save the **subject→split assignment** (which of the 347 patients is train/val/
  test) *and* the expanded per-split file lists.
- Every notebook loads that file. Nobody re-splits.
- Use a fixed `random_state=42` so the split itself is reproducible from scratch.

Saving the *subject* assignment (not just the slice lists) is what makes your
split auditable — anyone can re-open it and confirm no patient crossed a
boundary.

---

## 9. The Concrete Recipe (with scikit-learn)

The tool that does grouping + stratification together is
`StratifiedGroupKFold`: it keeps class ratios balanced across folds *while*
guaranteeing no group (patient) spans two folds.

```python
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from src.data import parse_subject_id   # your existing OAS1_XXXX extractor

# --- 1. Gather (file, label, subject) for every slice -----------------------
CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
files, labels, groups = [], [], []
root = Path("datasets/Data")
for label_idx, cls in enumerate(CLASSES):
    for f in sorted((root / cls).glob("*.jpg")):
        files.append(str(f))
        labels.append(label_idx)
        groups.append(parse_subject_id(f.name))   # e.g. "OAS1_0001"

files  = np.array(files)
labels = np.array(labels)
groups = np.array(groups)

# --- 2. Carve TEST off first (10%), grouped + stratified --------------------
# 10-fold => each fold ~10%. Take ONE fold as test.
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
trainval_idx, test_idx = next(sgkf.split(files, labels, groups))

# --- 3. Carve VAL off the remainder (~1/9 of trainval ≈ 10% of total) -------
tv_files, tv_labels, tv_groups = files[trainval_idx], labels[trainval_idx], groups[trainval_idx]
sgkf2 = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
train_rel, val_rel = next(sgkf2.split(tv_files, tv_labels, tv_groups))

train_idx = trainval_idx[train_rel]
val_idx   = trainval_idx[val_rel]
```

> ⚠️ **Reality check on Moderate:** with only 2 Moderate patients,
> `StratifiedGroupKFold` **cannot** place Moderate in all three splits — it will
> land in at most two. This is expected (Section 6). After splitting, *verify*
> where Moderate went and handle it via your chosen option from Section 6 (e.g.
> manually force `OAS1_0308`→train, `OAS1_0351`→test). Do not let the tool decide
> silently; decide deliberately and log it.

### Then: verify, then save

```python
from collections import Counter

for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    print(name, "class counts:", Counter(labels[idx].tolist()))

# THE leakage check — must all be empty:
s_train, s_val, s_test = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
assert not (s_train & s_val),  f"leak train/val:  {s_train & s_val}"
assert not (s_train & s_test), f"leak train/test: {s_train & s_test}"
assert not (s_val   & s_test), f"leak val/test:   {s_val & s_test}"
print("No patient appears in two splits. ✅")

# Save subject assignment + file lists so all 7 notebooks reuse THIS split.
out = Path("data/splits"); out.mkdir(parents=True, exist_ok=True)
np.savez(out / "split_seed42.npz",
         train_files=files[train_idx], train_labels=labels[train_idx],
         val_files=files[val_idx],     val_labels=labels[val_idx],
         test_files=files[test_idx],   test_labels=labels[test_idx])
```

---

## 10. Common Mistakes (a checklist of ways to fool yourself)

| Mistake | What breaks | Fix |
|---|---|---|
| Splitting by slice, not patient | Massive leakage; fake ~99% accuracy | Group by `OAS1_XXXX` |
| Random (non-stratified) split | Rare classes vanish from test; noisy metrics | Stratify by class |
| ADASYN before splitting | Synthetic data leaks into val/test | Split first, ADASYN on train only |
| Oversampling val/test | Fake balanced test; inflated scores | Keep val/test real & imbalanced |
| Fitting a scaler on all data | Test stats leak into training | Fit on train, apply to val/test |
| Re-splitting in each notebook | Models train/test on different partitions | Split once, save, reuse |
| Peeking at test, then tweaking | Test becomes a second val set; no honest number | Open test exactly once |
| Ignoring the 2-patient Moderate problem | Silently leaks or misreports the key class | Decide + document (Section 6) |

---

## 11. One-Paragraph Summary

**Splitting exists to get an honest estimate of how your model handles *unseen*
patients, so you divide data into train (learn), validation (tune/early-stop/
ensemble-weight), and test (report once). Two enemies threaten this: an
unrepresentative split — beaten by *stratifying* on class so every pile keeps the
real ~138× imbalance — and leakage — beaten by *grouping on patient* so all of a
subject's ~240 near-identical MRI slices move together, never straddling two
piles. Your OASIS data is really only 347 patients, and one class (Moderate) has
just 2, which makes a strict grouped 80/10/10 impossible for it; you resolve that
by a deliberate, documented compromise rather than a silent slice-level cheat.
Do the split first, once, saved to disk with `random_state=42`, and only then —
on the training pile alone — apply ADASYN. The whole discipline is one rule: the
split is a one-way information barrier, and nothing may cross it backwards.**

---

### Where this sits in your pipeline

```
EDA (subjects, classes)  →  [ THIS STEP: split by patient ]  →  ADASYN (train only)  →  train 3 backbones  →  ensemble (val weights)  →  test once  →  XAI
```

> **Next:** `adasyn.md` — how to intelligently balance the *training* pile you
> just carved out (and only that pile).
