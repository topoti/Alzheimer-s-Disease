# Phase 1 · Step 1 — Frame the Problem Correctly

> **Phase 1 — Research Foundation** (no code yet — pure understanding)
> The goal of this step: be able to state *what problem you're solving* and *what makes it hard* — because every later decision (ADASYN, the 3-network ensemble, ignoring "accuracy") only makes sense once this clicks.

The one question this step answers:

> **What problem am I actually solving, and what makes it hard?**

---

## 1. The problem, stated like an ML engineer

> *Given a 2D MRI brain-slice image, predict which one of 4 Alzheimer's severity stages it shows.*

Every word matters:

- **Supervised** — You have 6,400 MRI images, each already labeled with the correct answer by an expert. The model learns from these (image → label) pairs so it can label *new, unseen* images. "Supervised" = *learning from labeled examples*.
- **Image classification** — Input = a picture; output = a single label naming what's in it. (Not *detection*, which draws boxes; not *segmentation*, which colors every pixel.)
- **Multi-class (4-class)** — Each image belongs to exactly one of four mutually exclusive buckets.
- **Probability output, not a single guess** — The model outputs four numbers that sum to 1, e.g. `[0.05, 0.10, 0.70, 0.15]` ("5% Non, 10% Very Mild, 70% Mild, 15% Moderate"). We take the largest (`argmax`) → "Mild." The function that squashes raw model outputs into this clean probability vector is **softmax**. This probability vector is what later makes the ensemble and XAI possible.
- **Ordinal flavor (keep in your back pocket)** — The classes have a natural order: healthy → very mild → mild → moderate. We *treat* it as plain multi-class (matching the reference paper), but **not all mistakes are equal**: confusing "Mild" vs "Very Mild" is minor; confusing "Moderate" vs "Non-Dementia" is catastrophic (a sick patient sent home). This idea returns in Phase 6 (confusion matrix).

### The four classes

| Class | Meaning | # images |
|---|---|---|
| Non-Dementia | healthy brain | 3,200 |
| Very Mild Dementia | earliest decline | 2,240 |
| Mild Dementia | clear symptoms | 896 |
| Moderate Dementia | significant impairment | 64 |
| **Total** | | **6,400** |

---

## 2. The fact that drives the entire project: **class imbalance**

The biggest class has **50× more examples** than the smallest (3,200 ÷ 64). That severe imbalance is the villain this whole research is built to defeat.

**Why it's dangerous:** a model is trained to make overall error small. If half the images are "Non-Dementia," the model finds a lazy shortcut — *lean toward common classes, basically never predict "Moderate."* It scores high overall while being **blind to the rarest, most medically urgent class.** That's the default behavior, not a hypothetical.

**Analogy (memorize this):** teaching a child with 3,200 dog photos, 2,240 cat photos, 896 rabbit photos, and only 64 hamster photos. The child gets great at dogs, decent at rabbits, and *never* says "hamster" — because guessing "dog" is usually right.

**This single fact is the root of three later decisions:**

1. **Pillar 1 — ADASYN** (Phase 3): smartly manufactures more minority-class training examples. It exists *only* because of this imbalance.
2. **We distrust "accuracy"** (Phase 6): we headline **macro-F1** and **per-class recall** — metrics that won't let the rare class hide.
3. **The confusion matrix** (Phase 6): exposes exactly which classes get confused.

> 🔑 **If you remember nothing else:** this is an **imbalanced 4-class medical image classification** problem, and **imbalance is the thread** running through every later decision.

---

## 3. What you have to learn (correct intuition, not heavy math yet)

1. **Supervised learning vocabulary** — features (X), labels (y), train/validation/test split (why we hold data back), and **overfitting** (memorizing vs. generalizing).
2. **Classification vs. regression**, and what **multi-class** means.
3. **Softmax & argmax** — output → probability vector → decision.
4. **Class imbalance** — why accuracy lies on imbalanced data; why we measure *per class*.

### Resources (a few hours total)

- **Google ML Crash Course** → *"Framing"* + *"Classification"* units — `developers.google.com/machine-learning/crash-course`
- **StatQuest with Josh Starmer** (YouTube) → *"Machine Learning Fundamentals"* and *"Confusion Matrix"*.
- **The paper you're beating** → Mahmud et al. (2024), *Diagnostics*. Read **Abstract + Introduction + results table** only. Know what accuracy they claim and on which classes.
- *(Optional intuition)* 3Blue1Brown *"Neural Networks"* series (YouTube), chapter 1.

---

## 4. Your task (this verifies Step 1 — do it before moving on)

Write these down (a notebook or text file — this becomes raw material for your paper's intro):

- [ ] **1. One paragraph, your own words:** state the problem — input, output, number of classes, and the imbalance. Don't copy the table; explain it like you're telling a friend.
- [ ] **2. Compute the imbalance ratios** relative to Moderate = 64:
  - Non-Dementia = 3200 / 64 = ?
  - Very Mild = 2240 / 64 = ?
  - Mild = 896 / 64 = ?
- [ ] **3. The "dumb baseline" question:** If a lazy model ALWAYS predicts "Non-Dementia," what accuracy does it score, and why is that number dangerously misleading? *(Hint: 3200 out of 6400.)*

✅ When you can answer #3 cleanly, you've understood the heart of this project — and you'll *feel* why ADASYN matters before we even define it.

---

## 5. When you're ready

Paste your answers to the 3 tasks (or say "done"), and we move to:

> **Phase 1 · Step 2 — Gap analysis: exactly how your 3 improvements beat the reference paper, starting with truly understanding ADASYN.**
