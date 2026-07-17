# Understanding train_kaggle.ipynb (for absolute beginners)

This guide explains what the notebook does, in plain words. No prior machine
learning knowledge needed. Read it top to bottom, like a story.

## The big picture first

We have brain MRI scans from **347 people**. Some are healthy, some have early
Alzheimer's, some have more advanced Alzheimer's. We want to build a computer
program that can look at a brain scan and guess which group the person is in.

The way a computer "learns" this is like a student studying for an exam:
1. We **show it examples** with the answers (this scan = healthy, this one = sick).
2. It **finds patterns** on its own.
3. We **test it on scans it has never seen** to check if it really learned, or
   if it just memorized the answers.

That last point is the whole game. If a student memorizes the exact practice
questions instead of understanding the topic, they ace practice but fail the real
exam. Our biggest job is to stop the computer from "cheating" in that way.

Now let's walk through the notebook, one step at a time.

---

## Step 1 — Settings (the "Config")

**What it is:** A list of all our choices written in one place at the top — things
like how big to make the images, how fast the computer should learn, and so on.

**Why we do it:** Imagine a recipe. You want all the ingredients and amounts
listed at the top, not hidden randomly through the instructions. If you want to
change something, you change it once, in one place.

**One important setting: the "seed."**
Computers use randomness (for example, to shuffle the data). A "seed" is like
telling the dice to always roll the same sequence. We fix it so that if we run the
notebook again tomorrow, we get the **exact same results**.

**Why that matters:** With only 347 people, results jump around a lot by luck. If
we didn't fix the seed, we could never tell whether a change actually helped or we
just got lucky. Same seed = fair comparison.

---

## Step 2 — Splitting the data (the most important step)

**What it is:** We divide our 347 people into groups: most are used for
**learning**, and some are locked away for the **final test**.

**The key rule: split by PERSON, not by picture.**

Here is the trap. Each person's brain was scanned and sliced into ~240 pictures.
All 240 pictures of one person's brain look almost identical — same skull shape,
same head size.

Now imagine we shuffled all the pictures randomly and put some of a person's
pictures in the "study" pile and the rest in the "test" pile. The computer would
see that person's skull during studying, then recognize the **same skull** during
the test and get it right — not because it understands Alzheimer's, but because it
recognized the head. That's cheating, and it produces fake sky-high scores.

**So the rule is:** all pictures of one person go **together** — either all in
studying, or all in testing. Never split one person across both. This is called
**splitting by subject** (subject = person).

**Why this is the #1 thing:** Most published Alzheimer papers using this dataset
got this wrong and reported fake "98–99% accuracy." Getting it right is exactly
what makes our work honest and trustworthy.

**A small merge:** One group ("Moderate" dementia) has only 2 people. Two people
is far too few to both study and test on, so we combine them with the "Mild" group.
That leaves us three honest groups: **healthy, very mild, and mild-or-worse.**

**Why we rebuild the split here (not reuse an old file):** The saved file has
Windows file paths (like `E:\...`) that don't exist on Kaggle's computers. So we
recreate the split fresh on Kaggle. Because we use the same seed, it comes out
**identical** to before — we checked, and not a single person landed in a different
group.

---

## Step 3 — Preparing the pictures

**What it is:** Before the computer sees a scan, we clean it up so every picture
is in the same format.

**Why:** The computer program we use was originally trained on ordinary color
photos (cats, cars, etc.). It expects pictures of a certain size and style. MRI
scans are gray and come in different sizes, so we adjust them to match what the
program expects: resize them all to the same size, and copy the gray shade into
the three color channels (red, green, blue) so it looks like a "color" image.

**A trick called "augmentation":**
Only for the study pile, we make small random changes to each picture every time —
rotate it slightly, flip it left-right, brighten it a touch.

**Why:** It's like showing a student the same flashcard held at slightly different
angles. It stops them from memorizing one exact image and forces them to focus on
the real content. This makes the computer more robust.

**Why only small changes, and only left-right flips:** A real brain scan is never
upside-down or sideways in the machine. If we flipped scans upside-down, we'd be
teaching the computer about brains that will never actually appear — useless and
confusing. Left-right is fine because brains are roughly symmetric.

**Important honesty note:** Augmentation does **not** give us more people. We still
only have 347. It just helps the computer stop memorizing exact pixels. It cannot
magically fix the fact that our dataset is small.

**And we never augment the test pile** — we must grade the computer on real,
untouched scans, the way they'd actually appear.

---

## Step 4 — The "brain" of the program (the model)

**What it is:** The actual program that looks at a picture and outputs a guess.

**The big idea: don't start from zero — borrow.**
Teaching a computer to "see" from scratch takes millions of pictures. We only have
scans from 347 people. So instead, we take a program that **already** learned to
see general shapes, edges, and textures from millions of ordinary photos, and we
just **re-teach its final decision step** for our brain-scan task.

This borrowing is called **transfer learning**. Think of hiring someone who already
knows how to read and just teaching them the medical vocabulary, instead of
teaching a baby to read from birth.

**How:** We keep the experienced "vision" part of the program and replace only the
last piece — the part that makes the final "healthy / very mild / mild-or-worse"
choice.

**Why we pick a SMALL program to start:** Bigger programs have more "memory" and
are more tempted to memorize our 347 people instead of learning real patterns. A
smaller program is forced to learn the general idea. We can always try a bigger one
later and compare.

---

## Step 5 — How we grade the computer

**What it is:** The rules for measuring how good the computer's guesses are.

**Rule 1: grade per PERSON, not per picture.**
Each person has many pictures. The computer guesses on each picture, then we
combine all of one person's picture-guesses into a single guess for that person
(basically, a vote). What we care about is: did we get the **person** right? That's
what matters in a clinic.

**Rule 2: don't use plain "accuracy."**
Here's the trap: out of 347 people, about 266 are healthy. So a lazy program that
**always says "healthy"** would be right about **77% of the time** — while being
completely useless, because it never catches a single sick person!

So "77% accuracy" sounds impressive but means nothing here. Instead we use a
fairer score called **macro-F1**. The important thing to know: **macro-F1 checks
how well we do on *each group separately* and averages them.** If the computer
ignores the rare sick groups, macro-F1 drops. That forces the computer to actually
be good at spotting dementia, not just at saying "healthy."

**We also handle the imbalance during learning** by telling the computer "getting
a rare sick person right is worth more" — so it pays attention to them instead of
lazily favoring the big healthy group.

---

## Step 6 — Actually teaching the computer (training)

We teach it in **two stages**, gently.

**Stage 1 — freeze most of it, train only the new part.**
Remember, we replaced the last decision step with a brand-new, untrained one. That
new part starts out clueless and makes wild mistakes. If we let those wild mistakes
ripple back into the experienced "vision" part, we'd damage the good knowledge it
came with. So first we **lock** the experienced part and only train the new step.

**Stage 2 — unlock everything, and fine-tune very gently.**
Once the new step is reasonable, we unlock the whole program and let it adjust
everything together — but very **slowly** (a small "learning rate"), so we polish
the existing knowledge instead of wrecking it.

*(Learning rate = how big a step the computer takes when it corrects itself. Big
steps = fast but reckless; small steps = slow but careful. We use tiny steps in
stage 2 on purpose.)*

**Stopping at the right time ("early stopping").**
If a student keeps re-reading the same practice test forever, they eventually just
memorize it. Same with our computer — train too long and it starts memorizing the
study group. So after each round we check its score on a held-aside "validation"
group, and we **keep the version that scored best**, not the last one.

---

## Step 7 — Reading the results

**What it is:** The final report card.

**Always compare to the "lazy baseline."** We remind ourselves that always-guess-
healthy scores 77%. So our real question is: **did we beat the lazy program on
macro-F1?** If yes, we genuinely learned something.

**We report an average, with a "give or take."** We train several times on
different slices of the data and report something like "macro-F1 = 0.65, give or
take 0.05." The "give or take" (standard deviation) tells us how stable — and
therefore how trustworthy — the result is.

**We also show a "confusion matrix"** — a simple grid showing which groups get
mixed up with which. For example, it might reveal the computer often confuses
"healthy" with "very mild." A single score can't tell you that; the grid can.

---

## The three golden rules (if you remember nothing else)

1. **Keep each person's scans together** — split by person, never by picture.
   Otherwise the computer cheats and every result is a lie.
2. **Grade per person, using macro-F1, and compare to the 77% lazy baseline.**
   Plain accuracy is a trap when most people are healthy.
3. **Only touch the locked test group once, at the very end.** If you peek and
   tweak, it stops being a fair test.
