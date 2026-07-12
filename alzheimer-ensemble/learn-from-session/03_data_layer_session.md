# Lesson 03 — Building the Data Layer (What This Session Taught)

> A walkthrough of the session where we wrote `src/data.py` — the file that turns MRI files
> on disk into normalized tensors a model can train on. Framed as lessons: not just *what*
> the code does, but *why* each choice is the defensible one, including the biggest judgment
> call of the session (refusing to blindly overwrite existing code).

## Session in three acts

**Act 1 — Notice the file already existed, and wasn't what the prompt assumed.** The prompt
said "Write `src/data.py`" as if starting fresh. But `data.py` already held **475 lines** of
working code — including the plan's single most important piece, the patient-level split. A
blind overwrite would have silently deleted it. So the session *started* with investigation,
not typing.

**Act 2 — Read the project's own script to resolve the conflict.** Rather than guess, I read
`CLI_PROMPTS.md` and found the answer: the project *intends* to separate `data.py` (this
prompt) from `src/split.py` (the next prompt). The existing file was an older **monolith**
that hadn't been split apart yet.

**Act 3 — Write the focused data layer.** Only then did I write the three requested pieces:
`parse_subject_id`, `OasisDataset`, and `get_transforms` — using `albumentations`, with the
split logic left to be re-homed in `split.py`.

---

## The concepts worth actually learning

### 1. The most important lesson: look before you overwrite
The prompt implied a blank file. Reality had 475 lines of critical code. The right move was
**not** to follow the instruction literally — it was to *stop and check* whether obeying
would destroy something valuable.

This is a general discipline, not a one-off: **when an instruction assumes a starting state
that doesn't match reality, surface the mismatch before acting.** Two things made the
overwrite safe to proceed with afterward:
- The project's own `CLI_PROMPTS.md` showed the split logic has a *designated new home*
  (`src/split.py`, the very next prompt) — so it's being *relocated*, not deleted.
- The old code was already committed to git, so it's recoverable (`git show HEAD:...`).

If neither had been true, the correct action would have been to *ask you first*, not
overwrite. Lesson: literal obedience that destroys work is a failure, not compliance.

### 2. Separation of concerns — why `data.py` and `split.py` are different files
The old monolith did *everything*: parsing, splitting, saving, transforms, datasets,
dataloaders. That's convenient to write and painful to maintain. The intended architecture
draws clean lines:

| File | Owns | Runs where |
|---|---|---|
| `src/data.py` | filename parsing, `Dataset`, transforms | anywhere |
| `src/split.py` | *which* subjects go to train/val/test | locally (no GPU) |
| training notebook | wiring datasets + split into dataloaders | Kaggle GPU |

Why bother? Because each file now has **one reason to change**. If you tweak augmentation,
you touch `data.py` and nothing else. If you change the split ratio, you touch `split.py`
and nothing else. A reviewer (or future-you) can understand one file without reading all of
them. This is the *single responsibility principle* in practice.

### 3. What a PyTorch `Dataset` actually is
`OasisDataset` implements the two methods PyTorch's data machinery requires:
- `__len__(self)` → how many samples exist.
- `__getitem__(self, idx)` → return the `idx`-th sample as `(image_tensor, label)`.

That's the entire contract. Once you implement those two methods, a `DataLoader` can batch,
shuffle, and parallelize your data for free. The `Dataset` is a *lazy* recipe: it doesn't
load all 86k images into memory — it loads one image only when asked for that index. This is
why you can train on datasets far larger than your RAM.

### 4. The preprocessing pipeline, step by step
Every image goes through the same Phase 3 pipeline before it reaches the model:

1. **Load** the slice from disk.
2. **Grayscale → 3-channel duplicate.** MRIs are single-channel gray, but ImageNet-pretrained
   backbones expect 3-channel RGB. We copy the gray values into all three channels
   (`np.stack([gray, gray, gray])`). The image *looks* identical; it just has the shape the
   network demands.
3. **Resize** to `input_size` (224 or 300 depending on the backbone).
4. **ImageNet normalize.** Subtract mean `[0.485, 0.456, 0.406]`, divide by std
   `[0.229, 0.224, 0.225]` — the *exact* statistics the pretrained weights were trained
   under. Skip this and the pretrained features misfire, because the input distribution no
   longer matches what the network expects.

### 5. Why augmentation happens *only* when `train=True`
Augmentation (rotation ±10°, horizontal flip, brightness ±10%) is applied **only to training
data**. This is not a stylistic choice — it's a correctness rule:

- **Training:** random perturbations each epoch make the model robust; it never sees the
  exact same image twice, which fights overfitting.
- **Validation/test:** must be *deterministic*. If you randomly rotate a test image, your
  reported accuracy becomes a dice roll and isn't reproducible or comparable. Test metrics
  have to reflect the *real* image.

So `get_transforms(input_size, train)` branches: `train=True` includes the random ops;
`train=False` is only resize + normalize. That single boolean enforces the rule everywhere.

### 6. Why `albumentations` (and a version gotcha)
The prompt specified `albumentations` over torchvision transforms. It's faster (built on
OpenCV) and has a richer library of image augmentations. A subtle real-world lesson surfaced:
**library APIs change across versions.** `requirements.txt` pins `albumentations==1.4.10`, so
I used `A.Rotate(value=0, ...)` — in that version the fill color parameter is `value`. Newer
versions renamed it to `fill`. Writing code against the *pinned* version, not the latest
docs, is what makes it actually run. (This is exactly why Lesson 01 harped on pinned
versions.)

One more detail: rotation fills the corners with **black (`value=0`)**, because MRI scans sit
on a black background — reflecting or wrapping the edge pixels would create unnatural
artifacts.

### 7. The clever bit: importing torch *lazily*
Here's a genuinely useful engineering trick. The problem: `split.py` needs only one tiny
function from `data.py` — `parse_subject_id` — and it should run **locally, without a GPU
stack** (torch, albumentations aren't installed on your laptop). But if `data.py` imports
torch at the top, then `import data` fails locally, and `split.py` breaks.

The solution — a *try-import with fallback*:

```python
try:
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:
    _DatasetBase = object          # fall back when torch is absent

class OasisDataset(_DatasetBase):
    ...
```

When torch exists (Kaggle), `OasisDataset` is a real PyTorch `Dataset`. When it doesn't
(your laptop), the class still *imports* fine — it just subclasses plain `object`, which is
harmless because you only ever *instantiate* it where torch is present. The heavy imports
(albumentations, cv2, PIL) are likewise done *inside* the functions that use them, not at the
top of the file. Net effect: `from src.data import parse_subject_id` works on a bare laptop.

**We verified this live:** torch isn't installed in the dev environment, and the import +
parsing still ran correctly.

### 8. The trick I *rejected* — and why that matters
My first draft used a much fancier version of the lazy-import idea: it mutated the class's
`__bases__` at runtime (`cls.__bases__ = (Dataset,)`) the first time you created an instance.
It worked, but I threw it away. Why?

- It would **break `isinstance(ds, Dataset)`** checks in subtle ways.
- It could **break DataLoader multiprocessing**, which pickles the class to send to worker
  processes — a class whose bases mutate at runtime is a pickling hazard.
- It's *clever*, and clever is a liability in code someone else has to trust.

Lesson: **the boring, standard solution (try/except import) beat the clever one.** When two
approaches both work, prefer the one a stranger can understand at a glance. Cleverness that
buys nothing is technical debt.

### 9. Honesty about what could and couldn't be tested
Neither `torch` nor `albumentations` is installed locally, so the *full* image pipeline can
only run on Kaggle. Rather than claim "it works," I verified exactly what was verifiable
here — syntax compiles, the torch-free import path runs, `parse_subject_id` returns the right
IDs and raises on bad input — and stated plainly that the augmentation pipeline awaits Kaggle.
Match the confidence of your claim to the evidence you actually have.

---

## Takeaways

1. **Investigate before you overwrite.** A prompt that assumes a blank file doesn't make an
   existing 475-line file disappear safely. Check what you'd destroy.
2. **One file, one responsibility.** `data.py` loads images; `split.py` decides splits; the
   notebook wires them. Clean seams make the whole project legible.
3. **`train=True` vs `train=False` is a correctness boundary, not a style knob.** Augment
   train, never test.
4. **Prefer boring over clever.** The try/except import beat the `__bases__` hack because it
   can't surprise anyone later.
5. **Code against your *pinned* versions.** `A.Rotate(value=...)` is right for 1.4.10 even if
   the latest docs say `fill`.

## Homework before the next step
1. Open the new `src/data.py` and trace one image through `__getitem__` in your head: file →
   grayscale array → 3-channel stack → `get_transforms` → tensor. Can you say what *shape* it
   is at each step? (Answer: `(H,W)` → `(H,W,3)` → `(3, input_size, input_size)`.)
2. The next prompt writes `src/split.py`. Predict: which function will it import from
   `data.py`, and why does that import need to work *without* torch installed?
3. Recover the old monolithic `data.py` with `git show HEAD:alzheimer-ensemble/src/data.py`
   and skim the split logic — you'll recognize most of it when we re-home it into `split.py`.
