# Lesson 02 — Building the Model Factory (What This Session Taught)

> A walkthrough of the session where we wrote `src/models.py` — the file that turns a
> model *name* into a ready-to-train network. Framed as lessons: not just *what* the code
> does, but *why* each choice is the defensible one for your paper.

## Session in two acts

**Act 1 — Re-read the plan, prove shared understanding.** Same discipline as Lesson 01:
before writing code I re-read `RESEARCH_PLAN.md` and confirmed the locked decisions
(PyTorch + `timm`, the three backbones, the classifier head shape, the per-model input
sizes). Code written against a misremembered plan is worse than no code.

**Act 2 — Write one function that builds all three backbones.** The whole file exists to
serve a single public function, `build_model(name, num_classes=4, pretrained=True)`, plus
one helper, `get_input_size(name)`, and a `__main__` smoke test that proves all three
models assemble without error.

---

## The concepts worth actually learning

### 1. What a "model factory" is and why you want one
A *factory* is a function that hands you a fully-built object from a short recipe — here,
a model name. Instead of copy-pasting three near-identical blocks of setup code (one per
backbone), you write the assembly logic **once** and call it three times:

```python
resnet   = build_model("resnet50")
effnet   = build_model("efficientnet_b3")
densenet = build_model("densenet121")
```

Why this matters for *your* project specifically: your whole thesis is an **ensemble of
three models trained identically**. If ResNet and DenseNet get their heads or dropout set
up even slightly differently, a reviewer can attack your comparison as unfair. One factory
function guarantees all three are built by the exact same rules — the differences between
them are *only* the backbones, which is precisely the variable you're studying.

### 2. Transfer learning — why we don't train from scratch
Every backbone is loaded with `pretrained=True`, meaning it arrives already trained on
**ImageNet** (1.2M everyday photos). Those early layers have learned generic vision
primitives — edges, textures, curves — that transfer to *any* image, including brain MRI.
You then adapt this to your 4 Alzheimer's classes instead of learning vision from zero.

Analogy: you're not teaching someone to see; you're teaching an already-sighted radiology
trainee to read *these particular* scans. That's why ~86k images is enough — the hard part
(learning to see) was paid for by ImageNet.

### 3. `timm` — one library, every backbone
`timm` (PyTorch Image Models) gives one-line access to hundreds of pretrained
architectures with a uniform API. We use `timm.create_model(name, pretrained=True, ...)`.
The alternative (`torchvision.models`) works too, but `timm` has a cleaner, *consistent*
way to strip and rebuild the classifier head across different architectures — which is the
next, most important idea.

### 4. The key trick: `num_classes=0, global_pool="avg"`
This is the cleverest line in the file and worth understanding deeply.

Every pretrained CNN ends with a classifier head sized for **ImageNet's 1000 classes** —
useless to us. We need to remove it and attach our own 4-class head. `timm` gives a clean
way:

```python
backbone = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
```

- `num_classes=0` tells `timm`: **"remove the final classification layer entirely."**
- `global_pool="avg"` tells it: **"but keep the Global Average Pooling step."**

The result is a backbone whose `forward(x)` returns a flat **feature vector** of shape
`(batch, in_features)` — the network's learned summary of the image, *just before* it would
have classified. `backbone.num_features` tells you how wide that vector is (2048 for
ResNet50, 1536 for EfficientNetB3, 1024 for DenseNet121).

This one choice buys you **two** things:
1. A place to bolt on your own head (next section).
2. A ready-made **embedding extractor** for Phase 3 — remember, ADASYN runs in *embedding
   space*. This is the exact vector you'll feed ADASYN. The model factory and the ADASYN
   pipeline are secretly the same tool.

### 5. The classifier head: `GlobalAvgPool → Dropout(0.3) → Linear(in_features, 4)`
On top of the pooled features we attach a tiny new head:

```python
head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes),   # e.g. 2048 -> 4
)
model = nn.Sequential(backbone, head)
```

Reading the three pieces:
- **GlobalAvgPool** (provided by the backbone) collapses each feature map to a single
  number — turning a spatial grid of features into one flat vector. It's what lets a
  fixed-size `Linear` layer sit on top regardless of image dimensions.
- **Dropout(0.3)** randomly zeroes 30% of the features *during training only*. This is
  regularization: it stops the head from over-relying on any single feature, which fights
  overfitting — critical when your Moderate class comes from just 2 patients and is
  desperate to be memorized.
- **Linear(in_features, 4)** is the actual decision layer: a weighted vote from the
  features into 4 class scores (logits).

Note the head outputs raw **logits**, not probabilities. Softmax happens later — inside the
loss function during training, and explicitly in the ensemble's soft-voting step. Keeping
logits raw here is standard and avoids applying softmax twice.

### 6. Why *these* three backbones — the "paradigm" argument
This is your paper's novelty, encoded in code:

| Model | Design paradigm | One-line idea |
|---|---|---|
| **ResNet50** | Residual learning | Skip-connections let gradients flow through 50 layers |
| **EfficientNetB3** | Compound scaling + MBConv | Width, depth, resolution scaled together; efficient blocks |
| **DenseNet121** | Dense connectivity | Every layer sees *all* previous layers' features |

These are three genuinely different *ways of building a CNN*. Different designs make
**different mistakes**, and ensembles gain their power from *uncorrelated errors* — when
three models that fail differently agree, the answer is strong. This is why the plan
insists on the word **"cross-paradigm,"** not "cross-architecture" (they're all still CNNs
— saying otherwise gets you rejected on novelty).

### 7. Why EfficientNetB3 needs 300×300 but the others use 224×224
`get_input_size(name)` returns `300` for EfficientNetB3 and `224` for the other two. This
isn't arbitrary: EfficientNet's *compound scaling* deliberately increases input resolution
as the model grows, and it was **pretrained at ~300px**. Feed it 224px images and you're
using it off-spec — throwing away the resolution it was designed to exploit. So your data
pipeline must resize images *per model*, which is exactly why the factory exposes the input
size as a queryable helper rather than hard-coding 224 everywhere.

Practical consequence you'll hit later: 300×300 images use more GPU memory, which is why
the config sets `batch_size_effnet: 16` while the others use 32.

### 8. Total vs. trainable parameters — a preview of Phase 4
The `__main__` block prints both **total** and **trainable** parameter counts:

```python
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Right now they're equal. But in Phase 4 you'll *freeze* the backbone (Stage 1: train only
the head) and later unfreeze the last blocks (Stage 2). When you do, `trainable` will drop
far below `total` — and this print statement becomes how you *verify freezing actually
worked*. Building the diagnostic now means you won't be guessing later.

### 9. The `__main__` smoke test — and why it uses `pretrained=False`
The script's entry point builds all three models, runs a dummy image through each, and
asserts the output shape is `(1, 4)`:

```python
with torch.no_grad():
    out = model(torch.zeros(1, 3, size, size))
assert out.shape == (1, 4)
```

Two deliberate choices:
- **`pretrained=False` in the smoke test.** Downloading ImageNet weights needs internet
  and time. For a *"does it even assemble?"* check we skip the download — the wiring
  (backbone → head → correct output shape) is identical whether or not the weights are
  real. On Kaggle you'll flip to `pretrained=True` for actual training; that's already the
  default of `build_model` itself.
- **The dummy forward pass.** Constructing a model can succeed while it's secretly broken
  (wrong feature count, shape mismatch). Pushing one fake image through and checking the
  output is `(1, 4)` proves the head actually fits the backbone. This is the cheapest
  possible integration test.

### 10. A judgment call: `efficientnet_b3` vs `efficientnetb3`
I flagged a naming inconsistency rather than silently picking one. The CLI prompt and
`timm` both use `efficientnet_b3` (with underscore); your `configs/default.yaml` keys it as
`efficientnetb3` (no underscore). I followed the prompt's spelling in the code but called
out the mismatch, because when you later wire config → model, `config["efficientnetb3"]`
won't match `build_model("efficientnet_b3")` and you'll get a confusing `KeyError`.

Lesson: **naming mismatches are silent bugs.** The fix is a decision *you* should make —
standardize on one spelling everywhere, or add an alias — so I surfaced it instead of
hiding a guess inside the code.

### 11. Why I couldn't run it here, and what I did instead
`torch` and `timm` aren't installed in this local environment — they're the heavy Kaggle
GPU stack. So the real smoke test can't run until you're on Kaggle. Rather than *claim* it
works, I did the honest fallback: `python -m py_compile src/models.py`, which parses the
file and confirms it's **syntactically valid** without executing it. That's a smaller
claim ("it will parse and import wherever torch exists") and I stated it as exactly that —
not "it runs."

Lesson: match the confidence of your claim to the evidence you actually have.

---

## Takeaways

1. **Write shared logic once.** One factory function makes your three-model comparison
   fair by construction — the ensemble differs only in the variable you're studying.
2. **`num_classes=0, global_pool="avg"` is a two-for-one.** It gives you both a clean head
   mount *and* the embedding extractor your ADASYN step needs.
3. **Build your diagnostics early.** The trainable-parameter print and the shape-assert
   smoke test cost nothing now and save hours of confused debugging in Phase 4.
4. **Surface mismatches; don't paper over them.** The `efficientnet_b3` naming flag is a
   bug prevented, not a detail.

## Homework before the next step
1. Once you're on Kaggle (or any env with `torch`+`timm`), run `python -m src.models` with
   real weights and read the parameter counts — confirm ~25.6M (ResNet50), ~12M
   (EfficientNetB3), ~8M (DenseNet121). Those numbers are your sanity check that the right
   backbones loaded.
2. Decide the `efficientnet_b3` vs `efficientnetb3` spelling **once**, and make the config
   and code agree. You'll thank yourself when wiring Phase 4.
3. Look at `build_model` and predict: *which lines will you touch to freeze the backbone in
   Stage 1?* (Hint: you'll iterate over `backbone.parameters()` and set
   `requires_grad = False`.) Understanding that now makes the training file easy later.
