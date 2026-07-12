# ResNet vs EfficientNet vs DenseNet

A simple comparison of three popular CNN architectures, all commonly used
as backbones for medical imaging (like Alzheimer's MRI classification).

---

## 1. The Core Problem They All Solve

Deep networks are hard to train. As you stack more layers:

- Gradients vanish (early layers stop learning)
- Accuracy gets *worse*, not better (degradation problem)
- Compute cost explodes

Each architecture attacks this differently.

---

## 2. ResNet (2015) — "Skip the layer"

**Key idea: Residual (skip) connections.**

Instead of forcing a block to learn the full mapping `H(x)`,
it learns only the *residual* `F(x)`, and the input is added back:

```
output = F(x) + x
```

```
x ──► [Conv → BN → ReLU → Conv → BN] ──► (+) ──► ReLU ──► output
│                                         ▲
└──────────── identity shortcut ──────────┘
```

**Why it works:**
- If a layer is useless, the network can just learn `F(x) = 0`
  and pass `x` through unchanged — adding layers can never hurt.
- Gradients flow directly through the shortcut, so very deep
  networks (50, 101, 152 layers) train easily.

**Example:** `ResNet50` = 50 layers, ~25M parameters. The default
"safe choice" backbone in most transfer-learning tutorials.

```python
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
```

---

## 3. DenseNet (2017) — "Connect everything"

**Key idea: Dense connections — every layer receives ALL previous layers' outputs.**

ResNet *adds* (`+`) the shortcut. DenseNet *concatenates* instead:

```
Layer 1 input:  x0
Layer 2 input:  [x0, x1]
Layer 3 input:  [x0, x1, x2]
Layer 4 input:  [x0, x1, x2, x3]
```

```
x0 ──┬──────────┬──────────┬──────► concat ──► ...
     ▼          │          │
   layer1 ──┬───┼──────────┤
            ▼   ▼          │
          layer2 ──┬───────┤
                   ▼       ▼
                 layer3 ──► ...
```

**Why it works:**
- Features are **reused**, not re-learned → far fewer parameters.
- Every layer has a direct path to the loss → strong gradient flow.
- Each layer only adds a small number of new channels
  (the "growth rate", e.g. k=32), keeping the model compact.

**Example:** `DenseNet121` = 121 layers but only ~8M parameters —
smaller than ResNet50 despite being much deeper. Popular in medical
imaging (e.g. CheXNet for chest X-rays) because feature reuse helps
when datasets are small.

```python
from tensorflow.keras.applications import DenseNet121
model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
```

**Trade-off:** concatenation eats a lot of GPU **memory** during
training, and can be slower per step than ResNet.

---

## 4. EfficientNet (2019) — "Scale smartly"

**Key idea: Compound scaling — grow depth, width, and resolution together, in balance.**

Before EfficientNet, people scaled networks arbitrarily:
- Deeper (more layers) → ResNet18 → ResNet152
- Wider (more channels)
- Bigger input images

EfficientNet says: scale **all three at once** with fixed ratios:

```
depth      = α^φ
width      = β^φ
resolution = γ^φ      (α·β²·γ² ≈ 2, φ = how much compute you have)
```

Turning the single knob `φ` gives the family **B0 → B7**:

| Model | Input size | Params | Rough use case            |
|-------|-----------|--------|---------------------------|
| B0    | 224×224   | ~5M    | Fast baseline             |
| B3    | 300×300   | ~12M   | Good balance              |
| B7    | 600×600   | ~66M   | Max accuracy, heavy       |

It also uses efficient building blocks (**MBConv**: depthwise-separable
convolutions + squeeze-and-excitation attention), which is why B0
beats ResNet50 accuracy with 5× fewer parameters.

```python
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
```

> Note: Keras `EfficientNet` includes its own input rescaling — feed it
> raw 0–255 pixels, unlike ResNet/DenseNet which need `preprocess_input`.

---

## 5. Side-by-Side Difference

| Aspect              | ResNet                     | DenseNet                       | EfficientNet                     |
|---------------------|----------------------------|--------------------------------|----------------------------------|
| Year                | 2015                       | 2017                           | 2019                             |
| Core trick          | Skip connection (add)      | Dense connection (concat)      | Compound scaling + MBConv        |
| How info flows      | `x + F(x)`                 | `[x0, x1, ..., xn]`            | Efficient blocks, scaled uniformly |
| Parameters (typical)| ResNet50 ≈ 25M             | DenseNet121 ≈ 8M               | B0 ≈ 5M, B7 ≈ 66M                |
| Params efficiency   | Moderate                   | High (feature reuse)           | Highest (accuracy per param)     |
| Training memory     | Moderate                   | High (concat is costly)        | Low–moderate                     |
| Speed               | Fast, well-optimized       | Slower per step                | Fast for its accuracy            |
| Strength            | Simple, robust, reliable   | Small data / medical imaging   | Best accuracy-vs-cost trade-off  |
| Weakness            | More params for same acc.  | GPU memory hungry              | Sensitive to preprocessing/aug.  |

---

## 6. Mental Models (one-liners)

- **ResNet**: "If in doubt, let the input skip through." (addition)
- **DenseNet**: "Never throw a feature away — reuse everything." (concatenation)
- **EfficientNet**: "Don't just build deeper — scale depth, width, and
  image size together." (balanced scaling)

---

## 7. Which One for an Alzheimer's MRI Ensemble?

All three make good, *diverse* ensemble members precisely because they
extract features differently:

- **ResNet50** — stable baseline, easiest to fine-tune.
- **DenseNet121** — often best on small medical datasets thanks to
  feature reuse; strong track record in radiology.
- **EfficientNetB0/B3** — best accuracy per parameter; pick B0 for
  speed, B3 if GPU allows larger inputs.

Because their inductive biases differ (add vs concat vs scaled MBConv),
their mistakes are less correlated — which is exactly what you want
when averaging predictions in an ensemble.

---

## 8. Quick Reference: Preprocessing in Keras

```python
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre     # caffe-style
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre   # torch-style (0-1, normalized)
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre # identity (built-in rescale)
```

Mixing these up is one of the most common transfer-learning bugs —
each backbone expects its own input normalization.
