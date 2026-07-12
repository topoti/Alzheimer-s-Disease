# Machine Learning Basics — CNNs and the Words Around Them

Read top to bottom. Each section builds on the previous one.

---

## 1. What is Machine Learning (ML)?

Instead of writing rules by hand ("if pixel is dark, then..."), you show the
computer many **examples** and it figures out the rules itself.

- **Traditional programming:** rules + data → answers
- **Machine learning:** data + answers → rules (the "model")

Example: show 10,000 brain MRI scans labeled "Alzheimer's" or "healthy",
and the model learns what patterns separate the two.

---

## 2. Why "Neural"? — Neural Networks

A **neural network** is a model loosely inspired by neurons in the brain.

- A biological neuron receives signals, and fires if the signal is strong enough.
- An artificial **neuron** takes numbers in, multiplies each by a **weight**,
  adds them up, adds a **bias**, and passes the result through an
  **activation function** (a simple non-linear squash, like ReLU: `max(0, x)`).

Stack neurons into **layers**, and layers into a network:

```
input layer  →  hidden layer(s)  →  output layer
(pixels)        (learned features)   (prediction: "cat" 0.92)
```

"Deep learning" just means a neural network with **many** hidden layers (deep = many layers).

The "learning" is: adjust the weights until predictions match the labels.

---

## 3. Why "Convolutional"? — The Convolution Operation

Problem: an image is huge. A 224×224 color image = 150,528 numbers. Connecting
every pixel to every neuron (a "fully connected" network) explodes in size and
ignores that nearby pixels are related.

Solution: **convolution** — slide a small window (called a **filter** or
**kernel**, e.g. 3×3 numbers) across the image. At each position, multiply the
window with the pixels underneath and sum them into one number.

```
Image (5×5)        Filter (3×3)       Output (feature map)
1 1 1 0 0
0 1 1 1 0          1 0 1
0 0 1 1 1    *     0 1 0      →       4 3 4
0 0 1 1 0          1 0 1              2 4 3
0 1 1 0 0                             2 3 4
```

Why this is smart:

1. **Local patterns** — a 3×3 filter looks at small neighborhoods, where image
   structure lives (edges, corners, textures).
2. **Weight sharing** — the *same* filter slides over the whole image, so an
   edge detector works anywhere in the image. Far fewer weights to learn.
3. **Translation tolerance** — a cat in the top-left or bottom-right still
   triggers the same filters.

A **CNN (Convolutional Neural Network)** = a neural network whose main layers
do convolutions. Early layers learn edges; middle layers learn shapes (eyes,
wheels); late layers learn whole objects. The filters themselves are learned
from data — you don't design them.

---

## 4. The Standard CNN Building Blocks

A typical CNN for image classification:

```
Image → [Conv → ReLU → Pool] × N → Flatten → Fully Connected → Softmax → "dog: 0.95"
```

- **Conv layer** — applies many filters; each produces a **feature map**
  (a grid showing where its pattern appears).
- **ReLU** — activation function; keeps positives, zeroes out negatives.
  Adds non-linearity so the network can learn complex patterns.
- **Pooling** — shrinks feature maps (e.g. **max pooling** keeps the largest
  value in each 2×2 block). Reduces computation and keeps the strongest signals.
- **Flatten** — unrolls the final 2D feature maps into one long vector.
- **Fully connected (dense) layer** — every input connects to every neuron;
  combines all detected features to make the final decision.
- **Softmax** — turns raw scores into probabilities that sum to 1.

Famous CNN architectures you'll see named: **LeNet** (1998), **AlexNet** (2012),
**VGG**, **ResNet** (added "skip connections" to train very deep nets),
**EfficientNet**.

---

## 5. How Does It Learn? — Training Vocabulary

- **Label / target** — the correct answer for an example ("Alzheimer's").
- **Prediction** — what the model currently outputs.
- **Loss function** — a number measuring how wrong the prediction is
  (e.g. **cross-entropy** for classification, **MSE** for regression).
- **Gradient descent** — nudge every weight slightly in the direction that
  reduces the loss. Repeat millions of times.
- **Backpropagation** — the algorithm that efficiently computes, for every
  weight, "which direction reduces the loss?" (it's calculus chain rule).
- **Learning rate** — how big each nudge is. Too big → chaos; too small → slow.
- **Epoch** — one full pass over the training data.
- **Batch** — training data is fed in small chunks (e.g. 32 images at a time).
- **Optimizer** — the recipe for applying the nudges. Common ones: **SGD**, **Adam**.

Training loop in one sentence: *predict → measure loss → backpropagate →
update weights → repeat.*

---

## 6. Did It Actually Learn? — Evaluation Vocabulary

- **Train / validation / test split** — train on one chunk, tune on a second,
  report final results on a third the model has never seen.
- **Overfitting** — model memorizes training data but fails on new data
  (great train accuracy, bad test accuracy). Like a student memorizing answers
  instead of understanding.
- **Underfitting** — model too simple to capture the pattern at all.
- **Regularization** — tricks to fight overfitting: **dropout** (randomly
  disable neurons during training), **weight decay**, **data augmentation**
  (flip/rotate/crop images to fake more data).
- **Accuracy** — fraction of correct predictions.
- **Precision / recall** — for imbalanced problems (e.g. rare disease):
  precision = "of the ones I flagged, how many were right?";
  recall = "of all real cases, how many did I catch?"
- **Confusion matrix** — table of predicted vs actual classes.

---

## 7. Types of Learning (Bigger Picture)

- **Supervised learning** — data comes with labels (CNN image classification lives here).
- **Unsupervised learning** — no labels; find structure (e.g. **clustering**).
- **Self-supervised learning** — invent labels from the data itself (mask a word, predict it).
- **Reinforcement learning** — learn by trial, error, and reward (game-playing AI).
- **Transfer learning** — take a CNN pre-trained on millions of images
  (e.g. ImageNet) and **fine-tune** it on your small dataset. Extremely common
  in medical imaging, where labeled data is scarce.

---

## 8. Related Keywords You'll Bump Into

- **Tensor** — just an n-dimensional array of numbers. Images are tensors
  (height × width × channels). Hence "TensorFlow" and "PyTorch tensors".
- **Channels** — a color image has 3 (red, green, blue); conv layers produce
  many more (one per filter).
- **Stride** — how many pixels the filter jumps each step.
- **Padding** — adding zeros around the image edge so filters can cover borders.
- **Feature extraction** — using early CNN layers as a general-purpose
  "pattern detector" for another task.
- **Ensemble** — combining several models and averaging/voting their
  predictions; usually beats any single model.
- **Inference** — using a trained model to make predictions (no more learning).
- **Hyperparameters** — settings *you* choose (learning rate, layers, batch
  size), as opposed to **parameters** (weights the model learns).
- **GPU** — graphics card; does the matrix math of training thousands of times
  faster than a CPU.
- **RNN / LSTM** — networks for sequences (text, time series); the older
  cousin of...
- **Transformer** — the architecture behind modern language models (GPT,
  Claude) and, increasingly, images too (**Vision Transformer / ViT**).

---

## 9. One-Paragraph Summary

A **CNN** is a **neural network** (layers of simple weighted units that learn
by gradient descent) whose core operation is **convolution** (sliding small
learned filters over an image to detect local patterns). Early layers find
edges, deeper layers find shapes and objects. You **train** it by showing
labeled examples, measuring **loss**, and **backpropagating** corrections to
the weights; you check for **overfitting** on held-out data; and in practice
you usually start from a **pre-trained** network and **fine-tune** it —
especially in domains like medical imaging where data is limited.
