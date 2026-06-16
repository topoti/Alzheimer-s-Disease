# Claude CLI Execution Playbook
### Alzheimer's Heterogeneous-Paradigm CNN Ensemble + ADASYN + Triple-XAI

**How to use this file**
- Run each prompt **one at a time**, in order. Wait for Claude Code to finish + verify before the next.
- Prompts 0–13 = build code locally with Claude Code (CLI). Prompts marked **[KAGGLE]** = work you actually *run* on Kaggle GPU, not in the CLI.
- After every code-writing prompt, commit (`git add -A && git commit -m "..."`) before moving on. This is your checkpoint trail.
- The CLI does **not** train. It writes the training code; Kaggle executes it.

**Two-place workflow**
1. **Laptop (Claude Code)** → writes repo, `src/` modules, notebooks, paper.
2. **GitHub** → the bridge. Push from laptop, pull into Kaggle.
3. **Kaggle (GPU)** → runs the notebooks, produces checkpoints + results, you download them back into the repo.

---

## Prompt 0 — Load the plan into context (run once, every new session)

```
Read RESEARCH_PLAN.md in full. This is the single source of truth for an
academic Alzheimer's MRI classification project. Do not write any code yet.
Summarize back to me in 10 bullets: the 3 contributions, the locked decisions
(framework, 3 backbones, XAI trio, compute), the dataset, and the patient-level
split rule. Confirm you understand that ADASYN is applied AFTER the split, on
the training set only, in feature/embedding space — not pixel space.
```
*Why: anchors Claude Code so every later prompt stays consistent with the plan. Re-run this first whenever you start a fresh CLI session — Claude Code has no memory between sessions.*

---

## Prompt 1 — Scaffold the repo (Phase 8)

```
Create the project skeleton exactly as specified under "Project folder
structure" in RESEARCH_PLAN.md. Create all folders and empty placeholder
files. Then write:
- requirements.txt with PINNED versions for: torch, torchvision, timm,
  numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, Pillow,
  opencv-python, albumentations, pytorch-grad-cam, shap, lime, tqdm, wandb.
- configs/default.yaml with all hyperparameters from the Phase 4 table.
- A .gitignore that excludes data/raw/, checkpoints/, and *.pth.
- src/utils.py with a set_seed(42) function seeding torch, numpy, random.
Initialize a git repo and make the first commit. Do not install anything.
```

After it finishes:
```
git remote add origin <your-private-github-repo-url>
git push -u origin main
```

---

## Prompt 2 — Model factory (Phase 2)

```
Write src/models.py. One function: build_model(name, num_classes=4,
pretrained=True). Support name in {"resnet50", "efficientnet_b3",
"densenet121"} via timm.create_model. Replace the classifier head with:
GlobalAvgPool -> Dropout(0.3) -> Linear(in_features, 4). Expose a helper that
returns the correct input size per model (224 for resnet50/densenet121, 300
for efficientnet_b3). Add a __main__ block that loads all 3 models and prints
their parameter counts, so I can verify they build without error.
```
*Verify: the print block matches Phase 2 expectations (~25.6M / ~12M / ~8M params).*

---

## Prompt 3 — Data layer (Phase 3)

```
Write src/data.py. Implement:
- A function parse_subject_id(filename) that extracts the OAS1_XXXX patient ID
  from an OASIS filename.
- A PyTorch Dataset class OasisDataset(root, file_list, labels, input_size,
  train=True) returning (image_tensor, label). Preprocessing per the Phase 3
  pipeline: load, resize to input_size, grayscale->3-channel duplicate,
  ImageNet normalize. Online augmentation (rot +/-10, hflip, brightness +/-10%)
  only when train=True. Use albumentations.
- A get_transforms(input_size, train) helper.
Keep the two input sizes (224 / 300) configurable. Add type hints and docstrings.
```

---

## Prompt 4 — Patient-level split (Phase 3, critical)

```
Write src/split.py. Build the unique-subject list per class by parsing
OAS1_XXXX from filenames. Do a SUBJECT-LEVEL stratified 80/10/10 split with
random_state=42 (StratifiedGroupKFold or a manual stratified split of the
subject list), then expand subjects to their slices. Special-case the Moderate
class (only 2 subjects): 1 subject -> train, 1 -> test, 0 -> val; record the IDs.
Save the subject->split assignment and the per-split file lists to data/splits/.
Then write an assertion that FAILS LOUDLY if any subject ID appears in two
splits. Print the per-class image and subject counts for each split.
```
*Verify: assertion passes (no patient in two splits) and Moderate shows 1 train / 1 test / 0 val. This is the single biggest leakage risk — do not skip the check.*

---

## Prompt 5 — EDA notebook (Phase 3)

```
Write notebooks/01_dataset_eda.ipynb. It should: load the dataset from the
Kaggle path /kaggle/input/imagesoasis, count images AND unique patients per
class (parse OAS1_XXXX), print a table matching the Phase 1 counts, and plot
the class distribution as a bar chart. Add a markdown cell at the top
explaining this notebook runs on Kaggle, not locally.
```

---

## Prompt 6 — ADASYN pipeline (Phase 3, the core novelty)

```
Write src/adasyn_pipeline.py. Steps:
1. extract_embeddings(model, dataloader): pass TRAINING images through a frozen
   pretrained backbone (default resnet50), return penultimate-layer embeddings
   + labels as numpy arrays.
2. apply_adasyn(embeddings, labels, sampling_strategy): use
   imblearn.over_sampling.ADASYN with a CAPPED sampling_strategy (bring
   minorities to ~25-50% of majority, NOT full balance — see Phase 3). Return
   balanced embeddings + labels.
3. Print class distribution BEFORE and AFTER. Assert val/test are never passed in.
4. A sanity helper: for a few synthetic embeddings, find the nearest real
   training image by embedding distance.
Make the cap configurable from configs/default.yaml.
```
*Verify: prints show a capped distribution, not 4 equal classes; majority class unchanged.*

---

## Prompt 7 — Training loop (Phase 4)

```
Write src/train.py with one reusable function:
train_one_model(model_name, dataloaders, config). Implement the two-stage
transfer learning from Phase 4: Stage 1 = freeze backbone, train head 3 epochs;
Stage 2 = unfreeze last 2 blocks + head, fine-tune ~25 epochs at lr 1e-4.
Use: AdamW, CosineAnnealingLR, class-balanced cross-entropy (effective-number
weights), torch.cuda.amp mixed precision, early stopping patience=5 on val
MACRO-F1. Log per epoch: train loss, val loss, val acc, val macro-F1, per-class
recall. Save the best checkpoint by val macro-F1 to checkpoints/. Integrate wandb
logging but make it optional via a config flag.
```
*Pick ONE primary imbalance corrector per the Phase 4 note — don't stack heavy weighted loss + aggressive sampler + full oversampling. Default to the capped ADASYN + class-balanced loss combo.*

---

## Prompt 8 — Train ResNet50 first, end to end (Phase 4)

Build the notebook locally:
```
Write notebooks/03_train_resnet50.ipynb that: installs requirements, pulls the
repo, runs the split, extracts embeddings, applies capped ADASYN, then calls
train_one_model("resnet50", ...). It must save the checkpoint to
/kaggle/working/ and log final val metrics. Add a markdown header noting this
runs on Kaggle GPU with the imagesoasis dataset attached.
```

**[KAGGLE]** Then actually run it:
1. New Kaggle notebook → Accelerator: **GPU T4 ×2**.
2. Attach dataset `ninadaithal/imagesoasis`.
3. Clone your GitHub repo in cell 1, run `03_train_resnet50.ipynb`.
4. Download the checkpoint, commit it (or commit just the metrics/logs).

*Get this ONE model fully working before touching the others — per your "build incrementally" principle. If ResNet50 hits >85% val accuracy, the pipeline is sound.*

---

## Prompt 9 — Train the other two backbones (Phase 4)

```
Clone notebook 03 into notebooks/04_train_effnetb3.ipynb and
notebooks/05_train_densenet121.ipynb, swapping the model name and input size
(300 for efficientnet_b3, 224 for densenet121). For effnetb3 set batch_size=16
to avoid OOM on T4. Keep everything else identical.
```
**[KAGGLE]** Run both on Kaggle the same way. Save all 3 checkpoints.

---

## Prompt 10 — Ensemble (Phase 4)

```
Write src/ensemble.py. Load the 3 saved checkpoints. Implement weighted soft
voting: for each input, average the 3 softmax outputs weighted by each model's
normalized validation macro-F1 (weights sum to 1), then argmax. Expose
ensemble_predict(x) -> probs so it can be wrapped by SHAP/LIME later. Add a
function to compute and print ensemble metrics on the test set.
```

---

## Prompt 11 — Evaluation + ablations (Phase 6)

```
Write src/evaluate.py: an evaluate(model_or_ensemble, test_loader) returning
accuracy, macro-F1, per-class precision/recall, balanced accuracy, AUC-ROC and
PR-AUC (one-vs-rest). Add functions to plot a 4x4 confusion matrix
(seaborn heatmap) and per-class ROC curves, saving to results/figures/.
Then write notebooks/06_ensemble_eval.ipynb that runs full evaluation on the 3
single models + the ensemble and fills the Phase 6 comparison table.
Also implement the two ablations: (a) without-ADASYN (plain augmentation only),
(b) same-family ensemble (3 ResNets). Add McNemar's test vs the best single model.
```

---

## Prompt 12 — XAI modules (Phase 5)

```
Write src/xai/gradcam.py, shap_explain.py, lime_explain.py per Phase 5:
- gradcam.py: per-model Grad-CAM via pytorch-grad-cam, then pixel-wise average
  the 3 normalized heatmaps into one ensemble heatmap.
- shap_explain.py: wrap ensemble_predict as the prediction function, run SHAP
  on ~20 representative test images (it's slow).
- lime_explain.py: LimeImageExplainer on the same 20 images.
Then write notebooks/07_xai_figures.ipynb producing the 4-column paper figure:
original MRI | Grad-CAM | SHAP | LIME, for 5 correct + 5 incorrect per class.
```
*Verify: Grad-CAM blobs land on medial-temporal / hippocampal regions for dementia-positive cases. If they don't, something's off in the pipeline.*

---

## Prompt 13 — Paper scaffolding (Phase 7)

```
Create paper/ as a LaTeX project using the Diagnostics (MDPI) template
structure. Generate section files following the Phase 7 outline (Abstract,
Introduction, Related Work, Methodology, Results, Discussion, Conclusion).
Pre-fill the Methodology and Results structure with \input placeholders for the
tables and figures I'll generate. Write the "positioning your novelty" paragraph
from the plan verbatim into the Introduction. Leave TBD markers where numbers go.
```

---

## Quick reference: which phase each prompt covers

| Prompt | Phase | Output |
|---|---|---|
| 0 | — | Context loaded |
| 1 | 8 | Repo skeleton, requirements, config, git |
| 2 | 2 | `models.py` |
| 3 | 3 | `data.py` |
| 4 | 3 | `split.py` (patient-level) |
| 5 | 3 | EDA notebook |
| 6 | 3 | `adasyn_pipeline.py` |
| 7 | 4 | `train.py` |
| 8 | 4 | ResNet50 trained (Kaggle) |
| 9 | 4 | EffNetB3 + DenseNet121 trained (Kaggle) |
| 10 | 4 | `ensemble.py` |
| 11 | 6 | `evaluate.py` + ablations |
| 12 | 5 | XAI modules + figures |
| 13 | 7 | Paper LaTeX scaffold |

---

## Three things that will save you

1. **Re-run Prompt 0 at the start of every CLI session.** Claude Code forgets between sessions; the plan is your re-anchor.
2. **Commit after every prompt.** Each commit is a rollback point if a later prompt breaks something.
3. **Verify before advancing.** Each prompt has a *Verify* note — the patient-level split check (Prompt 4) and the capped-ADASYN distribution (Prompt 6) are the two that, if wrong, silently invalidate your whole paper.
