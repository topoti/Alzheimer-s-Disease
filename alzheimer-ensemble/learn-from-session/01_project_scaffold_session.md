# Lesson 01 — Scaffolding the Project (What This Session Taught)

> A walkthrough of the project-skeleton session, framed as lessons — not just *what*
> was done but *why*, so you can understand and defend each decision.

## Session in two acts

**Act 1 — Read the plan and prove understanding.** Before touching code, I read
`RESEARCH_PLAN.md` end-to-end and summarized the locked decisions back to you. The
point: the human and the assistant must share the same mental model first. The one
rule easiest to get catastrophically wrong was confirmed up front — **ADASYN runs
after the split, on training data only, in embedding space.**

**Act 2 — Build the empty skeleton and commit it.** No real logic yet — just folders,
placeholder files, the dependency list, the config, and a reproducibility helper.
"Scaffolding" means creating the whole shape of the project first so every later
piece of code has an obvious home.

---

## The concepts worth actually learning

### 1. Why scaffold *before* writing code
A research codebase that grows organically becomes a mess of `final_v2_REAL.ipynb`
files. Creating `src/data.py`, `src/models.py`, `src/train.py`, `src/xai/...` as empty
stubs up front commits you to *separation of concerns*: data loading in one file,
model definitions in another, explainability in its own folder. The structure alone
tells the story when a reviewer opens your repo.

### 2. Empty directories and the `.gitkeep` trick
**Git does not track empty folders** — only files. So `results/figures/` would silently
vanish for anyone who clones the repo. The convention: drop an empty `.gitkeep` file
inside, giving git something to track and preserving the folder. That's why `.gitkeep`
was added to `data/adasyn_cache/`, `paper/`, and the three `results/` subfolders.

### 3. `.gitignore` — keeping junk out of version control
Three kinds of things should *never* be committed:
- **Large raw data** (`data/raw/`) — gigabytes of MRI slices; they live on Kaggle.
- **Model weights** (`*.pth`, `checkpoints/`) — huge binaries that bloat history forever.
- **Generated cruft** (`__pycache__/`, `*.pyc`) — Python bytecode, regenerated automatically.

I also found a `.pyc` file accidentally committed earlier and untracked it
(`git rm --cached`). Lesson: `.gitignore` only stops *future* tracking — a file already
committed stays tracked until you explicitly untrack it.

### 4. Pinned dependency versions = reproducibility
`requirements.txt` uses `torch==2.3.1`, not `torch>=2.3`. The `==` matters enormously:
if a reviewer installs a newer version a year from now, results might not replicate.
**Pinning exact versions is how you make "it worked on my machine" a defensible claim.**
Subtlety: the library you call "pytorch-grad-cam" installs under the PyPI name
`grad-cam` — package names and import names don't always match.

### 5. Seeding — the `set_seed(42)` function
The single most important reproducibility tool. From `src/utils.py`:

```python
random.seed(seed)                  # Python's built-in RNG
np.random.seed(seed)               # NumPy (used by ADASYN, sklearn)
torch.manual_seed(seed)            # PyTorch CPU operations
torch.cuda.manual_seed_all(seed)   # PyTorch GPU operations
torch.backends.cudnn.deterministic = True   # force repeatable GPU math
torch.backends.cudnn.benchmark = False      # disable nondeterministic speed tricks
```

Neural network training uses randomness *everywhere*: weight init, data shuffling,
dropout, augmentation. Without a fixed seed you'd get slightly different accuracy every
run and could never tell whether a change *helped* or you just got lucky. Seeding every
RNG to the same number (42) makes runs repeatable. The `cudnn.deterministic` line is the
subtle one — GPUs use fast algorithms that aren't bit-for-bit reproducible by default;
this forces the slower-but-repeatable versions.

### 6. The config file — *why* I changed yours instead of just adding to it
The most important judgment call of the session. Your existing `configs/default.yaml`
was from an **earlier draft** and contradicted decisions the plan later locked in:

| Field | Was | Changed to | Why |
|---|---|---|---|
| `adasyn.apply_on` | `pixel_space` | `embedding_space` | Pixel-space ADASYN makes blurry fake MRIs — the plan *rejects* it as a baseline |
| `loss` | `cross_entropy` | `class_balanced_ce` | Plain CE ignores your 138× imbalance; class-balanced loss is your biggest minority-recall lever |
| `sampling_strategy` | `not_majority` (full balance) | `0.5` (capped) | Full-balancing 488 Moderate samples to ~54k just memorizes 2 patients and blows the GPU budget |
| `data.root` | a wrong dataset slug | `imagesoasis/Data` | Match the actually-locked Kaggle dataset |

General lesson: **a config file is a contract with your future experiments.** If it
disagrees with your methodology, every run inherits the bug. All four changes were
flagged explicitly so *you* can veto them — nothing was silently "fixed," because maybe
one old value was intentional. (It's on you to confirm they weren't.)

### 7. Why I refused to run `git init`
You asked to "initialize a git repo," but the folder was *already inside* a git repo
(the parent `Alzheimer-s-Disease/`). Running `git init` would have created a **nested
repo** — a confusing situation where git doesn't know which repo owns your files.
Lesson: always check `git rev-parse --show-toplevel` before initializing. I committed
into the existing repo instead and explained why, rather than blindly following the
instruction into a mess.

---

## Takeaways

1. **Structure first, logic later** — the skeleton is a map.
2. **Reproducibility is a discipline, not an afterthought** — pinned versions + seeds +
   a clean config separate a publishable result from an anecdote.
3. **An assistant that contradicts your instructions when reality disagrees is doing its
   job** — the nested-repo refusal and the config corrections are both examples. Read the
   flags raised; don't rubber-stamp them.

## Homework before the next step
Open `configs/default.yaml` and confirm the four changes match your intent —
especially `adasyn.apply_on: embedding_space` and the capped `sampling_strategy: 0.5`.
Those two decisions shape your entire data pipeline, and you'll have to defend them to a
reviewer.
