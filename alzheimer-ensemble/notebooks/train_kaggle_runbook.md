# train_kaggle.ipynb on Kaggle — the runbook

This is the **operations** half of the guide: how to actually run the notebook on
Kaggle, how to not burn your free GPU hours, how to make sure a dead session
can't eat your results, and what every error message means.

The companion file `train_kaggle_architecture.md` covers the code itself — the
settings, the model, and which knob to turn when a run goes wrong. This file
assumes you have read it, or at least that you'll flip back to it when a setting
name comes up. (`train_kaggle_concept.md` covers *why* the method is designed
this way.)

No prior Kaggle experience assumed. Read top to bottom.

---

## 1. What a "notebook" is, and how to run it

A Kaggle **notebook** is a web page made of stacked boxes called **cells**. Each
cell holds a bit of code. You run cells **top to bottom**, one at a time (or all
at once with "Run All"). Each cell must finish before the next makes sense,
because later cells use things earlier ones created.

A cell is running when a `[*]` sits next to it; it's done when that becomes a
number. Nothing you type into a later cell matters until the earlier ones have
finished.

Our notebook has 8 numbered sections:

| Section | What it does | Cost |
|---|---|---|
| 0. GPU sanity check | Confirms the GPU is actually on before you waste time | seconds |
| 1. Config | All the knobs and settings in one place | instant |
| 2. Rebuild the split | Decide which people go in which group | slow once, instant after (cache) |
| 3. Dataset + transforms | Load, clean, and augment each picture | instant (defines tools) |
| 4. Model factory | Build the "brain," and decide which parts may change | instant (defines tools) |
| 5. Train / eval helpers | The grading and one-round-of-teaching tools | instant (defines tools) |
| 6. Two-stage training | The driver that actually does the learning | **this is the expensive one** |
| 7. Results | The final report card | seconds |
| 8. Persist outputs | Save your checkpoints so a dead session can't eat them | seconds |

**Golden habit:** run sections 0 → 5 once (they just define tools and are fast),
then section 6 is the slow one that does the real work. **Always finish with
section 8** — Section 6 of this file has the horror story that motivates it.

---

## 2. Your general Kaggle workflow (step by step)

A repeatable routine for each work session:

1. **Open the notebook** on Kaggle.
2. **Turn on Persistence if your sidebar offers it** (Settings → Persistence →
   Files only). If it isn't there, skip this — Method 1 in Section 6 covers you.
3. **Attach the dataset** if needed: "Add Input" → search `ninadaithal/imagesoasis`
   (or your own upload) → Add.
4. **Set the accelerator** — GPU **T4 x2** or **P100** — and set it **before** you
   run anything, not in the middle. Changing it mid-session wipes your files.
5. **Run Section 0** (GPU check). If it says `CPU only`, stop and fix that first.
6. **Run sections 1–5.** Fast. Check the printed `data_root ->` line and that the
   per-fold subject counts look sane (~56 subjects per fold, ~70 in fold `-1`).
7. **Run Section 6** (training). Watch the `[s2 ...]` lines — read them using
   §10 of the architecture file.
8. **Run Section 7** to read the report card. Compare macro-F1 to the 0.767 lazy
   baseline.
9. **Run Section 8** and **click the download link**.
10. **Change ONE thing**, then repeat from step 7. One change at a time is how you
    learn what actually helps.
11. **When happy:** set `run_folds = [0,1,2,3,4]`, run once, record mean ± std.
12. **Save Version**, then **Stop the session** so you don't leak GPU hours.

---

## 3. Quick pre-flight checklist

Before you press "Run" on a training session, confirm:

- [ ] Persistence is ON, **if** your sidebar has it (Section 6, Method 3).
- [ ] Dataset is attached and `data_root ->` printed a real path.
- [ ] Accelerator is set to **GPU**, and you will **not** change it mid-session.
- [ ] Section 0 reported a real GPU, not `CPU only`.
- [ ] `run_folds` matches your intent — `[0]` to smoke-test, `[0,1,2,3,4]` to believe.
- [ ] You changed **only one** setting since the last run.
- [ ] `seed` is still 42 (fair comparison).
- [ ] You have enough weekly GPU hours left.

And after it finishes:

- [ ] Ran Section 8 and **clicked the download link**.
- [ ] Saved a Version if the run is worth keeping.
- [ ] Stopped the session.

---

## 4. Not hitting the Kaggle free limits

Kaggle gives you, for free:

- **~30 GPU hours per week** (resets weekly, rolling).
- **Max 12 hours per single session** (a longer run gets killed).
- A **9-hour** cap for committed ("Save & Run All") runs.

Think of the 30 hours like a phone data plan. How to make it last:

1. **Smoke-test on ONE fold.** Keep `run_folds = [0]` while you're just checking
   the code runs. Five folds costs 5×. This is by far your biggest lever.

2. **Budget for the wider slice window.** `central_lo/hi = 124..136` is ~2.6×
   slower per epoch than the old `128..132`. This is a deliberate trade — the old
   window produced an unusable model. Don't "save time" by reverting it for a run
   whose number you intend to report.

3. **Keep mixed precision on** (`use_amp = True`). Free ~2× GPU speedup.

4. **Let the cache work.** With `cache_split = True`, the slow file-scan happens
   once per session and writes
   `/kaggle/working/central_124_136.parquet`; reruns of Section 2 load instantly.
   The cache is included in `outputs.zip`, so you can re-upload it as a dataset to
   skip the scan entirely. Changing the slice window changes the filename, so it
   rebuilds automatically — you can't accidentally train on a stale cache.

5. **Don't leave the tab idle with the GPU on.** The clock runs whenever the
   session is active with GPU enabled — even while you're just reading. Click
   **Stop Session** (top-right) when you pause.

6. **Turn the GPU OFF while writing/editing code** — but read the warning in
   Section 6 first. Changing the accelerator **wipes `/kaggle/working`**.

7. **Watch the session timer** in the right-hand panel before a long run.

8. **Estimate before committing.** Note how long one stage-2 epoch takes (the gap
   between two `[s2 ...]` prints). Multiply by `stage2_epochs` (12) for one fold,
   then by 5 for all folds. If that exceeds your remaining hours, don't start.

**One-line summary:** *smoke-test on fold 0, keep AMP on, and Stop the session the
moment you're not actively training.*

---

## 5. Reusing a saved model (skip retraining)

Because each run saves `model_fold0.pt`, you can load it back instead of training
again. The file is a **dictionary**, not bare weights, so it explains itself:

```python
ck = torch.load("/kaggle/working/model_fold0.pt", map_location=device,
                weights_only=False)

model = build_model(ck["cfg"]["backbone"]).to(device)
model.load_state_dict(ck["state_dict"])
model.eval()          # switch to "just make predictions" mode

print("saved score:", ck["val_macro_f1"], "from epoch", ck["best_epoch"])
```

The keys inside are `state_dict`, `cfg`, `fold`, `val_macro_f1`, `best_epoch`,
and `class_names`.

`weights_only=False` is needed because the file also carries the config dict, not
just tensors. Only load checkpoints you produced yourself.

You still need to have run sections 1–5 first, because `build_model` and
`subject_eval` are defined there. Then you can call
`subject_eval(model, some_loader)` without spending a single training minute.

If you downloaded `outputs.zip` from a previous session, you can re-upload it as a
Kaggle Dataset and attach it as an input — then load from
`/kaggle/input/<your-dataset>/model_fold0.pt` instead.

---

## 6. Saving your outputs (read this before your first run)

`/kaggle/working/` is **scratch space on a temporary machine**. It is erased when
the session ends, when the machine restarts, and — the one that catches
everybody — **when you change the accelerator setting**. That is a real thing that
happened here: a finished run's `model_fold0.pt` vanished because the accelerator
dropdown was touched afterwards. There is no undo and nothing to recover.

Two methods that always work, plus one that may not exist in your UI. None needs
an API key or a paid account.

### Method 1 — Download the zip (Section 8 of the notebook)

`persist_outputs()` runs automatically at the end of the notebook. It:

- lists every artifact in `/kaggle/working` with its size (`*.pt`, `*.parquet`,
  `*.csv`),
- writes a `run_summary.json` (your config + per-fold macro-F1 and balanced
  accuracy) so a downloaded bundle explains itself,
- zips everything into `outputs.zip`,
- prints a **click-to-download link**.

**Click that link.** One second, no setup, nothing to configure, works offline.
This is your primary safety net — treat the others as extras.

### Method 2 — Save Version

**Save Version → "Save & Run All (Commit)"** re-runs the whole notebook top to
bottom on a clean machine, and whatever lands in `/kaggle/working` becomes a
permanent, downloadable **Output** attached to that version. It survives
everything, including you editing the notebook later, and other notebooks can
mount it as an input dataset.

This is the closest thing to true automation: press it once, walk away, and the
run persists itself. Remember the 9-hour cap on committed runs. The other option,
"Quick Save," snapshots your current interactive state instead of re-running.

### Method 3 — Persistence (only if your sidebar has it)

Some Kaggle UIs offer **Settings → Persistence → "Files only"**, which keeps
`/kaggle/working` across restarts of that notebook. If you see it, turn it on.

**It is not present in every Kaggle UI or notebook type** — if your Settings pane
only shows Internet, Environment options, and Accelerator, you simply don't have
it, and nothing is wrong. Kaggle also documents persistence as *best-effort*: it
can fail silently on a crash or with large files. Never make it your only copy.

> **Why the Output panel can look empty even after a successful save:** the
> right-hand Output pane doesn't always auto-refresh during an interactive
> session — hit its refresh icon. And files only become a *permanent* Output
> after a Save Version; before that, they exist only on the live machine.

**Do not rely on the accelerator dropdown being safe.** Set it once, before you
start, and leave it alone for the rest of the session.

---

## 7. Common errors and what they mean

| Message you see | What's really wrong | Fix |
|---|---|---|
| `No images found under ...` | dataset not attached, or wrong path | Add the dataset via "Add Input"; check the `data_root ->` printout. The message also lists what *is* under `/kaggle/input`. |
| `FileNotFoundError: .../model_fold0.pt` **and `ls` prints nothing** | the session restarted (usually from an accelerator change) and wiped `/kaggle/working` | Nothing to recover — retrain, then download the zip (Section 6) the moment it finishes. |
| No "Persistence" option in Settings | not offered in every Kaggle UI / notebook type | Nothing is broken. Use Section 6 Methods 1 and 2 instead. |
| `CUDA out of memory` | batch too big for the GPU | Lower `batch_size` to 16 (or 8) |
| Training is extremely slow | GPU is off — or your weekly quota ran out and Kaggle silently gave you CPU | Run Section 0; set Accelerator to GPU; check remaining hours |
| `dropped N unreadable image(s)` | a few corrupt files (normal in small numbers) | Ignore if small; if most are bad, re-upload the dataset |
| `All central-slice images were unreadable` | the dataset upload is corrupt | Re-upload the dataset |
| `'layer4' not in ['features.0', ...]` | `freeze_until` doesn't match the chosen `backbone` | Use a block name from that backbone (architecture file, §8) |
| Session died at ~12 hours | hit the per-session limit | Run fewer folds, or narrow the slice window |
| Scores look *too* good (0.98+) | probably a data leak (split by picture, not person) | Don't change the split code — that's exactly what protects you |
| `val_macroF1` falls while `train_loss` falls | overfitting | Architecture file, §11 |
| Output panel looks empty after saving | pane didn't refresh, or you haven't committed | Hit refresh on the Output pane; see Section 6 |

---

That's the operational half. Use the architecture file for the *what* and the
concept file for the *why*, and keep the pre-flight checklist (Section 3) handy
until it's second nature.
