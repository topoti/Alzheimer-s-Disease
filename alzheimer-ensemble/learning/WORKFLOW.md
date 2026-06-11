# Research Workflow — Cross-Paradigm Alzheimer's Ensemble

A single flow chart of the end-to-end research, distilled from the 33 step files in this
folder. For the step-by-step curriculum see [`README.md`](README.md); for the full plan see
[`../RESEARCH_PLAN.md`](../RESEARCH_PLAN.md).

The project is **mostly linear (Phase 1 → 7)**, with **Phase 8 (Tools & Environment)
front-loaded** alongside Phase 1. Everything traces back to one root cause — the **~138× class
imbalance** — which spawns the **three pillars** (ADASYN, heterogeneous ensemble, triple XAI)
that thread across multiple phases. A small **data lane** on the right shows how artifacts
flow from raw OASIS (`ninadaithal/imagesoasis`, ~86k slices) to the final paper.

```mermaid
flowchart TD
    ROOT["⚠️ Root cause:<br/>~138× class imbalance<br/>(67,222 Non Demented vs 488 Moderate)"]:::root

    %% ---------- Phase 8 (front-loaded) ----------
    subgraph P8["Phase 8 · Tools & Environment (front-loaded)"]
        direction LR
        P8S1["S1 Kaggle setup<br/>GPU · attach OASIS · quota"]
        P8S2["S2 Project scaffold<br/>folders · git · pinned reqs · seeds"]
        P8S1 --> P8S2
    end

    %% ---------- Phase 1 ----------
    subgraph P1["Phase 1 · Research Foundation (no code)"]
        direction TB
        P1S1["S1 Problem framing<br/>imbalanced 4-class MRI"]
        P1S2["S2 ADASYN + leakage rule<br/>(Pillar 1)"]:::adasyn
        P1S3["S3 Heterogeneous ensembles<br/>uncorrelated errors (Pillar 2)"]:::ens
        P1S4["S4 XAI landscape<br/>Grad-CAM · SHAP · LIME (Pillar 3)"]:::xai
        P1S5["S5 Gap analysis<br/>1-page gap vs Mahmud et al."]
        P1S1 --> P1S2 --> P1S3 --> P1S4 --> P1S5
    end

    %% ---------- Phase 2 ----------
    subgraph P2["Phase 2 · Architecture Selection"]
        direction TB
        P2S1["S1 Three backbones<br/>ResNet50 · EffNetB3 · DenseNet121"]
        P2S2["S2 Ensemble strategy<br/>weighted soft-voting by val-F1 (Pillar 2)"]:::ens
        P2S3["S3 4-class head<br/>+ sanity-load via timm"]
        P2S1 --> P2S2 --> P2S3
    end

    %% ---------- Phase 3 ----------
    subgraph P3["Phase 3 · Dataset & Preprocessing (code starts)"]
        direction TB
        P3S1["S1 OASIS EDA<br/>class distribution"]
        P3S2["S2 Preprocessing<br/>resize · gray→3ch · norm · augment"]
        P3S3["S3 Subject-grouped 80/10/10 split<br/>(patient-level) + PyTorch Dataset"]
        P3S4["S4 Apply ADASYN<br/>after split · train-only (Pillar 1)"]:::adasyn
        P3S5["S5 Verify balance<br/>eyeball synthetic samples"]
        P3S1 --> P3S2 --> P3S3 --> P3S4 --> P3S5
    end

    %% ---------- Phase 4 ----------
    subgraph P4["Phase 4 · Model Training"]
        direction TB
        P4S1["S1 Transfer learning<br/>freeze → fine-tune"]
        P4S2["S2 Training loop<br/>AMP · early stopping"]
        P4S3["S3 Train 3 models<br/>checkpoint (>85% val acc)"]
        P4S4["S4 Ensemble inference<br/>weighted soft-voting (Pillar 2)"]:::ens
        P4S1 --> P4S2 --> P4S3 --> P4S4
    end

    %% ---------- Phase 5 ----------
    subgraph P5["Phase 5 · Explainability (Pillar 3)"]
        direction TB
        P5S1["S1 Grad-CAM<br/>per-model + ensemble"]:::xai
        P5S2["S2 SHAP<br/>on ensemble predict()"]:::xai
        P5S3["S3 LIME<br/>superpixels"]:::xai
        P5S4["S4 XAI figures<br/>4-column + clinical check"]:::xai
        P5S1 --> P5S2 --> P5S3 --> P5S4
    end

    %% ---------- Phase 6 ----------
    subgraph P6["Phase 6 · Evaluation"]
        direction TB
        P6S1["S1 Metrics suite<br/>macro-F1 + per-class recall"]
        P6S2["S2 Confusion matrix<br/>+ one-vs-rest ROC"]
        P6S3["S3 Ablations<br/>no-ADASYN · same-family"]
        P6S4["S4 Comparison + significance<br/>McNemar / bootstrap"]
        P6S1 --> P6S2 --> P6S3 --> P6S4
    end

    %% ---------- Phase 7 ----------
    subgraph P7["Phase 7 · Paper Writing"]
        direction TB
        P7S1["S1 Structure + novelty"]
        P7S2["S2 Intro + Related Work"]
        P7S3["S3 Methodology"]
        P7S4["S4 Results · Discussion · Conclusion"]
        P7S5["S5 Figures · refs · submission"]
        P7S1 --> P7S2 --> P7S3 --> P7S4 --> P7S5
    end

    %% ---------- Main spine ----------
    ROOT --> P1S1
    P1 --> P2 --> P3 --> P4
    P4 --> P5
    P4 --> P6
    P5 --> P7
    P6 --> P7
    P8 -.front-loaded.-> P1

    %% ---------- Pillar threads (dotted) ----------
    P1S2 -.Pillar 1 ADASYN.-> P3S4
    P1S3 -.Pillar 2 ensemble.-> P2S2
    P2S2 -.weights → inference.-> P4S4
    P1S4 -.Pillar 3 XAI.-> P5S1

    %% ---------- Data / artifact lane ----------
    subgraph DATA["Data & Artifact Flow"]
        direction TB
        D1["raw OASIS<br/>data/raw/"]:::data
        D2["splits<br/>data/splits/"]:::data
        D3["ADASYN cache<br/>data/adasyn_cache/ (train-only)"]:::data
        D4["3 checkpoints<br/>checkpoints/"]:::data
        D5["test preds + probs<br/>ensemble predict()"]:::data
        D1 --> D2 --> D3 --> D4 --> D5
    end

    P3S3 -.-> D2
    P3S4 -.-> D3
    P4S3 -.-> D4
    P4S4 -.-> D5
    D5 -.consumed by.-> P5S2
    D5 -.consumed by.-> P6S1

    %% ---------- Styles ----------
    classDef root fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,color:#000;
    classDef adasyn fill:#fdebd0,stroke:#e67e22,stroke-width:1.5px,color:#000;
    classDef ens fill:#d6eaf8,stroke:#2e86c1,stroke-width:1.5px,color:#000;
    classDef xai fill:#e8daef,stroke:#8e44ad,stroke-width:1.5px,color:#000;
    classDef data fill:#eafaf1,stroke:#27ae60,stroke-width:1px,color:#000;
```

## Legend

| Element | Meaning |
|---|---|
| ⚠️ **Red node** | Root cause — the ~138× class imbalance every decision traces back to |
| 🟠 **Orange nodes** | **Pillar 1 — ADASYN** (designed P1·S2, applied P3·S4) |
| 🔵 **Blue nodes** | **Pillar 2 — Heterogeneous ensemble** / weighted soft-voting by val-F1 (P1·S3 → P2·S2 → P4·S4) |
| 🟣 **Purple nodes** | **Pillar 3 — Triple XAI** Grad-CAM + SHAP + LIME (surveyed P1·S4, produced Phase 5) |
| 🟢 **Green nodes** | Data/artifact lane — files passed between phases |
| **Solid arrow** | Sequential dependency (main phase/step order) |
| **`-.front-loaded.->`** | Phase 8 runs in parallel with Phase 1, not after Phase 7 |
| **Dotted arrow** | A pillar or artifact thread linking steps across phases |

## How to read it

1. **Top to bottom = chronology.** Start at the imbalance, set up the environment (Phase 8)
   while you build foundational understanding (Phase 1), then design (Phase 2), build the
   data (Phase 3), train (Phase 4), explain & evaluate in parallel (Phases 5 & 6), and write
   the paper (Phase 7).
2. **Color = pillar.** The orange/blue/purple nodes show how a single idea introduced in
   Phase 1 is *cashed out* later — e.g. the soft-voting weights designed in P2·S2 are the
   exact ones used at inference in P4·S4.
3. **Green lane = where data lives.** It makes the critical discipline visible: the split
   happens *before* ADASYN, ADASYN touches *train only*, and the ensemble's saved test
   predictions feed both XAI and metrics.
