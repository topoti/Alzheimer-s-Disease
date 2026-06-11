# Learning Curriculum — Alzheimer's Cross-Paradigm Ensemble

This folder breaks `../RESEARCH_PLAN.md` into **fine-grained, do-one-at-a-time steps**.
Each file teaches one concept in plain language, lists a few resources, and ends with a
short **"Your task"** checklist that verifies you understood it before moving on. Work
through them in order; each file points to the next.

> **Tip:** Phase 8 (Tools & Environment) is *front-loaded* — do it in parallel with Phase 1,
> even though it's listed last.

## Phase 1 — Research Foundation (understanding, no code)
| Step | File | Scope |
|---|---|---|
| 1 | [phase1_step1_problem_framing.md](phase1_step1_problem_framing.md) | Frame the problem; the 50× class imbalance |
| 2 | [phase1_step2_understanding_adasyn.md](phase1_step2_understanding_adasyn.md) | Augmentation → SMOTE → ADASYN; the leakage rule (Pillar 1) |
| 3 | [phase1_step3_heterogeneous_ensembles.md](phase1_step3_heterogeneous_ensembles.md) | Uncorrelated-error theory; cross-paradigm vs cross-architecture (Pillar 2) |
| 4 | [phase1_step4_xai_landscape.md](phase1_step4_xai_landscape.md) | Grad-CAM vs SHAP vs LIME; why three (Pillar 3) |
| 5 | [phase1_step5_gap_analysis.md](phase1_step5_gap_analysis.md) | Gap vs Mahmud et al.; the 1-page gap statement |

## Phase 2 — Architecture Selection
| Step | File | Scope |
|---|---|---|
| 1 | [phase2_step1_three_backbones.md](phase2_step1_three_backbones.md) | ResNet50 / EfficientNetB3 / DenseNet121 paradigms + input sizes |
| 2 | [phase2_step2_ensemble_strategy.md](phase2_step2_ensemble_strategy.md) | Hard vs soft vs stacking → weighted soft voting by val-F1 |
| 3 | [phase2_step3_head_and_loading.md](phase2_step3_head_and_loading.md) | 4-class head design; sanity-load all 3 via `timm` |

## Phase 3 — Dataset & Preprocessing (code starts)
| Step | File | Scope |
|---|---|---|
| 1 | [phase3_step1_oasis_eda.md](phase3_step1_oasis_eda.md) | OASIS dataset + class-distribution EDA |
| 2 | [phase3_step2_preprocessing_pipeline.md](phase3_step2_preprocessing_pipeline.md) | Resize, grayscale→3ch, normalize, noise, crop, augment |
| 3 | [phase3_step3_split_and_dataset.md](phase3_step3_split_and_dataset.md) | Stratified 80/10/10 split + PyTorch `Dataset` |
| 4 | [phase3_step4_adasyn_application.md](phase3_step4_adasyn_application.md) | ADASYN after split, train-only; pixel-space Option A |
| 5 | [phase3_step5_verify_balance.md](phase3_step5_verify_balance.md) | Verify balance; eyeball synthetic samples |

## Phase 4 — Model Training
| Step | File | Scope |
|---|---|---|
| 1 | [phase4_step1_transfer_learning.md](phase4_step1_transfer_learning.md) | Two-stage freeze-then-fine-tune |
| 2 | [phase4_step2_training_loop.md](phase4_step2_training_loop.md) | `train_one_model()`, hyperparameters, AMP, early stopping |
| 3 | [phase4_step3_train_three_models.md](phase4_step3_train_three_models.md) | Train + checkpoint all three (>85% val acc) |
| 4 | [phase4_step4_ensemble_inference.md](phase4_step4_ensemble_inference.md) | Weighted soft-voting inference over 3 checkpoints |

## Phase 5 — XAI
| Step | File | Scope |
|---|---|---|
| 1 | [phase5_step1_gradcam.md](phase5_step1_gradcam.md) | Per-model + averaged ensemble Grad-CAM |
| 2 | [phase5_step2_shap.md](phase5_step2_shap.md) | SHAP on the ensemble predict-function (~20 images) |
| 3 | [phase5_step3_lime.md](phase5_step3_lime.md) | LIME superpixels on the same images |
| 4 | [phase5_step4_xai_figures.md](phase5_step4_xai_figures.md) | 4-column figure + clinical sanity check |

## Phase 6 — Evaluation
| Step | File | Scope |
|---|---|---|
| 1 | [phase6_step1_metrics_suite.md](phase6_step1_metrics_suite.md) | Why macro-F1 + per-class recall over accuracy; `evaluate()` |
| 2 | [phase6_step2_confusion_and_roc.md](phase6_step2_confusion_and_roc.md) | Confusion matrix + one-vs-rest ROC |
| 3 | [phase6_step3_ablations.md](phase6_step3_ablations.md) | No-ADASYN + same-family ablations |
| 4 | [phase6_step4_comparison_and_significance.md](phase6_step4_comparison_and_significance.md) | Comparison table + McNemar / bootstrap |

## Phase 7 — Paper Writing
| Step | File | Scope |
|---|---|---|
| 1 | [phase7_step1_structure_and_novelty.md](phase7_step1_structure_and_novelty.md) | Section outline + novelty-positioning paragraph |
| 2 | [phase7_step2_intro_and_related_work.md](phase7_step2_intro_and_related_work.md) | Draft Introduction + Related Work |
| 3 | [phase7_step3_methodology.md](phase7_step3_methodology.md) | Draft Methodology (reproducible) |
| 4 | [phase7_step4_results_discussion_conclusion.md](phase7_step4_results_discussion_conclusion.md) | Draft Results, Discussion (limits), Conclusion |
| 5 | [phase7_step5_figures_refs_submission.md](phase7_step5_figures_refs_submission.md) | Figures, references, review, submission |

## Phase 8 — Tools & Environment (front-loaded; do alongside Phase 1)
| Step | File | Scope |
|---|---|---|
| 1 | [phase8_step1_kaggle_setup.md](phase8_step1_kaggle_setup.md) | Kaggle account, GPU, attach OASIS, quota budget |
| 2 | [phase8_step2_project_scaffold.md](phase8_step2_project_scaffold.md) | Folder structure, Git, pinned requirements, seeds |
