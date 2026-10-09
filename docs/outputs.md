# Generated files

Output and file names are unchanged from the original scripts.

## Benchmark (`outputs/`, or `--output`)

```
outputs/
├── all_model_metrics.csv           # stage 6, every dataset × model
├── all_xai_metrics.csv             # stage 7
├── all_fewshot_zeroday.csv         # episodic few-shot / zero-day scores
├── all_zeroday_class_report.csv    # stage 6B
├── all_zeroday_detection.csv       # stage 6C (one row per model × novelty score)
├── comparison_figures/             # cross-dataset figures
└── <dataset>/
    ├── model_metrics.csv           # Accuracy, AUC, Precision, Recall, F1, Kappa, times
    ├── timing.csv                  # train / inference (ms per 1k flows) / XAI time
    ├── fewshot_zeroday_metrics.csv
    ├── zeroday_class_report.csv
    ├── zeroday_detection_metrics.csv
    ├── xai_metrics.csv
    ├── dataset_summary.json        # selected features, classes, zero-day families
    ├── meta_loss_histories.json, cnn_loss_history.json
    ├── figures/                    # confusion matrices, ROC, losses, zero-day plots, heatmaps
    └── xai/                        # SHAP/LIME figures, LIME HTML, shap_values_*.npy, lime_values_*.npy
```

Main CSV columns:

| File | Columns |
|---|---|
| `model_metrics.csv` | `Accuracy, Precision, Recall, F1, Kappa, AUC, Dataset, Model, Train_time_s, Eval_time_s, Eval_ms_per_1k, XAI_time_s` |
| `fewshot_zeroday_metrics.csv` | `{fewshot,zeroday}_episodic_accuracy`, `_episodic_f1_macro`, `_episodes`, `_n_way` |
| `zeroday_class_report.csv` | `Class, Support, DetectionRate, Precision, F1, Regime` |
| `zeroday_detection_metrics.csv` | `Score, Regime, N_known, N_zeroday, AUROC, AUPR, FPR@95TPR, TPR@FPR5, TPR@FPR10, BestF1, BestF1_Precision, BestF1_Recall, BestF1_Threshold` |
| `xai_metrics.csv` | `Model, Method, Fidelity, Infidelity, Sparsity, Stability, Comprehensiveness, Sufficiency, Time_s, FeatureAgreement, RankCorrelation` |

## Revision experiments (`outputs_revision/`, or `--output`)

```
outputs_revision/
├── hardware.json
├── protocol_audit.json, protocol_audit_classes.csv
├── E1_feature_sensitivity.csv
├── E2_trajectories.csv, E2_checkpoints.csv, revision_global_adapt_convergence.png
├── E3_calibration.csv
├── E4_latency.csv
├── E5_episodic_ci.csv
├── tables/   tab_ksens_rows.tex, tab_adapt_rows.tex, tab_calib_rows.tex,
│             tab_latency_rows.tex, tab_episodic_ci_rows.tex
└── <dataset>/E1_k<k>/          # open-set figures for each k of E1
```

Both output folders are ignored by git (`.gitignore`).
