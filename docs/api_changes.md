# Renamed files and identifiers

Version 1.0.0 of the repository reorganises the two original scripts
(`xai_meta_5.py`, `revision_experiments.py`) into the `meta_ids` package.
Only **file names and Python identifiers** changed: no computation,
hyper-parameter, random seed, output file name or CSV column changed, and the
renamed code produces the same numbers as the original scripts
(see [reproducibility.md](reproducibility.md#equivalence-with-the-original-scripts)).

## Files

| Original | New location |
|---|---|
| `xai_meta_5.py` (stage 0, configuration) | `src/meta_ids/config.py` |
| `xai_meta_5.py` (utilities) | `src/meta_ids/utils.py` |
| `xai_meta_5.py` (stages 1-4) | `src/meta_ids/data.py` |
| `xai_meta_5.py` (stage 5A) | `src/meta_ids/models/baselines.py` |
| `xai_meta_5.py` (stage 5B) | `src/meta_ids/models/meta_learning.py` |
| `xai_meta_5.py` (stages 6, 6B) | `src/meta_ids/evaluation/classification.py` |
| `xai_meta_5.py` (stage 6C) | `src/meta_ids/evaluation/open_set.py` |
| `xai_meta_5.py` (stage 7) | `src/meta_ids/explainability.py` |
| `xai_meta_5.py` (stage 8) | `src/meta_ids/reporting.py` |
| `xai_meta_5.py` (`run_dataset`, `main`) | `src/meta_ids/pipeline.py`, launched by `scripts/run_benchmark.py` |
| `revision_experiments.py` (shared preparation, audit) | `src/meta_ids/revision/preparation.py` |
| `revision_experiments.py` (E1) | `src/meta_ids/revision/e1_feature_sensitivity.py` |
| `revision_experiments.py` (E2) | `src/meta_ids/revision/e2_adaptation_budget.py` |
| `revision_experiments.py` (E3) | `src/meta_ids/revision/e3_calibration.py` |
| `revision_experiments.py` (E4) | `src/meta_ids/revision/e4_latency.py` |
| `revision_experiments.py` (E5) | `src/meta_ids/revision/e5_episodic_ci.py` |
| `revision_experiments.py` (LaTeX tables) | `src/meta_ids/revision/latex_tables.py` |
| `revision_experiments.py` (`main`) | `src/meta_ids/revision/cli.py`, launched by `scripts/run_revision.py` |

## Identifiers

Names not listed here are unchanged (e.g. `CONFIG`, `DATASETS`, `SEED`, `meta_train`, `train_lightgbm`, `train_cnn`, `CNN1D`, `sample_episode`, `xai_fidelity`, ...).

### Benchmark pipeline (`import meta_ids as P`)

| Original | New | Module |
|---|---|---|
| `savefig` | `save_figure` | `meta_ids.utils` |
| `sanitize_proba` | `sanitize_probabilities` | `meta_ids.utils` |
| `GENERIC_DROP_EXACT` | `GENERIC_DROP_COLUMNS` | `meta_ids.config` |
| `load_and_clean` | `load_and_clean_dataset` | `meta_ids.data` |
| `split_and_scale` | `split_and_standardize` | `meta_ids.data` |
| `select_features` | `select_kbest_features` | `meta_ids.data` |
| `balance_classes` | `balance_training_classes` | `meta_ids.data` |
| `make_full_proba` | `expand_to_full_label_space` | `meta_ids.models.baselines` |
| `TorchWrapper` | `TorchClassifierWrapper` | `meta_ids.models.baselines` |
| `BODY_KEYS` | `BODY_PARAM_KEYS` | `meta_ids.models.meta_learning` |
| `HEAD_KEYS` | `HEAD_PARAM_KEYS` | `meta_ids.models.meta_learning` |
| `init_meta_params` | `init_mlp_params` | `meta_ids.models.meta_learning` |
| `body_forward` | `mlp_embed` | `meta_ids.models.meta_learning` |
| `full_forward` | `mlp_logits` | `meta_ids.models.meta_learning` |
| `inner_adapt` | `inner_loop_adapt` | `meta_ids.models.meta_learning` |
| `proto_logits` | `prototype_logits` | `meta_ids.models.meta_learning` |
| `MetaWrapper` | `MetaLearnerClassifier` | `meta_ids.models.meta_learning` |
| `global_adapt` | `global_few_shot_adapt` | `meta_ids.models.meta_learning` |
| `episodic_eval` | `evaluate_episodic` | `meta_ids.models.meta_learning` |
| `compute_metrics` | `compute_classification_metrics` | `meta_ids.evaluation.classification` |
| `plot_confusion` | `plot_confusion_matrix` | `meta_ids.evaluation.classification` |
| `plot_roc` | `plot_roc_curves` | `meta_ids.evaluation.classification` |
| `plot_lgbm_importance` | `plot_lightgbm_importance` | `meta_ids.evaluation.classification` |
| `plot_cnn_loss` | `plot_cnn_learning_curves` | `meta_ids.evaluation.classification` |
| `_smooth` | `_moving_average` | `meta_ids.evaluation.classification` |
| `plot_meta_losses` | `plot_meta_training_losses` | `meta_ids.evaluation.classification` |
| `plot_time_comparison` | `plot_runtime_comparison` | `meta_ids.evaluation.classification` |
| `zeroday_global_report` | `zero_day_per_class_report` | `meta_ids.evaluation.classification` |
| `plot_zeroday_figures` | `plot_zero_day_figures` | `meta_ids.evaluation.classification` |
| `novelty_scores` | `compute_novelty_scores` | `meta_ids.evaluation.open_set` |
| `openset_detection_metrics` | `open_set_detection_metrics` | `meta_ids.evaluation.open_set` |
| `_slice_renorm` | `_slice_and_renormalize` | `meta_ids.evaluation.open_set` |
| `build_known_only_models` | `build_known_class_models` | `meta_ids.evaluation.open_set` |
| `run_zeroday_detection` | `run_open_set_detection` | `meta_ids.evaluation.open_set` |
| `plot_zeroday_detection` | `plot_open_set_detection` | `meta_ids.evaluation.open_set` |
| `plot_openset_metric_suite` | `plot_open_set_metric_suite` | `meta_ids.evaluation.open_set` |
| `shap_explain` | `explain_with_shap` | `meta_ids.explainability` |
| `lime_explain` | `explain_with_lime` | `meta_ids.explainability` |
| `lime_instance_figures` | `plot_lime_instances` | `meta_ids.explainability` |
| `shap_figures` | `plot_shap_summaries` | `meta_ids.explainability` |
| `lime_figure` | `plot_lime_global_importance` | `meta_ids.explainability` |
| `run_xai_for_model` | `explain_model` | `meta_ids.explainability` |
| `heatmap_df` | `plot_heatmap` | `meta_ids.reporting` |
| `dataset_summary_figures` | `plot_dataset_summary` | `meta_ids.reporting` |
| `cross_dataset_figures` | `plot_cross_dataset_comparison` | `meta_ids.reporting` |
| `run_dataset` | `run_benchmark_on_dataset` | `meta_ids.pipeline` |

### Revision experiments (`meta_ids.revision`)

| Original | New | Module |
|---|---|---|
| `OUT` | `OUTPUT_DIR` | `meta_ids.revision.preparation` |
| `GRAD_MODELS` | `GRADIENT_META_MODELS` | `meta_ids.revision.preparation` |
| `prepare` | `prepare_dataset` | `meta_ids.revision.preparation` |
| `protocol_audit` | `audit_zero_day_protocol` | `meta_ids.revision.preparation` |
| `selection_stability` | `feature_selection_stability` | `meta_ids.revision.e1_feature_sensitivity` |
| `run_E1` | `run_e1_feature_sensitivity` | `meta_ids.revision.e1_feature_sensitivity` |
| `adapt_trajectory` | `global_adaptation_trajectory` | `meta_ids.revision.e2_adaptation_budget` |
| `run_E2` | `run_e2_adaptation_budget` | `meta_ids.revision.e2_adaptation_budget` |
| `plot_E2` | `plot_e2_convergence` | `meta_ids.revision.e2_adaptation_budget` |
| `known_only_logits` | `known_class_log_probabilities` | `meta_ids.revision.e3_calibration` |
| `_lse` | `_logsumexp` | `meta_ids.revision.e3_calibration` |
| `softmax_T` | `tempered_softmax` | `meta_ids.revision.e3_calibration` |
| `ece` | `expected_calibration_error` | `meta_ids.revision.e3_calibration` |
| `run_E3` | `run_e3_calibration` | `meta_ids.revision.e3_calibration` |
| `hardware_report` | `collect_hardware_info` | `meta_ids.revision.e4_latency` |
| `run_E4` | `run_e4_latency` | `meta_ids.revision.e4_latency` |
| `episodic_eval_ci` | `evaluate_episodic_with_ci` | `meta_ids.revision.e5_episodic_ci` |
| `run_E5` | `run_e5_episodic_ci` | `meta_ids.revision.e5_episodic_ci` |
| `f3` | `_fmt3` | `meta_ids.revision.latex_tables` |
| `latex_tables` | `write_latex_tables` | `meta_ids.revision.latex_tables` |

## New (additive) features

- `load_config(path, base)` and `set_data_dir(folder)` in `meta_ids.config`, used by the new `--config` and `--data-dir` command-line options. Without these options the reference `CONFIG` / `DATASETS` are used exactly as before.
- `scripts/make_synthetic_dataset.py`: tiny synthetic CSV for installation checks and CI (no scientific meaning).
- In the revision CLI, `--output` is forwarded to E1 explicitly (`_e1.OUTPUT_DIR = OUTPUT_DIR`) because E1 now lives in its own module; E1 still writes to the requested folder, as before.
