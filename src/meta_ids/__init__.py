"""
meta_ids -- Explainable Meta-Learning for Few-Shot and Zero-Day Intrusion
Detection in IoT Networks (benchmark pipeline).

Datasets      : RT-IoT2022, UNSW-NB15, ASEADOS-SDN-IoT
Base learners : LightGBM, 1-D CNN
Meta-learners : Reptile, MAML, FOMAML, Meta-SGD, ANIL, BOIL, ProtoNet
XAI           : SHAP + LIME with 8 explanation-quality metrics

Package layout (one module per pipeline stage)
----------------------------------------------
  config                     stage 0  -- CONFIG, DATASETS, seed, device
  data                       stages 1-4 -- load/clean, split+scale, SelectKBest, balancing
  models.baselines           stage 5A -- LightGBM, 1-D CNN
  models.meta_learning       stage 5B -- 7 meta-learners, global adaptation, episodic eval
  evaluation.classification  stages 6/6B -- metrics, plots, per-class zero-day stats
  evaluation.open_set        stage 6C -- unified open-set (novelty) zero-day detection
  explainability             stage 7  -- SHAP + LIME + 8 quality metrics
  reporting                  stage 8  -- summary / cross-dataset figures
  pipeline                   run_benchmark_on_dataset() + main() (CLI)
  revision                   reviewer experiments E1-E5 (see meta_ids.revision)

Every public name is re-exported here, so ``import meta_ids as P`` gives a
flat namespace (``P.meta_train``, ``P.CONFIG``, ...).
"""
__version__ = "1.0.0"

from .config import (  # noqa: F401
    SEED,
    DEVICE,
    CONFIG,
    DATASETS,
    GENERIC_DROP_COLUMNS,
    BASE_MODELS,
    META_MODELS,
    ALL_MODELS,
    _deep_update,
    load_config,
    set_data_dir,
)
from .utils import (  # noqa: F401
    HAS_SNS,
    set_seed,
    ensure_dir,
    log,
    save_json,
    save_figure,
    sanitize_probabilities,
)
from .data import (  # noqa: F401
    detect_label_column,
    load_and_clean_dataset,
    split_and_standardize,
    select_kbest_features,
    balance_training_classes,
)
from .models.baselines import (  # noqa: F401
    train_lightgbm,
    expand_to_full_label_space,
    CNN1D,
    train_cnn,
    TorchClassifierWrapper,
)
from .models.meta_learning import (  # noqa: F401
    BODY_PARAM_KEYS,
    HEAD_PARAM_KEYS,
    init_mlp_params,
    mlp_embed,
    mlp_logits,
    clone_params,
    sample_episode,
    inner_loop_adapt,
    prototype_logits,
    meta_train,
    MetaLearnerClassifier,
    global_few_shot_adapt,
    evaluate_episodic,
)
from .evaluation.classification import (  # noqa: F401
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_lightgbm_importance,
    plot_cnn_learning_curves,
    _moving_average,
    plot_meta_training_losses,
    plot_runtime_comparison,
    zero_day_per_class_report,
    plot_zero_day_figures,
)
from .evaluation.open_set import (  # noqa: F401
    compute_novelty_scores,
    open_set_detection_metrics,
    _slice_and_renormalize,
    build_known_class_models,
    run_open_set_detection,
    plot_open_set_detection,
    plot_open_set_metric_suite,
)
from .explainability import (  # noqa: F401
    _extract_class_attributions,
    explain_with_shap,
    explain_with_lime,
    plot_lime_instances,
    _prob_of_class,
    xai_fidelity,
    xai_infidelity,
    xai_sparsity,
    xai_stability,
    xai_comp_suff,
    xai_agreement,
    xai_rank_correlation,
    plot_shap_summaries,
    plot_lime_global_importance,
    explain_model,
)
from .reporting import (  # noqa: F401
    plot_heatmap,
    plot_dataset_summary,
    _ordered_pivot,
    plot_cross_dataset_comparison,
)
from .pipeline import (  # noqa: F401
    run_benchmark_on_dataset,
    main,
)

__all__ = [
    "SEED",
    "DEVICE",
    "CONFIG",
    "DATASETS",
    "GENERIC_DROP_COLUMNS",
    "BASE_MODELS",
    "META_MODELS",
    "ALL_MODELS",
    "load_config",
    "set_data_dir",
    "HAS_SNS",
    "set_seed",
    "ensure_dir",
    "log",
    "save_json",
    "save_figure",
    "sanitize_probabilities",
    "detect_label_column",
    "load_and_clean_dataset",
    "split_and_standardize",
    "select_kbest_features",
    "balance_training_classes",
    "train_lightgbm",
    "expand_to_full_label_space",
    "CNN1D",
    "train_cnn",
    "TorchClassifierWrapper",
    "BODY_PARAM_KEYS",
    "HEAD_PARAM_KEYS",
    "init_mlp_params",
    "mlp_embed",
    "mlp_logits",
    "clone_params",
    "sample_episode",
    "inner_loop_adapt",
    "prototype_logits",
    "meta_train",
    "MetaLearnerClassifier",
    "global_few_shot_adapt",
    "evaluate_episodic",
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_lightgbm_importance",
    "plot_cnn_learning_curves",
    "plot_meta_training_losses",
    "plot_runtime_comparison",
    "zero_day_per_class_report",
    "plot_zero_day_figures",
    "compute_novelty_scores",
    "open_set_detection_metrics",
    "build_known_class_models",
    "run_open_set_detection",
    "plot_open_set_detection",
    "plot_open_set_metric_suite",
    "explain_with_shap",
    "explain_with_lime",
    "plot_lime_instances",
    "xai_fidelity",
    "xai_infidelity",
    "xai_sparsity",
    "xai_stability",
    "xai_comp_suff",
    "xai_agreement",
    "xai_rank_correlation",
    "plot_shap_summaries",
    "plot_lime_global_importance",
    "explain_model",
    "plot_heatmap",
    "plot_dataset_summary",
    "plot_cross_dataset_comparison",
    "run_benchmark_on_dataset",
    "main",
]
