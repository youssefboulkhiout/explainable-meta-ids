"""Global configuration: seed, device, hyper-parameters and dataset paths.

`CONFIG` and `DATASETS` below are the reference values used for the
manuscript. They can optionally be overridden from a YAML file
(`--config configs/default.yaml`) or by pointing to another data folder
(`--data-dir`); without these options the values below are used unchanged.
"""
import copy
import os

import torch


# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = {
    "output_dir": "outputs",
    "test_size": 0.20,          # 70 / 10 / 20
    "val_size":  0.10,
    "k_best": 30,               # SelectKBest features
    "min_class_size_to_keep": 50,
    "target_min_class_size": 10000,
    "max_class_size": 50000,
    "max_rows": 200000,         # stratified cap on raw rows (None = no cap)
    "min_class_presplit": 10,   # needed for stratified 70/10/20 split

    "cnn": {
        "epochs": 40, 
        "batch_size": 512, 
        "lr": 1e-3, 
        "patience": 5,
    },
    
    "meta": {
        "episodes": 500,        # meta-training episodes (more -> better convergence)
        "meta_batch": 4,        # tasks per meta-update (more -> stabler gradients)
        "n_way": 5,             # capped at number of available classes
        "k_shot": 20,           # more support examples per class per episode
        "q_query": 15,          # more query examples -> less noisy meta-gradient
        "inner_steps": 5,       # more inner-loop adaptation steps
        "inner_lr": 0.01,      # 0.001/0.01
        "meta_lr": 1e-3,        # lower, steadier outer LR for the longer run 5e-2/1e-3
        "reptile_lr": 0.05,     # inner Adam lr replaced by SGD lr 0.5/0.05
        "reptile_eps": 0.5,     # meta-interpolation (linearly annealed)
        "hidden": (128, 64),    # larger network capacity
        "adapt_shot": 20,       # support per class for global adaptation
        "adapt_steps": 50,      # fine-tuning steps for global adaptation 50/60
        "zero_day": True,       # hold out rare attack classes from meta-train
        "zero_day_n_classes": 2,
        "eval_episodes": 200,   # episodic few-shot / zero-day evaluation
    },

    "xai": {
        "n_background": 100,    # background samples for SHAP / LIME
        "n_explain": 40,        # instances explained per model/method
        "kernel_nsamples": 150, # SHAP KernelExplainer budget
        "lime_num_samples": 1000,
        "top_k_list": [5, 10, 15],
        "stability_instances": 5,
        "stability_perturbs": 3,
        "stability_sigma": 0.05,
        "infidelity_perturbs": 20,
        "infidelity_sigma": 0.3,
    },

    # ---- unified zero-day (open-set / novelty) detection (pipeline stage 6C) --
    # Settings that control the model-agnostic "is this an unseen attack?"
    # detector applied identically to every base learner and meta-learner.
    "zeroday_detect": {
        "enabled": True,                # master on/off switch for stage 6C
        "score_methods": ["msp", "entropy", "margin"],  # novelty scores to test
        "target_fprs": [0.05, 0.10],    # report detection rate (TPR) at these FPRs
        "retrain_base_known_only": True,  # re-fit LightGBM/CNN WITHOUT zero-day
        "max_test_known": 20000,        # cap known negatives when scoring (None=all)
    },
}


# Per-dataset configuration: edit "path" entries to match your local files.
DATASETS = {
    "ASEADOS_SDN_IoT": {
        "path": "data/ASEADOS_SDN_IoT.csv",
        "label_candidates": ["label", "class", "attack_type", "category"],
        "drop": ["id", "flow_id", ],
    },
    "RT-IoT2022": {
        "path": "data/RT-IoT2022.csv",
        "label_candidates": ["attack_type", "label", "class"],
        "drop": ["no.", "no", "id", "unnamed: 0"],
    },
    "UNSW-NB15": {
        "path": "data/UNSW-NB15.csv",
        "label_candidates": ["attack_cat"],
        "drop": ["id", "label", ],
    },
}


GENERIC_DROP_COLUMNS = {
    "id", "no", "no.", "index", "unnamed: 0", "flow id", "flow_id",
}


BASE_MODELS = ["LightGBM", "CNN"]


META_MODELS = ["Reptile", "MAML", "FOMAML", "Meta-SGD", "ANIL", "BOIL", "ProtoNet"] #


ALL_MODELS = BASE_MODELS + META_MODELS


# =============================================================================
# OPTIONAL OVERRIDES (YAML file / data directory)
# =============================================================================
def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base` (in place) and return it."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, base=None):
    """Return a deep copy of `base` (default: CONFIG) updated with the
    `pipeline:` section of the YAML file at `path`. Dataset entries found in
    its `datasets:` section are merged into the global DATASETS registry.
    With `path=None`, the reference configuration is returned unchanged."""
    cfg = copy.deepcopy(CONFIG if base is None else base)
    if path is None:
        return cfg
    import yaml                                  # optional dependency
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    _deep_update(cfg, doc.get("pipeline", {}) or {})
    if isinstance(cfg.get("meta", {}).get("hidden"), list):   # YAML has no tuples
        cfg["meta"]["hidden"] = tuple(cfg["meta"]["hidden"])
    for name, entry in (doc.get("datasets", {}) or {}).items():
        DATASETS.setdefault(name, {})
        _deep_update(DATASETS[name], entry or {})
    return cfg


def set_data_dir(data_dir):
    """Point every registered dataset to `<data_dir>/<file name>` (in place)."""
    for entry in DATASETS.values():
        entry["path"] = os.path.join(data_dir, os.path.basename(entry["path"]))
