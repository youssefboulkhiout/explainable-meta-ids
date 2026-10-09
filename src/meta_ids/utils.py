"""Generic helpers (seeding, logging, I/O, figure saving, probability guard)
and the shared Matplotlib / seaborn setup used by every plotting module."""
import os
import json
import time
import random
import warnings

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.9)
    HAS_SNS = True
except Exception:
    sns = None
    HAS_SNS = False

from .config import SEED

warnings.filterwarnings("ignore")


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed: int = SEED):
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    """Create `path` (and parents) if missing and return it, for chaining."""
    os.makedirs(path, exist_ok=True)
    return path


def log(msg: str):
    """Timestamped, immediately-flushed stdout logger used throughout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(obj, path):
    """Dump `obj` to `path` as UTF-8 JSON (non-serialisable -> str())."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_figure(fig, path):
    """Save a Matplotlib figure at 200 DPI with a tight bounding box, then
    close it to free memory (important given the many figures produced)."""
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sanitize_probabilities(p):
    """Replace NaN/+-inf with 0 and renormalise each row to a valid
    probability distribution. A diverged meta-learner (or a softmax over
    inf/NaN logits) can emit non-finite probabilities; downstream sklearn
    metrics (roc_curve, roc_auc_score, ...) reject non-finite input, so every
    predict_proba is passed through this guard before use."""
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum(axis=1, keepdims=True)
    # rows that summed to 0 (all-NaN) fall back to a uniform distribution
    p = np.where(s > 0, p / np.maximum(s, 1e-12), 1.0 / max(p.shape[1], 1))
    return p
