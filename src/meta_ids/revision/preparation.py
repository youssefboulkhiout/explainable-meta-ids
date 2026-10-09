"""Shared preparation (identical to run_benchmark_on_dataset stages 1-4 + zero-day
choice) and the zero-day protocol audit."""
from collections import Counter

import numpy as np

import meta_ids as P                        # benchmark pipeline (unchanged)


OUTPUT_DIR = "outputs_revision"


GRADIENT_META_MODELS = ["Reptile", "MAML", "FOMAML", "Meta-SGD", "ANIL", "BOIL"]


# =============================================================================
# Shared preparation (identical to run_benchmark_on_dataset stages 1-4 + zero-day choice)
# =============================================================================
def prepare_dataset(ds_name, cfg, k=None):
    """Replicates run_benchmark_on_dataset() stages 1-4 and the zero-day selection, with an
    optional override of the number of SelectKBest features k."""
    P.set_seed()
    dcfg = P.DATASETS[ds_name]
    X_df, y, le = P.load_and_clean_dataset(ds_name, dcfg, cfg)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), _ = P.split_and_standardize(X_df, y, cfg)
    k = cfg["k_best"] if k is None else k
    if k == "all":
        k = X_tr.shape[1]
    X_tr, X_va, X_te, feat_names, selector = P.select_kbest_features(
        X_tr, y_tr, X_va, X_te, list(X_df.columns), int(k))
    y_tr_unbal = y_tr.copy()
    X_tr, y_tr, kept = P.balance_training_classes(X_tr, y_tr, cfg)
    X_va, y_va = X_va[np.isin(y_va, kept)], y_va[np.isin(y_va, kept)]
    X_te, y_te = X_te[np.isin(y_te, kept)], y_te[np.isin(y_te, kept)]
    n_classes = len(le.classes_)
    class_names = [str(c) for c in le.classes_]
    mcfg = cfg["meta"]
    train_counts = Counter(y_tr)
    benign_ids = [i for i, c in enumerate(le.classes_)
                  if any(t in str(c).lower() for t in ("benign", "normal"))]
    cands = sorted([c for c in kept if c not in benign_ids],
                   key=lambda c: train_counts[c])
    zero_day = cands[:mcfg["zero_day_n_classes"]] if mcfg["zero_day"] else []
    meta_pool = np.array([c for c in kept if c not in zero_day])
    return dict(X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te,
                y_te=y_te, n_classes=n_classes, class_names=class_names,
                feat_names=feat_names, zero_day=list(zero_day),
                meta_pool=meta_pool, benign_ids=benign_ids, kept=kept,
                y_tr_unbal=y_tr_unbal, selector=selector,
                n_raw_features=X_df.shape[1])


def audit_zero_day_protocol(ds_name, D):
    """Records facts about the zero-day protocol that the manuscript reports:
    which families were withheld, their balanced/unbalanced training counts
    (ties at the SMOTE target are broken by label index), and the size of the
    zero-day episodes (2-way if no class name contains 'benign'/'normal')."""
    cn = D["class_names"]
    cnt_bal, cnt_raw = Counter(D["y_tr"]), Counter(D["y_tr_unbal"])
    rows = []
    for c in D["kept"]:
        rows.append({"Dataset": ds_name, "Class": cn[c],
                     "train_count_before_balancing": int(cnt_raw[c]),
                     "train_count_after_balancing": int(cnt_bal[c]),
                     "test_count": int(np.sum(D["y_te"] == c)),
                     "is_benign_by_name": c in D["benign_ids"],
                     "zero_day": c in D["zero_day"]})
    pool = D["zero_day"] + D["benign_ids"][:1]
    info = {"dataset": ds_name, "zero_day": [cn[c] for c in D["zero_day"]],
            "benign_class_used_in_zero_day_episodes":
                [cn[c] for c in D["benign_ids"][:1]],
            "zero_day_episode_n_way": len(pool),
            "n_raw_features_after_cleaning": D["n_raw_features"],
            "test_size": int(len(D["y_te"]))}
    return rows, info
