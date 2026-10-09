"""E1 (Reviewer 1, C4) -- sensitivity to the number of ANOVA features k and
stability of the selected feature set."""
import os
import copy

import numpy as np
from sklearn.metrics import f1_score

import meta_ids as P                        # benchmark pipeline (unchanged)
from .preparation import OUTPUT_DIR, prepare_dataset


# =============================================================================
# E1 -- sensitivity to k and stability of the ANOVA selection
# =============================================================================
def feature_selection_stability(ds_name, cfg, k, n_boot=10, frac=0.8):
    """Mean pairwise Jaccard index of the top-k ANOVA feature sets selected on
    n_boot stratified 80% sub-samples of the TRAINING split (no model fit)."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split
    P.set_seed()
    X_df, y, _ = P.load_and_clean_dataset(ds_name, P.DATASETS[ds_name], cfg)
    (X_tr, y_tr), _, _, _ = P.split_and_standardize(X_df, y, cfg)
    k = min(int(k), X_tr.shape[1])
    sets = []
    for b in range(n_boot):
        idx, _ = train_test_split(np.arange(len(y_tr)), train_size=frac,
                                  stratify=y_tr, random_state=P.SEED + b)
        s = SelectKBest(f_classif, k=k).fit(X_tr[idx], y_tr[idx])
        sets.append(set(s.get_support(indices=True)))
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets)
           for b in sets[i + 1:]]
    return float(np.mean(jac)), float(np.std(jac))


def run_e1_feature_sensitivity(ds_name, cfg, k_list, models):
    rows = []
    for k in k_list:
        P.log(f"[E1] {ds_name} k={k}")
        D = prepare_dataset(ds_name, cfg, k=k)
        kk = len(D["feat_names"])
        jac_m, jac_s = feature_selection_stability(ds_name, cfg, kk)
        r = {"Dataset": ds_name, "k": kk, "Jaccard_mean": jac_m,
             "Jaccard_std": jac_s}
        # LightGBM (closed set + open set)
        lgbm, _ = P.train_lightgbm(D["X_tr"], D["y_tr"], D["X_va"], D["y_va"])
        pp = P.expand_to_full_label_space(lgbm, D["n_classes"])
        pred = np.argmax(pp(D["X_te"]), 1)
        r["LightGBM_F1"] = f1_score(D["y_te"], pred, average="macro",
                                    zero_division=0)
        meta_state = {}
        for algo in models:
            params, lr_dict, _, _ = P.meta_train(
                algo, D["X_tr"], D["y_tr"], D["meta_pool"], kk,
                D["n_classes"], cfg["meta"])
            meta_state[algo] = (params, lr_dict)
            w = P.global_few_shot_adapt(algo, params, lr_dict, D["X_tr"], D["y_tr"],
                               D["n_classes"], cfg["meta"])
            r[f"{algo}_F1"] = f1_score(D["y_te"], w.predict(D["X_te"]),
                                       average="macro", zero_division=0)
            zd = P.evaluate_episodic(algo, params, lr_dict, D["X_te"], D["y_te"],
                                 D["zero_day"] + D["benign_ids"][:1],
                                 cfg["meta"], tag="zeroday")
            r[f"{algo}_ZD_F1"] = zd.get("zeroday_episodic_f1_macro", np.nan)
        # open-set AUROC (MSP) for LightGBM + the requested meta-learners
        c2 = copy.deepcopy(cfg)
        c2["zeroday_detect"]["retrain_base_known_only"] = True
        det = P.run_open_set_detection(
            ds_name, meta_state, D["X_tr"], D["y_tr"], D["X_va"], D["y_va"],
            D["X_te"], D["y_te"], D["meta_pool"], D["zero_day"],
            D["n_classes"], D["class_names"], c2,
            P.ensure_dir(os.path.join(OUTPUT_DIR, ds_name, f"E1_k{kk}")),
            P.ensure_dir(os.path.join(OUTPUT_DIR, ds_name, f"E1_k{kk}")))
        for d in det:
            if d["Score"] == "MSP":
                r[f"{d['Model']}_AUROC"] = d["AUROC"]
        rows.append(r)
    return rows
