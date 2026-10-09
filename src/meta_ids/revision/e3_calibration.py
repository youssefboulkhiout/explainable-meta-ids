"""E3 (Reviewer 1, C6) -- post-hoc calibration (temperature scaling, isotonic)
and its effect on ECE and open-set detection."""
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

import meta_ids as P                        # benchmark pipeline (unchanged)
from .preparation import prepare_dataset


# =============================================================================
# E3 -- post-hoc calibration and open-set detection
# =============================================================================
def known_class_log_probabilities(meta_state, D, cfg):
    """Returns {model: f(X) -> (n, K) LOG-probabilities over the known classes}
    built exactly like P.build_known_class_models (LightGBM/CNN re-fit on known
    classes; meta-learners re-adapted on known classes and sliced). Working in
    log space makes temperature scaling exact: softmax(log p / T) equals
    softmax(z / T) for any logits z with p = softmax(z)."""
    known = sorted(int(c) for c in D["meta_pool"])
    remap = {c: i for i, c in enumerate(known)}
    K = len(known)
    m_tr, m_va = np.isin(D["y_tr"], known), np.isin(D["y_va"], known)
    Xk_tr, yk_tr = D["X_tr"][m_tr], np.array([remap[c] for c in D["y_tr"][m_tr]])
    Xk_va, yk_va = D["X_va"][m_va], np.array([remap[c] for c in D["y_va"][m_va]])
    out = {}
    lgbm, _ = P.train_lightgbm(Xk_tr, yk_tr, Xk_va, yk_va)
    cls = np.asarray(lgbm.classes_, int)

    def lgbm_logp(X, m=lgbm, cls=cls):
        raw = m.predict_proba(np.asarray(X), raw_score=True)   # (n, |cls|)
        if raw.ndim == 1:                     # binary booster: log-odds
            raw = np.c_[np.zeros_like(raw), raw]
        full = np.full((len(X), K), -1e9); full[:, cls] = raw
        return full - _logsumexp(full)
    out["LightGBM"] = lgbm_logp
    cnn, _, _ = P.train_cnn(Xk_tr, yk_tr, Xk_va, yk_va, K, cfg["cnn"])

    def cnn_logp(X, m=cnn):
        X = np.asarray(X, np.float32); z = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                z.append(m(torch.as_tensor(X[i:i + 4096]).to(P.DEVICE))
                         .double().cpu().numpy())
        z = np.vstack(z); return z - _logsumexp(z)
    out["CNN"] = cnn_logp
    cols = np.array(known)
    for algo, (params, lr_dict) in meta_state.items():
        w = P.global_few_shot_adapt(algo, params, lr_dict, D["X_tr"][m_tr],
                           D["y_tr"][m_tr], D["n_classes"], cfg["meta"])

        def meta_logp(X, w=w):
            X = np.asarray(X, np.float32); z = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    z.append(w._logits(torch.as_tensor(X[i:i + 4096])
                                       .to(P.DEVICE)).double().cpu().numpy())
            z = np.vstack(z)[:, cols]; return z - _logsumexp(z)
        out[algo] = meta_logp
    return out, known, (Xk_va, yk_va)


def _logsumexp(z):
    m = z.max(1, keepdims=True)
    return m + np.log(np.exp(z - m).sum(1, keepdims=True))


def tempered_softmax(logp, T):
    z = logp / T
    return np.exp(z - _logsumexp(z))


def fit_temperature(logp_val, y_val, grid=np.exp(np.linspace(np.log(0.02),
                                                             np.log(50), 400))):
    """Temperature minimising validation NLL (1-D grid search in log T; exact,
    derivative-free and robust to saturated logits)."""
    nll = [-np.mean((logp_val / T - _logsumexp(logp_val / T))[np.arange(len(y_val)),
                                                         y_val]) for T in grid]
    return float(grid[int(np.argmin(nll))])


def expected_calibration_error(proba, y, n_bins=15):
    conf, pred = proba.max(1), proba.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def run_e3_calibration(ds_name, cfg, D=None, meta_state=None):
    D = D or prepare_dataset(ds_name, cfg)
    if meta_state is None:
        meta_state = {}
    if set(meta_state) != set(P.META_MODELS):
        for algo in P.META_MODELS:
            if algo not in meta_state:
                params, lr_dict, _, _ = P.meta_train(
                    algo, D["X_tr"], D["y_tr"], D["meta_pool"],
                    D["X_tr"].shape[1], D["n_classes"], cfg["meta"])
                meta_state[algo] = (params, lr_dict)
    logp_fns, known, (Xk_va, yk_va) = known_class_log_probabilities(meta_state, D, cfg)
    # open-set evaluation set: identical construction to P.run_open_set_detection
    rng = np.random.RandomState(P.SEED)
    known_idx = np.where(np.isin(D["y_te"], known))[0]
    cap = cfg["zeroday_detect"]["max_test_known"]
    if cap and len(known_idx) > cap:
        known_idx = rng.choice(known_idx, cap, replace=False)
    zero_idx = np.where(np.isin(D["y_te"], D["zero_day"]))[0]
    X_os = D["X_te"][np.concatenate([known_idx, zero_idx])]
    is_zero = np.r_[np.zeros(len(known_idx), int), np.ones(len(zero_idx), int)]
    remap = {c: i for i, c in enumerate(known)}
    yk_te = np.array([remap[c] for c in D["y_te"][known_idx]])
    rows = []
    for name, f in logp_fns.items():
        lv, lo = f(Xk_va), f(X_os)
        T = fit_temperature(lv, yk_va)
        p_raw, p_ts = tempered_softmax(lo, 1.0), tempered_softmax(lo, T)
        pv_raw = tempered_softmax(lv, 1.0)
        # isotonic recalibration of the MSP confidence (monotone in MSP)
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            pv_raw.max(1), (pv_raw.argmax(1) == yk_va).astype(float))
        r = {"Dataset": ds_name, "Model": name, "T": T,
             "ECE_raw": expected_calibration_error(p_raw[:len(known_idx)], yk_te),
             "ECE_TS": expected_calibration_error(p_ts[:len(known_idx)], yk_te),
             "N_known": len(known_idx), "N_zeroday": len(zero_idx)}
        for tag, p in [("raw", p_raw), ("TS", p_ts)]:
            sc = P.compute_novelty_scores(p, ["msp", "entropy", "margin"])
            for sname, s in sc.items():
                m, _ = P.open_set_detection_metrics(s, is_zero, [0.05, 0.10])
                for key in ["AUROC", "AUPR", "FPR@95TPR", "BestF1", "TPR@FPR5"]:
                    r[f"{sname}_{tag}_{key}"] = m[key]
        s_iso = 1.0 - iso.predict(p_raw.max(1))
        m, _ = P.open_set_detection_metrics(s_iso, is_zero, [0.05, 0.10])
        r["MSP_iso_AUROC"] = m["AUROC"]
        rows.append(r)
    return rows
