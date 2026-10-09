"""E5 (protocol check) -- episode-restricted decoding + 95% confidence
intervals over episodes."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

import meta_ids as P                        # benchmark pipeline (unchanged)
from .preparation import prepare_dataset


# =============================================================================
# E5 -- restricted episodic decoding + confidence intervals
# =============================================================================
def evaluate_episodic_with_ci(algo, params, X_te, y_te, class_pool, mcfg, restrict):
    """Same sampler, seed and adaptation as P.evaluate_episodic, but (i) returns
    per-episode scores (mean, std, 95% CI) and (ii) optionally restricts the
    arg-max of gradient-based learners to the classes present in the episode,
    which is what ProtoNet does by construction."""
    rng = np.random.RandomState(P.SEED + 7)
    pool = [c for c in class_pool if np.sum(y_te == c) >= mcfg["k_shot"] + 2]
    if len(pool) < 2:
        return {}
    n_way = min(mcfg["n_way"], len(pool))
    f1s, accs, overlap = [], [], 0
    for _ in range(mcfg["eval_episodes"]):
        sx, sy, qx, qy = P.sample_episode(X_te, y_te, np.array(pool), n_way,
                                          mcfg["k_shot"], mcfg["q_query"], rng)
        if algo == "ProtoNet":
            with torch.no_grad():
                lg, classes = P.prototype_logits(sx, sy, qx, params)
                pred = classes[lg.argmax(1)]
        else:
            keys = (P.HEAD_PARAM_KEYS if algo == "ANIL" else P.BODY_PARAM_KEYS
                    if algo == "BOIL" else list(params.keys()))
            fast = P.clone_params(params)
            opt = torch.optim.SGD([fast[k] for k in keys], lr=mcfg["inner_lr"])
            for _ in range(mcfg["inner_steps"] * 2):
                opt.zero_grad()
                F.cross_entropy(P.mlp_logits(sx, fast), sy).backward()
                opt.step()
            with torch.no_grad():
                z = P.mlp_logits(qx, fast)
                if restrict:
                    ep = torch.unique(sy)
                    pred = ep[z[:, ep].argmax(1)]
                else:
                    pred = z.argmax(1)
        yq, pr = qy.cpu().numpy(), pred.cpu().numpy()
        accs.append(accuracy_score(yq, pr))
        f1s.append(f1_score(yq, pr, average="macro", zero_division=0))
    f1s = np.array(f1s)
    half = 1.96 * f1s.std(ddof=1) / np.sqrt(len(f1s))
    return {"F1_mean": f1s.mean(), "F1_std": f1s.std(ddof=1),
            "F1_CI95_low": f1s.mean() - half, "F1_CI95_high": f1s.mean() + half,
            "Acc_mean": float(np.mean(accs)), "n_way": n_way,
            "small_class_resampled": any(np.sum(y_te == c) < mcfg["k_shot"]
                                         + mcfg["q_query"] for c in pool)}


def run_e5_episodic_ci(ds_name, cfg, D=None, meta_state=None):
    D = D or prepare_dataset(ds_name, cfg)
    if meta_state is None:
        meta_state = {}
    rows = []
    for algo in P.META_MODELS:
        if algo not in meta_state:
            params, lr_dict, _, _ = P.meta_train(
                algo, D["X_tr"], D["y_tr"], D["meta_pool"], D["X_tr"].shape[1],
                D["n_classes"], cfg["meta"])
            meta_state[algo] = (params, lr_dict)
        params, _ = meta_state[algo]
        for regime, pool in [("fewshot", list(D["meta_pool"])),
                             ("zeroday", D["zero_day"] + D["benign_ids"][:1])]:
            for restrict in ([False] if algo == "ProtoNet" else [False, True]):
                r = evaluate_episodic_with_ci(algo, params, D["X_te"], D["y_te"], pool,
                                     cfg["meta"], restrict)
                rows.append({"Dataset": ds_name, "Model": algo,
                             "Regime": regime,
                             "Decoding": "episode-restricted" if (
                                 restrict or algo == "ProtoNet") else
                             "full label space", **r})
    return rows
