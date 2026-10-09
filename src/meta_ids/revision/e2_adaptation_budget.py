"""E2 (Reviewer 1, C5/C7) -- convergence of the global K-shot adaptation and
ablation of its budget (steps x inner learning rate)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

import meta_ids as P                        # benchmark pipeline (unchanged)
from .preparation import GRADIENT_META_MODELS, prepare_dataset


# =============================================================================
# E2 -- global-adaptation convergence and budget ablation
# =============================================================================
def global_adaptation_trajectory(algo, params, lr_dict, D, mcfg, lr, max_steps,
                     checkpoints, use_learned_rates=False):
    """Plain-SGD global adaptation exactly as P.global_few_shot_adapt (same support set,
    same parameter subset), but logging the support loss/accuracy at every step
    and the full-test-set macro-F1 at the requested checkpoints. If
    use_learned_rates is True (Meta-SGD only), each parameter is updated with
    its meta-learned per-parameter rate instead of the scalar lr."""
    rng = np.random.RandomState(P.SEED)          # same support as global_few_shot_adapt
    sx, sy = [], []
    for c in np.unique(D["y_tr"]):
        idx = np.where(D["y_tr"] == c)[0]
        take = rng.choice(idx, size=min(len(idx), mcfg["adapt_shot"]),
                          replace=False)
        sx.append(D["X_tr"][take]); sy.append(np.full(len(take), c))
    sx = torch.as_tensor(np.concatenate(sx), dtype=torch.float32, device=P.DEVICE)
    sy = torch.as_tensor(np.concatenate(sy), dtype=torch.long, device=P.DEVICE)
    keys = (P.HEAD_PARAM_KEYS if algo == "ANIL" else P.BODY_PARAM_KEYS if algo == "BOIL"
            else list(params.keys()))
    fast = P.clone_params(params)
    Xte = torch.as_tensor(D["X_te"], dtype=torch.float32, device=P.DEVICE)
    log_rows, ck_rows = [], []

    def test_f1():
        with torch.no_grad():
            pr = torch.cat([P.mlp_logits(Xte[i:i + 4096], fast).argmax(1)
                            for i in range(0, len(Xte), 4096)]).cpu().numpy()
        return (f1_score(D["y_te"], pr, average="macro", zero_division=0),
                accuracy_score(D["y_te"], pr))

    for step in range(max_steps + 1):
        logits = P.mlp_logits(sx, fast)
        loss = F.cross_entropy(logits, sy)
        acc = float((logits.argmax(1) == sy).float().mean())
        log_rows.append({"step": step, "support_loss": float(loss),
                         "support_acc": acc})
        if step in checkpoints:
            f1, a = test_f1()
            ck_rows.append({"step": step, "test_F1": f1, "test_acc": a})
        if step == max_steps:
            break
        grads = torch.autograd.grad(loss, [fast[k] for k in keys])
        with torch.no_grad():
            for k, g in zip(keys, grads):
                rate = lr_dict[k] if (use_learned_rates and lr_dict) else lr
                fast[k] -= rate * g
    return log_rows, ck_rows


def run_e2_adaptation_budget(ds_name, cfg, lr_grid, max_steps, D=None, meta_state=None):
    D = D or prepare_dataset(ds_name, cfg)
    mcfg = cfg["meta"]
    checkpoints = sorted({0, 10, 25, 50, 100, 150, 200, max_steps}
                         & set(range(max_steps + 1)))
    traj, ck = [], []
    for algo in GRADIENT_META_MODELS:
        if meta_state and algo in meta_state:
            params, lr_dict = meta_state[algo]
        else:
            P.log(f"[E2] {ds_name} meta-training {algo}")
            params, lr_dict, _, _ = P.meta_train(
                algo, D["X_tr"], D["y_tr"], D["meta_pool"], D["X_tr"].shape[1],
                D["n_classes"], mcfg)
            if meta_state is not None:
                meta_state[algo] = (params, lr_dict)
        variants = [(lr, False) for lr in lr_grid]
        if algo == "Meta-SGD":
            variants.append(("learned", True))
        for lr, learned in variants:
            lg, cp = global_adaptation_trajectory(algo, params, lr_dict, D, mcfg,
                                      None if learned else lr, max_steps,
                                      checkpoints, use_learned_rates=learned)
            for r in lg:
                traj.append({"Dataset": ds_name, "Model": algo, "lr": lr, **r})
            for r in cp:
                ck.append({"Dataset": ds_name, "Model": algo, "lr": lr, **r})
    return traj, ck


def plot_e2_convergence(traj_df, path, lr=0.01):
    import matplotlib.pyplot as plt
    dss = list(dict.fromkeys(traj_df["Dataset"]))
    fig, axes = plt.subplots(1, len(dss), figsize=(4.6 * len(dss), 3.6),
                             squeeze=False)
    for ax, ds in zip(axes[0], dss):
        sub = traj_df[(traj_df["Dataset"] == ds) & (traj_df["lr"] == lr)]
        for m in GRADIENT_META_MODELS:
            s = sub[sub["Model"] == m]
            ax.plot(s["step"], s["support_loss"], lw=1.3, label=m)
        ax.axvline(50, color="grey", ls="--", lw=0.8)
        ax.set_title(ds); ax.set_xlabel("Global-adaptation step")
        ax.set_ylabel("Support cross-entropy"); ax.set_yscale("log")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); P.save_figure(fig, path)
