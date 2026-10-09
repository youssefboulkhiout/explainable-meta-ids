"""E4 (Reviewer 1, C8) -- controlled inference-latency protocol."""
import os
import time
import platform

import numpy as np
import torch

import meta_ids as P                        # benchmark pipeline (unchanged)
from .preparation import prepare_dataset


# =============================================================================
# E4 -- controlled inference-latency protocol
# =============================================================================
def collect_hardware_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    try:
        import lightgbm; info["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    try:                                        # human-readable CPU model
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import psutil; info["ram_GB"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return info


def run_e4_latency(ds_name, cfg, batch_sizes, repeats, D=None, meta_state=None):
    D = D or prepare_dataset(ds_name, cfg)
    X = D["X_te"].astype(np.float32)
    lgbm, _ = P.train_lightgbm(D["X_tr"], D["y_tr"], D["X_va"], D["y_va"])
    cnn, _, _ = P.train_cnn(D["X_tr"], D["y_tr"], D["X_va"], D["y_va"],
                            D["n_classes"], cfg["cnn"])
    models = {"LightGBM": ("lgbm", lgbm), "CNN": ("torch", cnn)}
    if meta_state is None:
        meta_state = {}
    for algo in P.META_MODELS:
        if algo not in meta_state:
            params, lr_dict, _, _ = P.meta_train(
                algo, D["X_tr"], D["y_tr"], D["meta_pool"], X.shape[1],
                D["n_classes"], cfg["meta"])
            meta_state[algo] = (params, lr_dict)
        params, lr_dict = meta_state[algo]
        models[algo] = ("meta", P.global_few_shot_adapt(algo, params, lr_dict, D["X_tr"],
                                               D["y_tr"], D["n_classes"],
                                               cfg["meta"]))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for name, (kind, m) in models.items():
        for dev in (["cpu"] if kind == "lgbm" else devices):
            if kind == "torch":
                m.to(dev); fwd = lambda xb, m=m: torch.softmax(m(xb), 1)
            elif kind == "meta":
                p_dev = {k: v.to(dev) for k, v in m.params.items()}
                prot = m.protos.to(dev) if m.protos is not None else None
                def fwd(xb, p=p_dev, pr=prot, algo=m.algo, pc=m.proto_classes,
                        C=len(m.classes_)):
                    if algo == "ProtoNet":
                        d = torch.cdist(P.mlp_embed(xb, p), pr) ** 2
                        z = torch.full((xb.shape[0], C), -1e9, device=xb.device)
                        z[:, pc] = -d; return torch.softmax(z, 1)
                    return torch.softmax(P.mlp_logits(xb, p), 1)
            for bs in batch_sizes:
                bs_eff = len(X) if bs == "full" else int(bs)
                n = min(len(X), max(bs_eff, 4096))   # time >= 4096 flows
                Xs = X[:n]
                def run_once():
                    if kind == "lgbm":
                        for i in range(0, n, bs_eff):
                            m.predict_proba(Xs[i:i + bs_eff])
                    else:
                        with torch.no_grad():
                            for i in range(0, n, bs_eff):
                                fwd(torch.as_tensor(Xs[i:i + bs_eff], device=dev))
                        if dev == "cuda":
                            torch.cuda.synchronize()
                run_once()                                    # warm-up
                ts = []
                for _ in range(repeats):
                    t0 = time.perf_counter(); run_once()
                    ts.append(time.perf_counter() - t0)
                ms_per_1k = 1e3 * np.array(ts) / n * 1e3
                rows.append({"Dataset": ds_name, "Model": name, "device": dev,
                             "batch_size": bs, "n_flows_timed": n,
                             "median_ms_per_1k": float(np.median(ms_per_1k)),
                             "iqr_ms_per_1k": float(np.subtract(
                                 *np.percentile(ms_per_1k, [75, 25])))})
            if kind == "torch":
                m.to(P.DEVICE)
    return rows
