"""Command-line entry point running the requested revision experiments."""
import os
import argparse

import pandas as pd

import meta_ids as P                        # benchmark pipeline (unchanged)
from . import e1_feature_sensitivity as _e1
from .preparation import OUTPUT_DIR, prepare_dataset, audit_zero_day_protocol
from .e1_feature_sensitivity import run_e1_feature_sensitivity
from .e2_adaptation_budget import run_e2_adaptation_budget, plot_e2_convergence
from .e3_calibration import run_e3_calibration
from .e4_latency import collect_hardware_info, run_e4_latency
from .e5_episodic_ci import run_e5_episodic_ci
from .latex_tables import write_latex_tables


# =============================================================================
def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(
        description="Revision experiments E1-E5 for manuscript IOT-D-26-03171")
    ap.add_argument("--exp", nargs="+", default=["all"],
                    choices=["all", "E1", "E2", "E3", "E4", "E5", "audit"])
    ap.add_argument("--datasets", nargs="+", default=list(P.DATASETS.keys()))
    ap.add_argument("--k-list", nargs="+", default=["10", "20", "30", "40"])
    ap.add_argument("--e1-models", nargs="+", default=list(P.META_MODELS))
    ap.add_argument("--lr-grid", nargs="+", type=float,
                    default=[0.005, 0.01, 0.05, 0.1])
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-sizes", nargs="+", default=["1", "256", "4096", "full"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=None,
                    help="optional YAML file overriding CONFIG / DATASETS")
    ap.add_argument("--data-dir", default=None,
                    help="folder containing the dataset CSV files")
    a = ap.parse_args()
    OUTPUT_DIR = a.output
    _e1.OUTPUT_DIR = OUTPUT_DIR          # propagate --output to E1 (now in its own module)
    cfg = P.load_config(a.config, P.CONFIG)
    if a.data_dir:
        P.set_data_dir(a.data_dir)
    if a.quick:
        cfg["max_rows"] = 30_000
        cfg["meta"].update(episodes=60, eval_episodes=20)
        cfg["cnn"].update(epochs=8)
    exps = {"E1", "E2", "E3", "E4", "E5", "audit"} if "all" in a.exp else set(a.exp)
    P.ensure_dir(OUTPUT_DIR)
    save = lambda rows, n: pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, n), index=False) if rows else None
    P.save_json(collect_hardware_info(), os.path.join(OUTPUT_DIR, "hardware.json"))
    acc = {k: [] for k in ["audit", "info", "E1", "E2t", "E2c", "E3", "E4", "E5"]}
    for ds in a.datasets:
        try:
            D, ms = None, {}
            if exps & {"audit", "E2", "E3", "E4", "E5"}:
                D = prepare_dataset(ds, cfg)
            if "audit" in exps:
                r, info = audit_zero_day_protocol(ds, D); acc["audit"] += r; acc["info"].append(info)
            if "E1" in exps:
                ks = [k if k == "all" else int(k) for k in a.k_list]
                acc["E1"] += run_e1_feature_sensitivity(ds, cfg, ks, a.e1_models)
            if "E2" in exps:
                t, c = run_e2_adaptation_budget(ds, cfg, a.lr_grid, a.max_steps, D, ms)
                acc["E2t"] += t; acc["E2c"] += c
            if "E3" in exps:
                acc["E3"] += run_e3_calibration(ds, cfg, D, ms)
            if "E4" in exps:
                acc["E4"] += run_e4_latency(ds, cfg, a.batch_sizes, a.repeats, D, ms)
            if "E5" in exps:
                acc["E5"] += run_e5_episodic_ci(ds, cfg, D, ms)
        except FileNotFoundError as e:
            P.log(str(e))
    save(acc["audit"], "protocol_audit_classes.csv")
    if acc["info"]:
        P.save_json(acc["info"], os.path.join(OUTPUT_DIR, "protocol_audit.json"))
    save(acc["E1"], "E1_feature_sensitivity.csv")
    save(acc["E2t"], "E2_trajectories.csv"); save(acc["E2c"], "E2_checkpoints.csv")
    if acc["E2t"]:
        plot_e2_convergence(pd.DataFrame(acc["E2t"]),
                os.path.join(OUTPUT_DIR, "revision_global_adapt_convergence.png"))
    save(acc["E3"], "E3_calibration.csv")
    save(acc["E4"], "E4_latency.csv")
    save(acc["E5"], "E5_episodic_ci.csv")
    write_latex_tables(OUTPUT_DIR)
    P.log(f"done -> {OUTPUT_DIR}/")
