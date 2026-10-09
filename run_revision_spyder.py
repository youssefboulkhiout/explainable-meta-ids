# -*- coding: utf-8 -*-
"""
Run the revision experiments (E1-E5 + protocol audit) from Spyder (or any
IDE): edit the SETTINGS block, then press F5 (Run file).

It is equivalent to `python scripts/run_revision.py <options>`; the settings
below are turned into exactly those command-line options, so results are the
same as from the terminal. After the run, the CSV files of the experiments that
were run are loaded into the Variable Explorer (audit, e1, e2_trajectories,
e2_checkpoints, e3, e4, e5).
"""
import os
import sys

# =============================================================================
# SETTINGS -- edit here
# =============================================================================
EXPERIMENTS = ["all"]    # "all" or any of "audit", "E1", "E2", "E3", "E4", "E5"
DATASETS = ["RT-IoT2022", "UNSW-NB15", "ASEADOS_SDN_IoT"]   # any subset
QUICK = False            # True = fast settings (NOT for reported results)
OUTPUT_DIR = "outputs_revision"

# E1 -- number of ANOVA features and meta-learners to re-train for each k
K_LIST = ["10", "20", "30", "40"]          # "all" = every feature
E1_MODELS = ["Reptile", "MAML", "FOMAML", "Meta-SGD", "ANIL", "BOIL", "ProtoNet"]
# E2 -- global-adaptation budget
LR_GRID = [0.005, 0.01, 0.05, 0.1]
MAX_STEPS = 200
# E4 -- latency protocol
BATCH_SIZES = ["1", "256", "4096", "full"]
REPEATS = 10

CONFIG_FILE = None       # e.g. "configs/default.yaml"; None = in-code reference CONFIG
DATA_DIR = None          # folder with the dataset CSV files; None = "data/"
# =============================================================================


def _find_repo_root():
    """Return the repository root (the folder containing src/meta_ids), looking
    upwards from this file's folder, then from the current working directory.
    This file can therefore live in the repository root or in scripts/."""
    starts = []
    if "__file__" in globals():
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    starts.append(os.getcwd())
    for start in starts:
        folder = start
        while True:
            if os.path.isfile(os.path.join(folder, "src", "meta_ids", "__init__.py")):
                return folder
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
    raise FileNotFoundError(
        "Cannot find the folder 'src/meta_ids'. Keep this file inside the "
        "explainable-meta-ids repository (root folder or scripts/).")


REPO_ROOT = _find_repo_root()
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(REPO_ROOT)          # relative paths (data/, outputs, configs/) resolve here


def build_argv():
    """Translate the SETTINGS block into the revision command-line options."""
    argv = ["run_revision",
            "--exp", *EXPERIMENTS,
            "--datasets", *DATASETS,
            "--k-list", *[str(k) for k in K_LIST],
            "--e1-models", *E1_MODELS,
            "--lr-grid", *[str(lr) for lr in LR_GRID],
            "--max-steps", str(MAX_STEPS),
            "--batch-sizes", *[str(b) for b in BATCH_SIZES],
            "--repeats", str(REPEATS),
            "--output", OUTPUT_DIR]
    if QUICK:
        argv.append("--quick")
    if CONFIG_FILE is not None:
        argv += ["--config", CONFIG_FILE]
    if DATA_DIR is not None:
        argv += ["--data-dir", DATA_DIR]
    return argv


def load_results(out_dir):
    """Read the CSV files written by the run (missing ones -> None)."""
    import pandas as pd
    names = {"audit": "protocol_audit_classes.csv",
             "e1": "E1_feature_sensitivity.csv",
             "e2_trajectories": "E2_trajectories.csv",
             "e2_checkpoints": "E2_checkpoints.csv",
             "e3": "E3_calibration.csv",
             "e4": "E4_latency.csv",
             "e5": "E5_episodic_ci.csv"}
    return {k: (pd.read_csv(os.path.join(out_dir, f))
                if os.path.exists(os.path.join(out_dir, f)) else None)
            for k, f in names.items()}


if __name__ == "__main__":
    from meta_ids.revision.cli import main     # NOT 'src.meta_ids': src/ is added to sys.path above

    saved_argv = sys.argv
    sys.argv = build_argv()          # Spyder's own sys.argv would confuse argparse
    print("Running:", " ".join(sys.argv))
    try:
        main()
    finally:
        sys.argv = saved_argv

    results = load_results(os.path.abspath(OUTPUT_DIR))
    audit = results["audit"]
    e1 = results["e1"]
    e2_trajectories = results["e2_trajectories"]
    e2_checkpoints = results["e2_checkpoints"]
    e3 = results["e3"]
    e4 = results["e4"]
    e5 = results["e5"]
    print(f"\nResults in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"LaTeX table bodies in: {os.path.join(os.path.abspath(OUTPUT_DIR), 'tables')}")
