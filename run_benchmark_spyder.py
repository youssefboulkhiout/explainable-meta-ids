# -*- coding: utf-8 -*-
"""
Run the benchmark pipeline from Spyder (or any IDE): edit the SETTINGS block,
then press F5 (Run file).

It is equivalent to `python scripts/run_benchmark.py <options>`; the settings
below are turned into exactly those command-line options, so results are the
same as from the terminal. After the run, the result tables are loaded into the
Variable Explorer (metrics, xai_metrics, fewshot_zeroday, zeroday_classes,
zeroday_detection).
"""
import os
import sys

# =============================================================================
# SETTINGS -- edit here
# =============================================================================
DATASETS = ["RT-IoT2022", "UNSW-NB15", "ASEADOS_SDN_IoT"]   # any subset
QUICK = False            # True = fast smoke-test settings (NOT for reported results)
EPISODES = None          # e.g. 1000 to override meta-training episodes; None = default (500)
MAX_ROWS = None          # e.g. 100000 to override the stratified row cap; None = default
OUTPUT_DIR = "outputs"   # relative to the repository root, or an absolute path
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
    """Translate the SETTINGS block into the benchmark's command-line options."""
    argv = ["run_benchmark", "--datasets", *DATASETS, "--output", OUTPUT_DIR]
    if QUICK:
        argv.append("--quick")
    if EPISODES is not None:
        argv += ["--episodes", str(EPISODES)]
    if MAX_ROWS is not None:
        argv += ["--max-rows", str(MAX_ROWS)]
    if CONFIG_FILE is not None:
        argv += ["--config", CONFIG_FILE]
    if DATA_DIR is not None:
        argv += ["--data-dir", DATA_DIR]
    return argv


def load_results(out_dir):
    """Read the aggregated CSV files written by the run (missing ones -> None)."""
    import pandas as pd
    names = {"metrics": "all_model_metrics.csv",
             "xai_metrics": "all_xai_metrics.csv",
             "fewshot_zeroday": "all_fewshot_zeroday.csv",
             "zeroday_classes": "all_zeroday_class_report.csv",
             "zeroday_detection": "all_zeroday_detection.csv"}
    return {k: (pd.read_csv(os.path.join(out_dir, f))
                if os.path.exists(os.path.join(out_dir, f)) else None)
            for k, f in names.items()}


if __name__ == "__main__":
    from meta_ids.pipeline import main     # NOT 'src.meta_ids': src/ is added to sys.path above

    saved_argv = sys.argv
    sys.argv = build_argv()          # Spyder's own sys.argv would confuse argparse
    print("Running:", " ".join(sys.argv))
    try:
        main()
    finally:
        sys.argv = saved_argv

    results = load_results(os.path.abspath(OUTPUT_DIR))
    metrics = results["metrics"]
    xai_metrics = results["xai_metrics"]
    fewshot_zeroday = results["fewshot_zeroday"]
    zeroday_classes = results["zeroday_classes"]
    zeroday_detection = results["zeroday_detection"]
    print(f"\nResults in: {os.path.abspath(OUTPUT_DIR)}")
    if metrics is not None:
        print(metrics[["Dataset", "Model", "Accuracy", "F1", "AUC", "Kappa"]]
              .to_string(index=False))
