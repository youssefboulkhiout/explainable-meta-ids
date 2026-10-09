"""End-to-end smoke test on a tiny synthetic dataset (marked `slow`).

    pytest -m slow
"""
import os
import subprocess
import sys

import pandas as pd
import pytest

from conftest import ROOT


@pytest.mark.slow
def test_benchmark_runs_end_to_end(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_synthetic_dataset.py"),
                    "--out", str(data / "RT-IoT2022.csv"), "--scale", "0.5"], check=True)
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_benchmark.py"),
                    "--config", os.path.join(ROOT, "configs", "smoke_test.yaml"),
                    "--datasets", "RT-IoT2022", "--data-dir", str(data),
                    "--output", str(out)], check=True, cwd=tmp_path)
    metrics = pd.read_csv(out / "all_model_metrics.csv")
    assert len(metrics) == 9                                  # 2 base + 7 meta
    assert (out / "all_zeroday_detection.csv").exists()
