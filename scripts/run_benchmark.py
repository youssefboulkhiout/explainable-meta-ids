#!/usr/bin/env python
"""
Run the full benchmark pipeline (stages 1-8) on one or more datasets.

Examples
--------
    python scripts/run_benchmark.py                                  # all datasets
    python scripts/run_benchmark.py --datasets RT-IoT2022 UNSW-NB15
    python scripts/run_benchmark.py --quick                          # smoke test
    python scripts/run_benchmark.py --config configs/default.yaml --data-dir /path/to/csvs

Equivalent to `python -m meta_ids` or the `meta-ids-benchmark` command once the
package is installed (`pip install -e .`).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from meta_ids.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
