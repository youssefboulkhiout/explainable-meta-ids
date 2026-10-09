#!/usr/bin/env python
"""
Run the reviewer-requested experiments E1-E5 (+ zero-day protocol audit).

Examples
--------
    python scripts/run_revision.py --exp all
    python scripts/run_revision.py --exp E1 E3 --datasets RT-IoT2022
    python scripts/run_revision.py --exp E1 --k-list 10 20 30 40

Equivalent to `python -m meta_ids.revision` or the `meta-ids-revision` command
once the package is installed (`pip install -e .`).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from meta_ids.revision.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
