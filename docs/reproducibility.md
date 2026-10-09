# Reproducibility

## Seeds

`SEED = 42` (`src/meta_ids/config.py`) seeds Python, NumPy and PyTorch (CPU and
CUDA) before each dataset, each CNN fit and each meta-training run. Splits,
SMOTE, undersampling, LightGBM, episode samplers (`SEED` for meta-training,
`SEED + 7` for episodic evaluation) and XAI sampling all use it.

## Determinism limits

- Results are reproducible on the same hardware and library versions.
- On GPU, some PyTorch kernels are non-deterministic; small differences between
  CPU and GPU runs, or between GPU models, are expected.
- LightGBM uses all CPU cores (`n_jobs=-1`); its results do not depend on the
  number of threads in practice, but this is not guaranteed across versions.
- Timing columns (`*_time_s`, `Eval_ms_per_1k`, E4) depend on the machine and
  load, and differ from run to run.

## Recording the environment

`requirements.txt` gives minimum versions. Record the exact versions used for
the published results:

```bash
pip freeze > requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`scripts/run_revision.py` also writes the hardware and library versions to
`outputs_revision/hardware.json`.

## Equivalence with the original scripts

Version 1.0.0 only moved code into modules and renamed files and identifiers
([api_changes.md](api_changes.md)). This was checked by:

1. **Static comparison** — every function, class and constant of
   `xai_meta_5.py` and `revision_experiments.py` has the same syntax tree as the
   original once identifiers are renamed back.
2. **Run comparison** — the original scripts and this repository were run with
   the same `--quick` settings on the same synthetic dataset
   (`scripts/make_synthetic_dataset.py`): every CSV of the benchmark and of E1-E5
   and every LaTeX table body were identical, apart from timing columns.
3. `tests/test_config.py` — `configs/default.yaml` equals the in-code `CONFIG`.
