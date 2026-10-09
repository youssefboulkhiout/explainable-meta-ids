# Changelog

## 1.0.0 — repository release

- Code reorganised from `xai_meta_5.py` and `revision_experiments.py` into the
  `meta_ids` package (`src/` layout), one module per pipeline stage and per
  revision experiment.
- Files and identifiers renamed for readability
  ([docs/api_changes.md](docs/api_changes.md)). No change to computations,
  hyper-parameters, seeds, output file names or CSV columns.
- Added `--config` (YAML overrides) and `--data-dir` options; without them the
  reference configuration is used unchanged.
- Added packaging (`pyproject.toml`, `requirements*.txt`, `environment.yml`),
  configuration files, documentation, tests, a synthetic-data generator and CI.
