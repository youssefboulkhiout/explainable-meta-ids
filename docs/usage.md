# Usage

Run every command from the repository root (dataset paths and output folders
are relative to the current directory).

## Benchmark pipeline

```bash
python scripts/run_benchmark.py [options]
# same as: meta-ids-benchmark [options]   /   python -m meta_ids [options]
```

| Option | Default | Effect |
|---|---|---|
| `--datasets NAME [NAME ...]` | all three | `RT-IoT2022`, `UNSW-NB15`, `ASEADOS_SDN_IoT` |
| `--episodes N` | 500 | meta-training episodes |
| `--max-rows N` | 200000 | stratified cap on raw rows |
| `--output DIR` | `outputs` | output folder |
| `--quick` | off | fast smoke-test settings (not for reported results) |
| `--config FILE.yaml` | none | override `CONFIG` / `DATASETS` from a YAML file |
| `--data-dir DIR` | `data` | folder holding the dataset CSV files |

Order of precedence: reference `CONFIG` → `--config` file → `--episodes`,
`--max-rows`, `--output` → `--quick`.

Examples:

```bash
python scripts/run_benchmark.py                                   # manuscript run
python scripts/run_benchmark.py --datasets UNSW-NB15 --output outputs_unsw
python scripts/run_benchmark.py --config configs/smoke_test.yaml --datasets RT-IoT2022
```

A dataset whose file is missing is skipped with a message; an error in one
dataset does not stop the others.

## Revision experiments (E1-E5)

```bash
python scripts/run_revision.py [options]
# same as: meta-ids-revision [options]   /   python -m meta_ids.revision [options]
```

| Option | Default | Effect |
|---|---|---|
| `--exp E [E ...]` | `all` | any of `E1 E2 E3 E4 E5 audit`, or `all` |
| `--datasets NAME [...]` | all three | datasets to process |
| `--k-list K [K ...]` | `10 20 30 40` | E1: numbers of ANOVA features (`all` = every feature) |
| `--e1-models M [...]` | all 7 meta-learners | E1: meta-learners to re-train for each k |
| `--lr-grid LR [...]` | `0.005 0.01 0.05 0.1` | E2: global-adaptation learning rates |
| `--max-steps N` | 200 | E2: maximum global-adaptation steps |
| `--batch-sizes B [...]` | `1 256 4096 full` | E4: inference batch sizes |
| `--repeats N` | 10 | E4: timed repetitions after one warm-up |
| `--output DIR` | `outputs_revision` | output folder |
| `--quick` | off | fast settings (not for reported results) |
| `--config FILE.yaml` | none | override `CONFIG` / `DATASETS` |
| `--data-dir DIR` | `data` | folder holding the dataset CSV files |

Note that the revision `--quick` keeps `adapt_steps = 50` (only the benchmark
`--quick` lowers it to 30). What each experiment measures:
[revision_experiments.md](revision_experiments.md).

## Python API

```python
import meta_ids as P

cfg = P.load_config()                       # copy of the reference CONFIG
X_df, y, le = P.load_and_clean_dataset("RT-IoT2022", P.DATASETS["RT-IoT2022"], cfg)
(X_tr, y_tr), (X_va, y_va), (X_te, y_te), scaler = P.split_and_standardize(X_df, y, cfg)

# full pipeline for one dataset (returns the five row lists written to CSV)
perf, xai, fewshot, zd_classes, zd_openset = P.run_benchmark_on_dataset(
    "RT-IoT2022", P.DATASETS["RT-IoT2022"], cfg)
```

Every public function is re-exported at package level; the module of each one
is listed in [api_changes.md](api_changes.md).
