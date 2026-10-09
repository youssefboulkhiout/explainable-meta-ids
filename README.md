# Explainable Meta-Learning for Few-Shot and Zero-Day Intrusion Detection in IoT Networks

Code accompanying the manuscript **IOT-D-26-03171**, *"Explainable Meta-Learning
for Few-Shot and Zero-Day Intrusion Detection in IoT Networks"*.

The repository provides a single, reproducible benchmark that compares two
supervised base learners and seven meta-learners on three IoT intrusion-detection
datasets, in three regimes:

| Regime | Question | Where |
|---|---|---|
| Closed-set classification | How well is every known attack family recognised? | stage 6 |
| Few-shot / zero-day episodic | How well does a model adapt from *K* labelled examples of a new family? | stage 5B |
| Open-set zero-day detection | Can a model flag traffic from a family it **never** saw, with no labelled example? | stage 6C |

Every model is then explained with **SHAP** and **LIME**, and the explanations are
scored with eight quality metrics.

| Component | Content |
|---|---|
| Datasets | RT-IoT2022, UNSW-NB15, ASEADOS-SDN-IoT |
| Base learners | LightGBM, 1-D CNN |
| Meta-learners | Reptile, MAML, FOMAML, Meta-SGD, ANIL, BOIL, ProtoNet |
| Novelty scores | maximum softmax probability (MSP), normalised entropy, top-2 margin |
| Open-set metrics | AUROC, AUPR, FPR@95%TPR, TPR at fixed FPR, best F1 |
| XAI | SHAP, LIME — fidelity, infidelity, sparsity, stability, comprehensiveness, sufficiency, feature agreement, rank correlation |
| Revision experiments | E1 feature-count sensitivity · E2 adaptation budget · E3 calibration · E4 latency · E5 episodic CIs |

## Repository structure

```
explainable-meta-ids/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml            # package metadata + `meta-ids-*` commands
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # + pytest, pyflakes
├── environment.yml           # conda environment
├── configs/
│   ├── default.yaml          # reference configuration of the manuscript
│   └── smoke_test.yaml       # small configuration for installation checks
├── data/
│   └── README.md             # where to download each dataset and how to name it
├── docs/                     # detailed documentation (see below)
├── scripts/
│   ├── run_benchmark.py      # stages 1-8 on one or more datasets
│   ├── run_revision.py       # reviewer experiments E1-E5
│   └── make_synthetic_dataset.py
├── src/meta_ids/
│   ├── config.py             # CONFIG, DATASETS, seed, device, YAML overrides
│   ├── utils.py
│   ├── data.py               # stages 1-4: cleaning, split, SelectKBest, balancing
│   ├── models/
│   │   ├── baselines.py      # stage 5A: LightGBM, 1-D CNN
│   │   └── meta_learning.py  # stage 5B: 7 meta-learners, adaptation, episodic eval
│   ├── evaluation/
│   │   ├── classification.py # stages 6/6B: metrics, confusion/ROC, zero-day stats
│   │   └── open_set.py       # stage 6C: open-set zero-day detection
│   ├── explainability.py     # stage 7: SHAP + LIME + 8 quality metrics
│   ├── reporting.py          # stage 8: summary and cross-dataset figures
│   ├── pipeline.py           # run_benchmark_on_dataset(), main()
│   └── revision/             # E1-E5, protocol audit, LaTeX table bodies
├── tests/
└── .github/workflows/ci.yml
```

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/<user>/explainable-meta-ids.git
cd explainable-meta-ids
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                       # or: pip install -r requirements.txt
```

For a GPU, install the PyTorch build matching your CUDA version first
(<https://pytorch.org/get-started/locally/>), then `pip install -e .`.
Conda users can run `conda env create -f environment.yml`.
Details: [docs/installation.md](docs/installation.md).

## Datasets

The datasets are not redistributed. Download them from their original sources
and place one CSV per dataset in `data/`:

| Dataset | Expected file | Label column | Source |
|---|---|---|---|
| RT-IoT2022 | `data/RT-IoT2022.csv` | `Attack_type` | UCI ML Repository, doi:10.24432/C5P338 |
| UNSW-NB15 | `data/UNSW-NB15.csv` | `attack_cat` | UNSW Canberra Cyber |
| ASEADOS-SDN-IoT | `data/ASEADOS_SDN_IoT.csv` | auto-detected (`label`, `class`, …) | aseados.ucd.ie |

Download links, citations, licences and preprocessing notes:
[data/README.md](data/README.md) and [docs/datasets.md](docs/datasets.md).

## Usage

```bash
# Check the installation in a few minutes on synthetic data
python scripts/make_synthetic_dataset.py --out data/RT-IoT2022.csv
python scripts/run_benchmark.py --quick --datasets RT-IoT2022 --output outputs_smoke

# Full benchmark (manuscript configuration), all datasets
python scripts/run_benchmark.py

# Selected datasets, CSV files stored elsewhere
python scripts/run_benchmark.py --datasets RT-IoT2022 UNSW-NB15 --data-dir /path/to/csvs

# Reviewer experiments
python scripts/run_revision.py --exp all
python scripts/run_revision.py --exp E1 E3 --datasets RT-IoT2022
```

After `pip install -e .` the same entry points are available as
`meta-ids-benchmark` / `meta-ids-revision` (or `python -m meta_ids` /
`python -m meta_ids.revision`). All options:
[docs/usage.md](docs/usage.md). Configuration keys:
[docs/configuration.md](docs/configuration.md).

## Outputs

`outputs/` (benchmark) and `outputs_revision/` (E1-E5) contain per-dataset CSV
metrics, figures, SHAP/LIME attributions, cross-dataset comparison figures and
LaTeX table bodies. File-by-file description: [docs/outputs.md](docs/outputs.md).

## Documentation

| Document | Content |
|---|---|
| [installation.md](docs/installation.md) | environments, GPU, troubleshooting |
| [datasets.md](docs/datasets.md) | sources, licences, expected format, preprocessing |
| [usage.md](docs/usage.md) | command-line options, Python API |
| [configuration.md](docs/configuration.md) | every key of `CONFIG` / `configs/default.yaml` |
| [methodology.md](docs/methodology.md) | pipeline stages, zero-day protocol, metrics |
| [revision_experiments.md](docs/revision_experiments.md) | E1-E5 and the zero-day protocol audit |
| [outputs.md](docs/outputs.md) | generated files |
| [reproducibility.md](docs/reproducibility.md) | seeds, determinism, recording versions |
| [api_changes.md](docs/api_changes.md) | old → new file and function names |

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"     # fast checks (configuration, API)
pytest -m slow           # end-to-end run on synthetic data
```

## Citation

If you use this code, please cite the article (see [CITATION.cff](CITATION.cff))
and the datasets you use (see [docs/datasets.md](docs/datasets.md)).

## Licence

Code released under the MIT licence ([LICENSE](LICENSE)). The datasets keep
their own licences.
