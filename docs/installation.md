# Installation

## Requirements

- Python 3.10 or newer (tested with 3.13).
- CPU is enough for every experiment; a CUDA GPU speeds up the CNN and the
  meta-learners. LightGBM, SHAP and LIME run on CPU.
- Disk: a few GB for the datasets and the generated figures.

## pip + virtual environment (recommended)

```bash
git clone https://github.com/<user>/explainable-meta-ids.git
cd explainable-meta-ids
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .                     # installs the package and its dependencies
```

`pip install -r requirements.txt` installs the same dependencies without
installing the package; the scripts in `scripts/` work either way.

### PyTorch with a GPU

`pip install -e .` installs the default PyTorch build. For a specific CUDA
version, install PyTorch first with the command given at
<https://pytorch.org/get-started/locally/>, then run `pip install -e .`.
For a CPU-only build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The device is selected automatically (`cuda` if available, otherwise `cpu`).

## conda

```bash
conda env create -f environment.yml
conda activate meta-ids
```

## Development tools

```bash
pip install -e ".[dev]"      # pytest, pyflakes
pip install -e ".[latency]"  # psutil, adds RAM size to the E4 hardware report
```

## Check the installation

```bash
python scripts/make_synthetic_dataset.py --out data/RT-IoT2022.csv
python scripts/run_benchmark.py --quick --datasets RT-IoT2022 --output outputs_smoke
pytest -m "not slow"
```

The smoke run takes a few minutes on a laptop CPU and writes results to
`outputs_smoke/`. Delete the synthetic `data/RT-IoT2022.csv` before using the
real dataset.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: [RT-IoT2022] file not found` | Put the CSV in `data/` with the exact name, or use `--data-dir`. The run skips missing datasets and continues. |
| `lime` fails to build | `pip install --no-build-isolation lime` |
| `ModuleNotFoundError: yaml` | `pip install pyyaml` (only needed with `--config`) |
| SHAP is very slow | Expected for KernelExplainer on the CNN and meta-learners; reduce `xai.n_explain` / `xai.kernel_nsamples` for exploratory runs. |
| CUDA out of memory | Inference is already batched by 4096; set `CUDA_VISIBLE_DEVICES=""` to run on CPU. |
