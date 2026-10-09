# Data folder

The datasets are **not** included in this repository (size and licence).
Download each one from its original source and save it here as a single CSV
file with the exact name below (or pass another folder with `--data-dir`).

```
data/
├── RT-IoT2022.csv
├── UNSW-NB15.csv
└── ASEADOS_SDN_IoT.csv
```

| File | Download | Licence |
|---|---|---|
| `RT-IoT2022.csv` | <https://archive.ics.uci.edu/dataset/942/rt-iot2022> | CC BY 4.0 |
| `UNSW-NB15.csv` | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> | see the dataset page |
| `ASEADOS_SDN_IoT.csv` | <https://aseados.ucd.ie/datasets/SDN-IoT/> | CC BY 4.0 (research use, see the dataset page) |

Only one CSV per dataset is read. If a dataset is distributed as several CSV
files, concatenate them into one file first (same header). See
[../docs/datasets.md](../docs/datasets.md) for the exact files used, the label
columns and the preprocessing applied by the pipeline.

To check the installation without downloading anything, generate a synthetic
file: `python scripts/make_synthetic_dataset.py --out data/RT-IoT2022.csv`
(overwrite it with the real file before running real experiments).
