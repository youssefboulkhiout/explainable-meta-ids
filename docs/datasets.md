# Datasets

Three public IoT / network intrusion-detection datasets are used. None is
redistributed here; download them from the sources below and store one CSV per
dataset under `data/` (file names in [data/README.md](../data/README.md)).

The registry of datasets (path, label-column candidates, columns to drop) is
`DATASETS` in `src/meta_ids/config.py`, mirrored in `configs/default.yaml`.

| Key (`--datasets`) | File | Label candidates (first match wins) | Columns dropped |
|---|---|---|---|
| `RT-IoT2022` | `data/RT-IoT2022.csv` | `attack_type`, `label`, `class` | `no.`, `no`, `id`, `unnamed: 0` |
| `UNSW-NB15` | `data/UNSW-NB15.csv` | `attack_cat` | `id`, `label` |
| `ASEADOS_SDN_IoT` | `data/ASEADOS_SDN_IoT.csv` | `label`, `class`, `attack_type`, `category` | `id`, `flow_id` |

Column matching is case-insensitive. Identifier-like columns (`id`, `index`,
`flow id`, …) are always dropped (`GENERIC_DROP_COLUMNS`).

## RT-IoT2022

- **Source:** UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/942/rt-iot2022> — doi:[10.24432/C5P338](https://doi.org/10.24432/C5P338). Licence CC BY 4.0.
- **Content:** 123,117 flows, 83 features, label column `Attack_type`. Normal traffic is split into device-specific classes (MQTT, Thing_speak, Wipro_bulb_Dataset, Amazon-Alexa); attacks include DOS_SYN_Hping, ARP_poisioning, NMAP scans, DDOS_Slowloris, Metasploit_Brute_Force_SSH.
- **Cite:** B. S. Sharmila and R. Nagapadma, "Quantized autoencoder (QAE) intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset," *Cybersecurity*, 2023; and the dataset DOI above.
- **Note on the zero-day protocol:** the pipeline recognises benign classes by name (`benign` or `normal` in the class name). No RT-IoT2022 class name contains these words, so no benign class is added to its zero-day episodes (they contain only the held-out attack families). `scripts/run_revision.py --exp audit` records this in `protocol_audit.json`.

## UNSW-NB15

- **Source:** UNSW Canberra Cyber, <https://research.unsw.edu.au/projects/unsw-nb15-dataset>.
- **Content:** label column `attack_cat` (Normal + 9 attack categories). The binary `label` column is dropped because it encodes the target.
- **Expected file:** a single CSV with a header row containing `attack_cat`. The column list in `DATASETS` (`id`, `label`) corresponds to the partitioned training/testing CSV files, which have a header; the four raw CSV files have none.
- **Cite:** N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," *Military Communications and Information Systems Conference (MilCIS)*, 2015.

> **Author to complete:** state here exactly which UNSW-NB15 files were
> concatenated into `UNSW-NB15.csv` for the manuscript, so that readers can
> rebuild the same file.

## ASEADOS-SDN-IoT

- **Source:** <https://aseados.ucd.ie/datasets/SDN-IoT/> (University College Dublin). Licence CC BY 4.0; the dataset page also restricts use to non-commercial research.
- **Content:** 457,044 flows, 83 features; classes Benign, DoS, DDoS, Bot(net), Probe. Distributed as PCAPs plus one merged, labelled CSV — use that CSV.
- **Cite:** T. L. Yasarathna and N.-A. Le-Khac, "ASEADOS-SDN-IoT: A Novel SDN-IoT Network Intrusion Detection Dataset," *Internet of Things*, 2026, doi:[10.1016/j.iot.2026.101891](https://doi.org/10.1016/j.iot.2026.101891).

## What the pipeline does to every dataset

Applied identically to all three (code: `src/meta_ids/data.py`):

1. Strip column names, detect the label column, drop identifier columns.
2. Encode object columns with ≤ 50 distinct values; drop those with more (identifier-like).
3. Remove rows with NaN/±inf, duplicate rows and constant columns.
4. Drop classes with fewer than `min_class_presplit` (10) rows.
5. Stratified cap to `max_rows` (200,000) rows, then re-drop classes made too rare.
6. Stratified 70 / 10 / 20 train / validation / test split; `StandardScaler` fitted on train.
7. `SelectKBest(f_classif, k=30)` fitted on train.
8. Train-only balancing: drop classes < 50 rows, SMOTE classes up to 10,000, undersample classes above 50,000. Validation and test keep only the surviving classes.

## Using another dataset

Add an entry to the `datasets:` section of a copy of `configs/default.yaml`:

```yaml
datasets:
  MyDataset:
    path: data/MyDataset.csv
    label_candidates: [label]
    drop: [flow_id, timestamp]
```

then run `python scripts/run_benchmark.py --config my_config.yaml --datasets MyDataset`.
