# Configuration

The reference configuration is `CONFIG` in `src/meta_ids/config.py`, mirrored
key for key in `configs/default.yaml` (a test checks that both are identical).
Without `--config`, the in-code values are used.

To change settings, copy `configs/default.yaml`, edit the copy and pass it with
`--config`. Keys you omit keep their reference value, so a file can contain only
the keys you change (see `configs/smoke_test.yaml`).

```yaml
pipeline:
  meta:
    episodes: 1000
datasets:
  UNSW-NB15:
    path: /data/unsw/UNSW-NB15.csv
```

Results reported in the manuscript use the reference values. Any change to the
`pipeline:` section changes the methodology; record the file you used together
with the results.

## `pipeline` keys

| Key | Reference | Meaning |
|---|---|---|
| `output_dir` | `outputs` | benchmark output folder |
| `test_size`, `val_size` | 0.20, 0.10 | stratified 70/10/20 split |
| `k_best` | 30 | ANOVA features kept |
| `min_class_size_to_keep` | 50 | balancing: drop smaller training classes |
| `target_min_class_size` | 10000 | balancing: SMOTE target |
| `max_class_size` | 50000 | balancing: undersampling cap |
| `max_rows` | 200000 | stratified cap on raw rows (`null` = none) |
| `min_class_presplit` | 10 | minimum class size before splitting |
| `cnn.epochs / batch_size / lr / patience` | 40 / 512 / 1e-3 / 5 | CNN training |
| `meta.episodes` | 500 | meta-training episodes |
| `meta.meta_batch` | 4 | tasks per meta-update |
| `meta.n_way / k_shot / q_query` | 5 / 20 / 15 | episode shape |
| `meta.inner_steps / inner_lr` | 5 / 0.01 | inner loop |
| `meta.meta_lr` | 1e-3 | outer Adam learning rate |
| `meta.reptile_lr / reptile_eps` | 0.05 / 0.5 | Reptile inner SGD rate / interpolation |
| `meta.hidden` | [128, 64] | MLP hidden sizes |
| `meta.adapt_shot / adapt_steps` | 20 / 50 | global adaptation |
| `meta.zero_day / zero_day_n_classes` | true / 2 | zero-day hold-out |
| `meta.eval_episodes` | 200 | episodic evaluation |
| `xai.n_background / n_explain` | 100 / 40 | SHAP/LIME sample sizes |
| `xai.kernel_nsamples` | 150 | SHAP KernelExplainer budget |
| `xai.lime_num_samples` | 1000 | LIME perturbations |
| `xai.top_k_list` | [5, 10, 15] | comprehensiveness / sufficiency |
| `xai.stability_*` | 5 / 3 / 0.05 | instances / perturbations / noise σ |
| `xai.infidelity_*` | 20 / 0.3 | perturbations / noise σ |
| `zeroday_detect.enabled` | true | run stage 6C |
| `zeroday_detect.score_methods` | msp, entropy, margin | novelty scores |
| `zeroday_detect.target_fprs` | 0.05, 0.10 | FPR budgets for TPR@FPR |
| `zeroday_detect.retrain_base_known_only` | true | re-fit LightGBM/CNN without zero-day families |
| `zeroday_detect.max_test_known` | 20000 | cap on known negatives |

## `datasets` keys

`path`, `label_candidates` (first matching column is the label) and `drop`
(columns removed). See [datasets.md](datasets.md).

## Fixed in code

`SEED = 42` and the device (`cuda` if available) are set in `config.py`;
model architectures and LightGBM settings are in `models/`.
