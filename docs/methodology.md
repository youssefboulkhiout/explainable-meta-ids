# Methodology implemented by the code

This page describes what the code does, stage by stage, with the reference
values of `CONFIG` (`configs/default.yaml`). Module names refer to
`src/meta_ids/`. A single seed (`SEED = 42`) is used throughout.

## Stages 1-4 — data preparation (`data.py`)

Cleaning, 70/10/20 stratified split, `StandardScaler` and
`SelectKBest(f_classif, k=30)` fitted on the training split only, then
train-only balancing (drop < 50, SMOTE up to 10,000, undersample above 50,000).
Details: [datasets.md](datasets.md#what-the-pipeline-does-to-every-dataset).

### Zero-day hold-out

After balancing, the `zero_day_n_classes = 2` **rarest non-benign** classes of
the (balanced) training split are held out as zero-day families. Benign classes
are those whose name contains `benign` or `normal`. Ties in training counts
(frequent after SMOTE) are broken by label index. The remaining classes form
the *meta-training pool*; meta-learners never see the zero-day families during
meta-training.

## Stage 5A — base learners (`models/baselines.py`)

| Model | Settings |
|---|---|
| LightGBM | 400 trees, learning rate 0.05, 63 leaves, subsample 0.9, colsample 0.9, early stopping (40 rounds) on the validation split |
| 1-D CNN | Conv1d(1→32, k=3) – ReLU – MaxPool(2) – Conv1d(32→64, k=3) – ReLU – AdaptiveAvgPool(4) – FC 256→128 – ReLU – Dropout 0.3 – FC; Adam (1e-3), batch 512, ≤ 40 epochs, early stopping (patience 5) on validation loss, best weights restored |

Base learners are trained on **all** classes, including the zero-day families,
and therefore act as a fully supervised reference in stages 6 and 6B.

## Stage 5B — meta-learners (`models/meta_learning.py`)

All seven share the same MLP (input → 128 → 64 → classes, ReLU,
Kaiming-uniform initialisation), the same episodic sampler (5-way, 20-shot,
15 queries per class, original label ids with a shared global head) and 500
episodes with 4 tasks per meta-update. The meta-gradient of the Adam-based
algorithms is clipped at norm 5 (Reptile has no meta-gradient).

| Algorithm | Inner loop (5 steps, lr 0.01) | Outer update |
|---|---|---|
| MAML | all parameters, second order | Adam 1e-3 |
| FOMAML | all parameters, first order | Adam 1e-3 |
| Meta-SGD | all parameters, learned per-parameter rates (init 0.01), second order | Adam 1e-3 on weights and rates |
| ANIL | head only, first order | Adam 1e-3 |
| BOIL | body only, first order | Adam 1e-3 |
| Reptile | all parameters, SGD lr 0.05 on support ∪ query | interpolation towards adapted weights, ε = 0.5 linearly annealed to 0 |
| ProtoNet | none (prototype = mean embedding, squared Euclidean distance) | Adam 1e-3 |

**Global adaptation** (used for the full test set, stage 6): from the
meta-learned initialisation, 50 plain-SGD steps (lr 0.01) on a support set of
20 training examples per class over **all** classes, updating the same
parameter subset as the inner loop (ProtoNet recomputes prototypes). Meta-SGD
uses the scalar rate here, not its learned rates (examined in E2).

**Episodic evaluation** (few-shot and zero-day): 200 episodes sampled from the
**test** split (seed `SEED + 7`), N-way = min(5, available classes), 20-shot,
15 queries; 10 SGD steps (2 × inner steps) on the episode's support set, then
prediction on its queries. Classes need ≥ 22 test rows; classes with fewer than
35 rows are sampled with replacement.
- *Few-shot*: classes of the meta-training pool.
- *Zero-day*: the held-out families plus the first benign class, if any.

## Stage 6 / 6B — closed-set evaluation (`evaluation/classification.py`)

Accuracy, macro precision / recall / F1, Cohen's κ and one-vs-rest macro AUC
(over the classes present in the test split), confusion matrices, ROC curves,
training and inference times. Stage 6B reports, for each zero-day family, the
detection rate (recall), precision and F1 on the full test set.

## Stage 6C — open-set zero-day detection (`evaluation/open_set.py`)

1. Every model is rebuilt **without** the zero-day families: LightGBM and the
   CNN are re-fitted on the known classes; meta-learners are re-adapted (global
   adaptation) on the known classes and their outputs restricted and
   renormalised to the known columns.
2. Open-set test set: known-class test flows (randomly capped at 20,000) as
   negatives, all zero-day test flows as positives.
3. Novelty scores from the known-class probabilities *p*:
   MSP = 1 − max *p*; normalised entropy *H(p)* / log *K*; margin = 1 − (p₍₁₎ − p₍₂₎).
4. Metrics: AUROC, AUPR, FPR at 95 % TPR, TPR at FPR 5 % and 10 %, best F1 over
   the precision–recall curve.

## Stage 7 — explainability (`explainability.py`)

For each model: 100 training rows as background, 40 test rows explained.
SHAP uses `TreeExplainer` for LightGBM and `KernelExplainer` (25 k-means
background points, 150 samples) for the other models; LIME uses
`LimeTabularExplainer` (no discretisation, 1000 samples). Attributions are those
of the predicted class. The reference input for removal-based metrics is the
training mean.

| Metric | Definition | Better |
|---|---|---|
| Fidelity | Spearman correlation between \|attribution\| and the probability drop when each feature alone is replaced by its mean | higher |
| Infidelity | Yeh et al. (2019), Gaussian perturbations σ = 0.3, 20 draws | lower |
| Sparsity | Gini index of \|attributions\| | higher |
| Stability | relative L2 change of the explanation under N(0, 0.05) input noise (5 instances × 3) | lower |
| Comprehensiveness | probability drop when the top-k features are removed, k ∈ {5, 10, 15} | higher |
| Sufficiency | probability drop when only the top-k features are kept | lower |
| Feature agreement | top-10 overlap between SHAP and LIME | higher |
| Rank correlation | Spearman correlation of global mean \|SHAP\| vs \|LIME\| | higher |

## Stage 8 — reporting (`reporting.py`)

Per-dataset heatmaps (performance, explanation quality, mean |SHAP|) and
cross-dataset comparison figures. See [outputs.md](outputs.md).
