# Revision experiments (E1-E5)

Additional analyses requested by the reviewers of manuscript IOT-D-26-03171,
in `src/meta_ids/revision/`. They reuse the benchmark's preprocessing, models,
hyper-parameters and seed, so their numbers are directly comparable with the
manuscript tables.

```bash
python scripts/run_revision.py --exp all                 # everything, all datasets
python scripts/run_revision.py --exp E3 --datasets UNSW-NB15
```

| Exp. | Reviewer point | Module | Question |
|---|---|---|---|
| audit | protocol | `preparation.py` | Which families were held out, with which counts, and how many classes are in zero-day episodes? |
| E1 | R1-C4 | `e1_feature_sensitivity.py` | Do conclusions depend on the number *k* of ANOVA features, and is the selected set stable? |
| E2 | R1-C5/C7 | `e2_adaptation_budget.py` | Has global adaptation converged after 50 steps, and how sensitive is it to steps × learning rate? |
| E3 | R1-C6 | `e3_calibration.py` | Are the models calibrated, and does post-hoc calibration change open-set detection? |
| E4 | R1-C8 | `e4_latency.py` | What is the inference latency under a controlled protocol? |
| E5 | protocol check | `e5_episodic_ci.py` | Does restricting predictions to the episode's classes change episodic zero-day scores; what are their 95 % CIs? |

## Audit — `protocol_audit_classes.csv`, `protocol_audit.json`

Per class: training counts before/after balancing, test count, benign flag,
zero-day flag. Per dataset: withheld families, benign class used in zero-day
episodes, zero-day episode size, number of features after cleaning, test size.

## E1 — feature count *k* and selection stability

For each *k* in `--k-list`: rerun preparation with `SelectKBest(k)`, compute the
mean pairwise Jaccard index of the top-*k* sets selected on 10 stratified 80 %
sub-samples of the training split, then retrain LightGBM and the meta-learners
in `--e1-models`, and report closed-set macro-F1, episodic zero-day F1 and
open-set AUROC (MSP). Output: `E1_feature_sensitivity.csv`,
`tables/tab_ksens_rows.tex`, open-set figures in `<dataset>/E1_k<k>/`.

## E2 — global-adaptation budget

For the six gradient-based meta-learners, replays global adaptation (same
support set and parameters as the benchmark) for up to `--max-steps` steps with
every learning rate of `--lr-grid`, logging support loss/accuracy at every step
and test macro-F1 at steps 0, 10, 25, 50, 100, 150, 200. Meta-SGD is also run
with its learned per-parameter rates (`lr = learned`). Outputs:
`E2_trajectories.csv`, `E2_checkpoints.csv`,
`revision_global_adapt_convergence.png`, `tables/tab_adapt_rows.tex` (loss at
step 50, relative loss change between steps 40 and 50, F1 at the default
budget, best F1 and where it is reached).

## E3 — calibration

Uses the known-class models of stage 6C. A temperature *T* is fitted by grid
search (400 values, 0.02-50, log-spaced) minimising NLL on the known-class
**validation** split; an isotonic regression maps MSP confidence to accuracy on
the same split. Reports ECE (15 bins) before/after temperature scaling and the
open-set metrics of the three novelty scores before/after scaling, plus AUROC of
the isotonic score. Output: `E3_calibration.csv`, `tables/tab_calib_rows.tex`.

## E4 — latency

Hardware report (`hardware.json`), then for each model, device (CPU, and CUDA if
available) and batch size: one warm-up pass and `--repeats` timed passes over
max(batch size, 4,096) test flows (capped at the test-set size). Reports the median and inter-quartile range in ms per
1,000 flows (`E4_latency.csv`, `tables/tab_latency_rows.tex`). LightGBM is timed
on CPU only.

## E5 — episodic CIs and restricted decoding

Same sampler, seed and adaptation as the benchmark's episodic evaluation, but
returns per-episode statistics (mean, s.d., 95 % CI = mean ± 1.96 s.d./√n) and,
for gradient-based learners, also scores predictions restricted to the classes
present in the episode (ProtoNet does this by construction). Output:
`E5_episodic_ci.csv`, `tables/tab_episodic_ci_rows.tex`.

## LaTeX tables

`outputs_revision/tables/*.tex` contain only table **rows**, to paste into the
placeholder tables of the revised manuscript (cells marked `\tbd{}`).
