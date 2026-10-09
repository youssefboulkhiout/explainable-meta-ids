"""
=============================================================================
 Revision experiments for manuscript IOT-D-26-03171
 "Explainable Meta-Learning for Few-Shot and Zero-Day Intrusion Detection
  in IoT Networks"
=============================================================================
 This sub-package EXTENDS the benchmark pipeline (package meta_ids, imported
 as `P` and left untouched) with the additional analyses requested by the
 reviewers. Every experiment re-uses the exact preprocessing, models and
 hyper-parameters of the original pipeline (same seed, same CONFIG), so the
 new numbers are directly comparable with the tables of the manuscript.

   E1  (Reviewer 1, C4)  Sensitivity to the number of ANOVA features k and
                         stability of the selected feature set.
   E2  (Reviewer 1, C5/C7) Convergence of the global K-shot adaptation and
                         ablation of its budget (steps x inner learning rate),
                         including Meta-SGD with its learned per-parameter rates.
   E3  (Reviewer 1, C6)  Post-hoc calibration (temperature scaling fitted on the
                         known-class VALIDATION split; isotonic recalibration of
                         the MSP score) and its effect on ECE and on open-set
                         detection (AUROC / AUPR / FPR@95).
   E4  (Reviewer 1, C8)  Controlled inference-latency protocol: hardware report,
                         warm-up, repeated timing, several batch sizes, CPU and
                         (if available) GPU.
   E5  (protocol check)  Episodic zero-day evaluation with predictions restricted
                         to the episode's classes (as ProtoNet does), plus 95%
                         confidence intervals over episodes.

 Every experiment writes CSV files and a LaTeX table body
 (outputs_revision/tables/*.tex) whose rows can be pasted into the
 placeholder tables of the revised manuscript (cells marked \\tbd{}).

 Usage
 -----
   python scripts/run_revision.py --exp all
   python scripts/run_revision.py --exp E1 E3 --datasets RT-IoT2022
   python scripts/run_revision.py --exp E1 --k-list 10 20 30 40
=============================================================================
"""
