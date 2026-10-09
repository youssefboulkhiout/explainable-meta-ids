"""Per-dataset driver (run_benchmark_on_dataset) and command-line entry point (main)."""
import os
import time
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .config import CONFIG, DATASETS, DEVICE, META_MODELS, load_config, set_data_dir
from .utils import set_seed, ensure_dir, log, save_json, sanitize_probabilities
from .data import load_and_clean_dataset, split_and_standardize, select_kbest_features, balance_training_classes
from .models.baselines import train_lightgbm, expand_to_full_label_space, train_cnn, TorchClassifierWrapper
from .models.meta_learning import meta_train, global_few_shot_adapt, evaluate_episodic
from .evaluation.classification import (compute_classification_metrics, plot_confusion_matrix, plot_roc_curves,
                         plot_lightgbm_importance, plot_cnn_learning_curves, plot_meta_training_losses,
                         plot_runtime_comparison, zero_day_per_class_report,
                         plot_zero_day_figures)
from .evaluation.open_set import run_open_set_detection
from .explainability import explain_model
from .reporting import plot_dataset_summary, plot_cross_dataset_comparison


# =============================================================================
# MAIN PER-DATASET DRIVER
# =============================================================================
def run_benchmark_on_dataset(ds_name, dcfg, cfg):
    """End-to-end pipeline for a single dataset: load -> split -> select
    features -> balance -> train (base + meta) -> evaluate -> explain (XAI)
    -> report. Returns the five row-lists (performance, XAI, few-shot/zero-day
    episodic, zero-day per-class report, zero-day open-set detection) that
    main() aggregates across datasets."""
    set_seed()
    outdir = ensure_dir(os.path.join(cfg["output_dir"], ds_name))
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    xaidir = ensure_dir(os.path.join(outdir, "xai"))

    # 1-2. load / clean / split / scale ---------------------------------------
    X_df, y, le = load_and_clean_dataset(ds_name, dcfg, cfg)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), scaler = \
        split_and_standardize(X_df, y, cfg)

    # 3. feature selection ------------------------------------------------------
    X_tr, X_va, X_te, feat_names, _ = select_kbest_features(
        X_tr, y_tr, X_va, X_te, list(X_df.columns), cfg["k_best"])

    # 4. class balancing (train only) ------------------------------------------
    X_tr, y_tr, kept = balance_training_classes(X_tr, y_tr, cfg)
    # keep val/test consistent with whichever classes survived balancing
    keep_mask_va = np.isin(y_va, kept); X_va, y_va = X_va[keep_mask_va], y_va[keep_mask_va]
    keep_mask_te = np.isin(y_te, kept); X_te, y_te = X_te[keep_mask_te], y_te[keep_mask_te]
    n_classes = len(le.classes_)
    class_names = [str(c) for c in le.classes_]

    # zero-day held-out classes (rarest non-benign train classes) ---------------
    # These classes are excluded from meta-training entirely (meta_pool) so
    # the meta-learners never see them during episodic training; they're
    # later evaluated purely via few-shot global/episodic adaptation to
    # simulate detecting a brand-new ("zero-day") attack type.
    mcfg = cfg["meta"]
    train_counts = Counter(y_tr)
    benign_ids = [i for i, c in enumerate(le.classes_)
                  if any(k in str(c).lower() for k in ("benign", "normal"))]
    candidates = sorted([c for c in kept if c not in benign_ids],
                        key=lambda c: train_counts[c])   # rarest first
    zero_day = candidates[:mcfg["zero_day_n_classes"]] if mcfg["zero_day"] else []
    meta_pool = np.array([c for c in kept if c not in zero_day])
    if len(meta_pool) < 2:                      # safety fallback
        meta_pool, zero_day = np.array(kept), []
    log(f"[{ds_name}] meta-train classes: {len(meta_pool)}  "
        f"zero-day held-out: {[class_names[c] for c in zero_day]}")

    perf_rows, xai_rows, fewshot_rows = [], [], []
    mean_abs_shap = {}
    models = {}
    train_times, xai_times = {}, {}
    meta_histories = {}
    meta_state = {}    # algo -> (params, lr_dict); reused for known-only re-adapt (stage 6C)

    # 5A. base learners ---------------------------------------------------------     
    log(f"[{ds_name}] ----------- Training -----------")
    log(f"[{ds_name}] training LightGBM ...")
    lgbm, lgbm_tr_time  = train_lightgbm(X_tr, y_tr, X_va, y_va)
    train_times["LightGBM"] = lgbm_tr_time
    models["LightGBM"] = (expand_to_full_label_space(lgbm, n_classes), lgbm)
    plot_lightgbm_importance(lgbm, feat_names,
                         os.path.join(figdir, "lgbm_feature_importance.png"))

    log(f"[{ds_name}] training CNN ...")
    cnn, cnn_tr_time, cnn_hist = train_cnn(X_tr, y_tr, X_va, y_va,
                                           n_classes, cfg["cnn"])
    train_times["CNN"] = cnn_tr_time
    plot_cnn_learning_curves(cnn_hist, ds_name, os.path.join(figdir, "loss_cnn.png"))
    cnn_w = TorchClassifierWrapper(cnn, n_classes)
    models["CNN"] = (cnn_w.predict_proba, None)    
    print("\t")

    # 5B. meta-learners ---------------------------------------------------------
    for algo in META_MODELS:
        log(f"[{ds_name}] meta-training {algo} ...")
        # (1) episodic meta-training on meta_pool classes only (zero-day excluded)
        params, lr_dict, algo_tr_time, algo_hist = meta_train(
            algo, X_tr, y_tr, meta_pool, X_tr.shape[1], n_classes, mcfg)
        train_times[algo] = algo_tr_time
        meta_histories[algo] = algo_hist
        # keep the meta-learned initialisation so stage 6C can re-adapt it to
        # the known-only label space without repeating meta-training
        meta_state[algo] = (params, lr_dict)
        # (2) adapt the meta-init to the FULL label space for full-test-set eval
        wrapper = global_few_shot_adapt(algo, params, lr_dict, X_tr, y_tr,
                               n_classes, mcfg)
        models[algo] = (wrapper.predict_proba, None)

        # (3) separately measure "true" episodic few-shot performance on
        #     meta_pool classes, and (if enabled) zero-day performance on the
        #     held-out classes (+ one benign class so episodes are 2+ way)
        fs = evaluate_episodic(algo, params, lr_dict, X_te, y_te,
                           list(meta_pool), mcfg, tag="fewshot")
        zd = (evaluate_episodic(algo, params, lr_dict, X_te, y_te,
                            zero_day + benign_ids[:1], mcfg, tag="zeroday")
              if zero_day else {})
        fewshot_rows.append({"Dataset": ds_name, "Model": algo, **fs, **zd})
    plot_meta_training_losses(meta_histories, ds_name, figdir)
    save_json({a: h for a, h in meta_histories.items()},
              os.path.join(outdir, "meta_loss_histories.json"))
    save_json({"cnn": cnn_hist}, os.path.join(outdir, "cnn_loss_history.json"))

    # 6. evaluation --------------------------------------------------------------    
    log(f"[{ds_name}] ----------- Evaluation -----------")
    for name, (pp, clf) in models.items():
        t0 = time.time()
        proba = pp(X_te)                       # inference only
        eval_time = time.time() - t0
        proba = sanitize_probabilities(proba)          # guard: no NaN/inf into metrics/plots
        pred = np.argmax(proba, axis=1)
        m = compute_classification_metrics(y_te, pred, proba, n_classes)
        m.update({"Dataset": ds_name, "Model": name,
                  "Train_time_s": round(train_times.get(name, np.nan), 3),
                  "Eval_time_s": round(eval_time, 4),
                  "Eval_ms_per_1k": round(1000.0 * eval_time /
                                          max(len(X_te), 1) * 1000, 3)})
        perf_rows.append(m)
        log(f"[{ds_name}] {name:9s}  Acc={m['Accuracy']:.4f} "
            f"F1={m['F1']:.4f}  AUC={m['AUC']:.4f}  Kappa={m['Kappa']:.4f}  "
            f"train={m['Train_time_s']:.1f}s  eval={eval_time:.2f}s")
        plot_confusion_matrix(y_te, pred, class_names, f"{ds_name} -- {name}",
                       os.path.join(figdir, f"cm_{name}.png"), normalize=False)
        plot_confusion_matrix(y_te, pred, class_names, f"{ds_name} -- {name}",
                       os.path.join(figdir, f"cm-norm_{name}.png"))
        
        plot_roc_curves(y_te, proba, class_names, f"{ds_name} -- {name} ROC",
                 os.path.join(figdir, f"roc_{name}.png"))

    # 6b. zero-day global detection statistics (full test set) --------------------
    zeroday_rows = (zero_day_per_class_report(ds_name, models, X_te, y_te,
                                          zero_day, class_names)
                    if zero_day else [])

    # 6c. unified zero-day (open-set / novelty) detection for ALL models ----------
    #     Re-fits/re-adapts every model on the known classes only, then scores
    #     the held-out zero-day classes as an open-set detection problem.
    log(f"[{ds_name}] ----------- Zero-day detection (open-set) -----------")
    zeroday_detect_rows = run_open_set_detection(
        ds_name, meta_state, X_tr, y_tr, X_va, y_va, X_te, y_te,
        meta_pool, zero_day, n_classes, class_names, cfg, figdir, outdir)

    # 7. explainability ----------------------------------------------------------- 
    log(f"[{ds_name}] ----------- Explainability -----------")
    for name, (pp, clf) in models.items():
        log(f"[{ds_name}] XAI (SHAP + LIME) for {name} ...")
        t0 = time.time()
        try:
            rows, attr = explain_model(name, pp, clf, X_tr, X_te,
                                           feat_names, n_classes,
                                           cfg["xai"], xaidir,
                                           class_names=class_names)
            for r in rows:
                r["Dataset"] = ds_name
            xai_rows.extend(rows)
            mean_abs_shap[name] = np.abs(attr["SHAP"]).mean(0)
            np.save(os.path.join(xaidir, f"shap_values_{name}.npy"),
                    attr["SHAP"])
            np.save(os.path.join(xaidir, f"lime_values_{name}.npy"),
                    attr["LIME"])
        except Exception as e:
            log(f"[{ds_name}] XAI failed for {name}: {e}")
        xai_times[name] = time.time() - t0
    for m in perf_rows:                       # merge total XAI time per model
        m["XAI_time_s"] = round(xai_times.get(m["Model"], np.nan), 2)

    # 8. per-dataset reporting -----------------------------------------------------
    log(f"[{ds_name}] ----------- Reporting -----------")
    perf_df = pd.DataFrame(perf_rows)
    xai_df = pd.DataFrame(xai_rows)
    fs_df = pd.DataFrame(fewshot_rows)
    zd_df = pd.DataFrame(zeroday_rows)
    perf_df.to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)
    xai_df.to_csv(os.path.join(outdir, "xai_metrics.csv"), index=False)
    fs_df.to_csv(os.path.join(outdir, "fewshot_zeroday_metrics.csv"), index=False)
    if not zd_df.empty:
        zd_df.to_csv(os.path.join(outdir, "zeroday_class_report.csv"),
                     index=False)
    zdet_df = pd.DataFrame(zeroday_detect_rows)          # stage-6C open-set metrics
    if not zdet_df.empty:
        zdet_df.to_csv(os.path.join(outdir, "zeroday_detection_metrics.csv"),
                       index=False)
    perf_df[["Model", "Train_time_s", "Eval_time_s", "Eval_ms_per_1k",
             "XAI_time_s"]].to_csv(os.path.join(outdir, "timing.csv"),
                                   index=False)
    plot_runtime_comparison(perf_df, ds_name,
                         os.path.join(figdir, "time_comparison.png"))
    plot_zero_day_figures(zd_df if not zd_df.empty else None, fs_df,
                         ds_name, figdir)
    save_json({"dataset": ds_name, "n_features": len(feat_names),
               "features": feat_names, "n_classes": n_classes,
               "classes": class_names,
               "zero_day_classes": [class_names[c] for c in zero_day]},
              os.path.join(outdir, "dataset_summary.json"))
    if not xai_df.empty:
        plot_dataset_summary(ds_name, perf_df, xai_df, mean_abs_shap,
                                feat_names, figdir)
    log(f"[{ds_name}] DONE -> {outdir}")
    return perf_rows, xai_rows, fewshot_rows, zeroday_rows, zeroday_detect_rows


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """CLI entry point: parse args, build the effective config, run every
    requested dataset through run_benchmark_on_dataset(), then produce cross-dataset
    aggregate CSVs + comparison figures."""
    ap = argparse.ArgumentParser(
        description="Explainable Meta-Learning for Few-Shot and Zero-Day "
                    "Intrusion Detection in IoT Networks")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--episodes", type=int, default=None,
                    help="override meta-training episodes")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="override the stratified row cap")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke-test configuration")
    ap.add_argument("--config", type=str, default=None,
                    help="optional YAML file overriding CONFIG / DATASETS")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="folder containing the dataset CSV files")
    args = ap.parse_args()

    # start from the global default CONFIG (optionally overridden by a YAML
    # file), then apply CLI overrides
    cfg = load_config(args.config, CONFIG)
    if args.data_dir:
        set_data_dir(args.data_dir)
    if args.episodes:
        cfg["meta"]["episodes"] = args.episodes
    if args.max_rows:
        cfg["max_rows"] = args.max_rows
    if args.output:
        cfg["output_dir"] = args.output
    if args.quick:
        # aggressively shrink every expensive knob for a fast smoke test
        cfg["max_rows"] = 30_000
        cfg["meta"].update(episodes=60, eval_episodes=20, adapt_steps=30)
        cfg["cnn"].update(epochs=8)
        cfg["xai"].update(n_explain=15, kernel_nsamples=100,
                          lime_num_samples=500, stability_instances=3,
                          stability_perturbs=2)

    ensure_dir(cfg["output_dir"])
    set_seed()
    log(f"device: {DEVICE}")

    # run each dataset independently; one dataset failing doesn't abort the rest
    all_perf, all_xai, all_fs, all_zd, all_zdet = [], [], [], [], []
    for ds in args.datasets:
        log(f"===== {ds} =====")
        try:
            p, x, f, z, zdet = run_benchmark_on_dataset(ds, DATASETS[ds], cfg)
            all_perf += p; all_xai += x; all_fs += f; all_zd += z
            all_zdet += zdet
        except FileNotFoundError as e:
            log(str(e))                     # missing CSV -> skip, don't crash
        except Exception as e:
            import traceback
            log(f"[{ds}] FAILED: {e}")
            traceback.print_exc()

    # cross-dataset aggregation: combined CSVs + comparison heatmaps/bars
    if all_perf:
        pd.DataFrame(all_perf).to_csv(
            os.path.join(cfg["output_dir"], "all_model_metrics.csv"), index=False)
        pd.DataFrame(all_xai).to_csv(
            os.path.join(cfg["output_dir"], "all_xai_metrics.csv"), index=False)
        pd.DataFrame(all_fs).to_csv(
            os.path.join(cfg["output_dir"], "all_fewshot_zeroday.csv"), index=False)
        if all_zd:
            pd.DataFrame(all_zd).to_csv(
                os.path.join(cfg["output_dir"], "all_zeroday_class_report.csv"),
                index=False)
        if all_zdet:
            pd.DataFrame(all_zdet).to_csv(
                os.path.join(cfg["output_dir"], "all_zeroday_detection.csv"),
                index=False)
        plot_cross_dataset_comparison(all_perf, all_xai, all_fs, all_zd, all_zdet,
                              cfg["output_dir"])
        log(f"ALL DONE -> {cfg['output_dir']}/")
