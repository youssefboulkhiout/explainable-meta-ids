"""LaTeX table bodies (pasted into the \\tbd{} placeholder tables)."""
import os

import numpy as np
import pandas as pd

import meta_ids as P                        # benchmark pipeline (unchanged)


# =============================================================================
# LaTeX table bodies (pasted into the \tbd{} placeholder tables)
# =============================================================================
def _fmt3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.3f}"


def write_latex_tables(outdir):
    tdir = P.ensure_dir(os.path.join(outdir, "tables"))
    p = lambda n: os.path.join(outdir, n)
    if os.path.exists(p("E1_feature_sensitivity.csv")):
        d = pd.read_csv(p("E1_feature_sensitivity.csv"))
        meta_cols = [c for c in d.columns if c.endswith("_F1")
                     and not c.startswith("LightGBM") and "_ZD_" not in c]
        lines = []
        for _, r in d.iterrows():
            best_meta = max(r[c] for c in meta_cols) if meta_cols else np.nan
            lines.append(f"{r['Dataset']} & {int(r['k'])} & "
                         f"{_fmt3(r['Jaccard_mean'])} & {_fmt3(r['LightGBM_F1'])} & "
                         f"{_fmt3(best_meta)} & {_fmt3(r.get('ProtoNet_ZD_F1'))} & "
                         f"{_fmt3(r.get('LightGBM_AUROC'))}\\\\")
        open(os.path.join(tdir, "tab_ksens_rows.tex"), "w").write("\n".join(lines))
    if os.path.exists(p("E2_checkpoints.csv")):
        c = pd.read_csv(p("E2_checkpoints.csv"))
        t = pd.read_csv(p("E2_trajectories.csv"))
        lines = []
        for (ds, m), g in c.groupby(["Dataset", "Model"], sort=False):
            base = g[(g["lr"].astype(str) == "0.01") & (g["step"] == 50)]
            best = g.loc[g["test_F1"].idxmax()]
            tt = t[(t["Dataset"] == ds) & (t["Model"] == m)
                   & (t["lr"].astype(str) == "0.01")]
            l40, l50 = (tt[tt["step"] == 40]["support_loss"].values,
                        tt[tt["step"] == 50]["support_loss"].values)
            rel = (l40[0] - l50[0]) / max(l40[0], 1e-12) if len(l40) and len(l50) else np.nan
            lines.append(f"{ds} & {m} & {_fmt3(l50[0] if len(l50) else np.nan)} & "
                         f"{_fmt3(rel)} & {_fmt3(base['test_F1'].values[0] if len(base) else np.nan)} & "
                         f"{_fmt3(best['test_F1'])} & {best['lr']}/{int(best['step'])}\\\\")
        open(os.path.join(tdir, "tab_adapt_rows.tex"), "w").write("\n".join(lines))
    if os.path.exists(p("E3_calibration.csv")):
        d = pd.read_csv(p("E3_calibration.csv"))
        g = d.groupby("Model", sort=False).agg(
            Tmin=("T", "min"), Tmax=("T", "max"), ECEr=("ECE_raw", "mean"),
            ECEt=("ECE_TS", "mean"), Ar=("MSP_raw_AUROC", "mean"),
            At=("MSP_TS_AUROC", "mean"), Ai=("MSP_iso_AUROC", "mean"),
            Fr=("MSP_raw_FPR@95TPR", "mean"), Ft=("MSP_TS_FPR@95TPR", "mean"))
        lines = [f"{m} & {r.Tmin:.2f}--{r.Tmax:.2f} & {_fmt3(r.ECEr)} & {_fmt3(r.ECEt)} & "
                 f"{_fmt3(r.Ar)} & {_fmt3(r.At)} & {_fmt3(r.Ai)} & {_fmt3(r.Fr)} & {_fmt3(r.Ft)}\\\\"
                 for m, r in g.iterrows()]
        open(os.path.join(tdir, "tab_calib_rows.tex"), "w").write("\n".join(lines))
    if os.path.exists(p("E4_latency.csv")):
        d = pd.read_csv(p("E4_latency.csv"))
        g = d.groupby(["Model", "device", "batch_size"], sort=False)[
            "median_ms_per_1k"].mean().unstack(["device", "batch_size"])
        lines = [" & ".join([m] + [_fmt3(v) for v in r.values]) + "\\\\"
                 for m, r in g.iterrows()]
        hdr = "% columns: " + ", ".join(f"{a}/bs={b}" for a, b in g.columns)
        open(os.path.join(tdir, "tab_latency_rows.tex"), "w").write(
            hdr + "\n" + "\n".join(lines))
    if os.path.exists(p("E5_episodic_ci.csv")):
        d = pd.read_csv(p("E5_episodic_ci.csv"))
        d = d[d["Regime"] == "zeroday"]
        lines = []
        for (ds, m), g in d.groupby(["Dataset", "Model"], sort=False):
            def cell(dec):
                s = g[g["Decoding"] == dec]
                return (f"{s['F1_mean'].values[0]:.3f} $\\pm$ "
                        f"{(s['F1_CI95_high'].values[0]-s['F1_mean'].values[0]):.3f}"
                        if len(s) else "--")
            lines.append(f"{ds} & {m} & {int(g['n_way'].values[0])} & "
                         f"{cell('full label space')} & {cell('episode-restricted')}\\\\")
        open(os.path.join(tdir, "tab_episodic_ci_rows.tex"), "w").write("\n".join(lines))
    P.log(f"LaTeX table bodies written to {tdir}")
