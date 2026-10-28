"""
13_normalization_backtest.py — reproducibility of the normalization effect across training periods.
Default architectures with RevIN and the fixed training budget (no early stopping, no tuning) and the
canonical PatchTST, retrained for each training cutoff (2018-12 ... 2022-12), compared with the classical
baselines on identical origins (the same layout as Table 17 / results/T11_backtest.csv).
Output: results_r3/R3_backtest.csv, manuscript_assets/r3_backtest.tex, r3_summary_backtest.json
"""
import glob
import json

import numpy as np
import pandas as pd

from common import ROOT, H

CUTS = ["2018-12", "2019-12", "2020-12", "2021-12", "2022-12"]
SEEDS = range(10)
LAB = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "nbeats": "MLP", "patchtst": "Simplified patch Transformer",
       "nbeats_c": "N-BEATS (canonical)", "patchtst_c": "PatchTST (canonical)", "naive": "Naive",
       "snaive": "Seasonal Naive", "ets": "ETS", "sarima": "SARIMA"}
ORDER = ["lstm", "gru", "tcn", "nbeats", "patchtst", "nbeats_c", "patchtst_c"]


def rd(pattern):
    fs = glob.glob(str(pattern))
    return pd.concat((pd.read_csv(f) for f in fs), ignore_index=True) if fs else pd.DataFrame()


rows = []
for c in CUTS:
    parts = [rd(ROOT / "runs_r3" / "revin_fixed" / c / "*" / "*" / "seed*.csv").assign(norm="revin"),
             rd(ROOT / "runs_r3" / "global_fixed" / c / "*" / "*" / "seed*.csv").assign(norm="global"),
             rd(ROOT / "runs" / c / "classical" / "*" / "seed*.csv").assign(norm="--")]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d = d[(d.strategy == "classical") | d.seed.isin(SEEDS)]
    d = d[(d.strategy == "classical") | (d.norm == "revin") | (d.model == "patchtst_c")]
    d = d[d.origin_period.isin(["backtest", "val", "test"])]
    c0 = pd.Period(c, "M")
    d = d[d.origin.isin([str(c0 + k) for k in range(1, 32)])]       # 31 months after the cutoff, as in Table 17
    cnt = d[d.strategy == "classical"].groupby("origin").h.nunique()
    comp = set(cnt[cnt == H].index)
    d["ape"] = 200 * (d.yhat - d.y).abs() / (d.y.abs() + d.yhat.abs())
    d["method"] = np.where(d.strategy == "classical", d.model, d.model + "-" + d.strategy)
    for m, g in d.groupby("method"):
        h1 = g[g.h == 1].groupby("seed").ape.mean()
        fu = g[g.origin.isin(comp)].groupby("seed").ape.mean()
        rows.append(dict(cutoff=c, method=m, model=g.model.iloc[0], strategy=g.strategy.iloc[0],
                         n_seeds=g.seed.nunique(), n_complete_origins=len(comp),
                         h1_sMAPE_mean=h1.mean(), full_sMAPE_mean=fu.mean(), full_sMAPE_sd=fu.std(ddof=1)))
bt = pd.DataFrame(rows)
bt["h1_rank"] = bt.groupby("cutoff").h1_sMAPE_mean.rank()
bt["full_rank"] = bt.groupby("cutoff").full_sMAPE_mean.rank()
(ROOT / "results_r3").mkdir(exist_ok=True)
bt.to_csv(ROOT / "results_r3" / "R3_backtest.csv", index=False)

piv = bt.pivot_table(index="method", columns="cutoff", values="full_sMAPE_mean")
mr = bt.groupby("method")[["h1_rank", "full_rank"]].mean()
methods = [f"{m}-{s}" for s in ["direct", "recursive"] for m in ORDER] + ["naive", "snaive", "ets", "sarima"]
methods = [m for m in methods if m in piv.index]
best_cls = piv.loc[["naive", "snaive", "ets", "sarima"]].min()
lines = []
for i, m in enumerate(methods):
    name = f"{LAB[m.split('-')[0]]} ({m.split('-')[1]})" if "-" in m else LAB[m]
    vals = []
    for c in CUTS:
        v = piv.loc[m, c] if c in piv.columns else np.nan
        vals.append("--" if pd.isna(v) else (f"\\textbf{{{v:.1f}}}" if v == piv[c].min() else f"{v:.1f}"))
    lines.append(f"{name} & " + " & ".join(vals) + f" & {mr.loc[m, 'h1_rank']:.1f} & {mr.loc[m, 'full_rank']:.1f} \\\\")
    if m in ("patchtst_c-direct", "patchtst_c-recursive"):
        lines.append("\\addlinespace")
n = len(methods)
tab = ["\\begin{table}[htbp]", "\\centering",
       "\\caption{Reproducibility of the normalization effect across training periods: mean sMAPE (\\%) over "
       "$h=1,\\ldots,12$ for each training cutoff (same origins as Table~\\ref{tab:backtest}; mean over 10 seeds) "
       "for the default architectures with RevIN and the fixed 250-epoch budget (no early stopping, no tuning) "
       "and for the canonical PatchTST, together with the classical baselines; mean rank across the five cutoffs "
       f"for rolling $h=1$ and for the full horizon among the {n} configurations shown. Bold: lowest value in the column. "
       "The corresponding results under the global normalization are given in Table~\\ref{tab:backtest}.}",
       "\\label{tab:r3_backtest}", "\\vspace{4pt}", "\\small", "\\begin{adjustbox}{max width=\\textwidth}",
       "\\begin{tabular}{lccccccc}", "\\toprule",
       "& \\multicolumn{5}{c}{\\textbf{Training cutoff}} & \\multicolumn{2}{c}{\\textbf{Mean rank}} \\\\",
       "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}",
       "\\textbf{Configuration (RevIN)} & 2018-12 & 2019-12 & 2020-12 & 2021-12 & 2022-12 & $h=1$ & Full \\\\",
       "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
(ROOT / "manuscript_assets" / "r3_backtest.tex").write_text("\n".join(tab) + "\n")

neural = [m for m in methods if "-" in m]
summ = {"best_classical_by_cutoff": best_cls.round(2).to_dict(),
        "n_neural_beating_best_classical_by_cutoff": {c: int((piv.loc[neural, c] < best_cls[c]).sum()) for c in CUTS
                                                      if c in piv.columns},
        "n_neural": len(neural),
        "mean_rank": mr.round(2).to_dict("index"),
        "full_by_cutoff": piv.round(2).to_dict("index")}
(ROOT / "results_r3" / "r3_summary_backtest.json").write_text(json.dumps(summ, indent=1))
print(piv.loc[methods].round(1).to_string())
print(mr.loc[methods].round(1).to_string())
print(json.dumps({k: summ[k] for k in ["best_classical_by_cutoff", "n_neural_beating_best_classical_by_cutoff"]}, indent=1))
