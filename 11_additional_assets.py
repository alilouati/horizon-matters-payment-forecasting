"""
11_additional_assets.py — LaTeX tables and figures for the additional experiments
(hyperparameter search, tuned models, learning curves, second series, multivariate inputs,
bootstrap intervals). Run after run_additional.py and 10_bootstrap.py.
Output: manuscript_assets/r2_*.tex, r2_*.png and results_r2/r2_summary.json
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT
import cfgmodels

OUT = ROOT / "manuscript_assets"
OUT.mkdir(exist_ok=True)
NEURAL = ["lstm", "gru", "tcn", "nbeats", "patchtst"]
CLASSICAL = ["naive", "snaive", "ets", "sarima"]
LAB = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "nbeats": "MLP", "patchtst": "Simplified patch Transformer",
       "naive": "Naive", "snaive": "Seasonal Naive", "ets": "ETS", "sarima": "SARIMA"}
f = lambda x, d=2: "--" if pd.isna(x) else f"{x:.{d}f}"
summary = {}


def label(m):
    if "-" in m:
        a, s = m.split("-")
        return f"{LAB[a]} ({s})"
    return LAB[m]


def write(name, s):
    (OUT / name).write_text(s)


def t1(path):
    return pd.read_csv(ROOT / path / "T1_rolling_h1.csv")


def t2(path):
    return pd.read_csv(ROOT / path / "T2_horizon_profile.csv")


def lookup(t, m, period="test", col="sMAPE_mean"):
    x = t[(t.method == m) & (t.period == period)]
    return float(x[col].iloc[0]) if len(x) else np.nan


def moh(t, m):
    x = t[t.method == m]
    return float(x.sMAPE_mean_over_h.iloc[0]) if len(x) else np.nan


# ---------------------------------------------------------------- search space
rows = []
for m in NEURAL:
    sp = "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in cfgmodels.SEARCH_ARCH[m].items())
    rows.append(f"{LAB[m]} & {sp.replace('_', ' ')} \\\\")
sp = "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in cfgmodels.SEARCH_TRAIN.items())
rows.append(f"All (training) & {sp.replace('_', ' ')} \\\\")
write("r2_hpo_space.tex", "\n".join(rows) + "\n")

# ---------------------------------------------------------------- selected configurations
best = json.loads((ROOT / "results_r2" / "hpo_best.json").read_text())
trials = pd.read_csv(ROOT / "results_r2" / "hpo_trials.csv")


def cfg_str(m, c):
    a, t = c["arch"], c["train"]
    keys = {"lstm": ["hidden", "layers", "dropout"], "gru": ["hidden", "layers", "dropout"],
            "tcn": ["channels", "kernel", "dilations", "dropout"], "nbeats": ["hidden", "layers", "dropout"],
            "patchtst": ["patch_len", "d_model", "nhead", "layers", "dim_ff", "dropout"]}[m]
    short = {"hidden": "h", "layers": "l", "dropout": "do", "channels": "ch", "kernel": "k", "dilations": "d",
             "patch_len": "P", "d_model": "dm", "nhead": "heads", "dim_ff": "ff"}
    arch = ", ".join(f"{short[k]}={'(' + ','.join(map(str, a[k])) + ')' if isinstance(a[k], list) else a[k]}" for k in keys)
    tr = f"lr={t['lr']:g}, b={t['batch']}, ep={t['epochs']}, wd={t['weight_decay']:g}"
    return arch, tr


rows, sel = [], {}
for s in ["direct", "recursive"]:
    for m in NEURAL:
        b = best[f"{m}-{s}"]
        arch, tr = cfg_str(m, b["cfg"])
        g = trials[(trials.strategy == s) & (trials.model == m)]
        rank_default = int((g.val_sMAPE_mean < g[g.trial == 0].val_sMAPE_mean.iloc[0]).sum()) + 1
        sel[f"{m}-{s}"] = dict(trial=b["trial"], default_val=b["default_val_sMAPE"], tuned_val=b["val_sMAPE"],
                               default_rank=rank_default, n_params=b["n_params"])
        npar = f"{b['n_params']:,}".replace(",", "{,}")
        rows.append(f"{LAB[m]} ({s}) & {arch} & {tr} & {npar} & {f(b['default_val_sMAPE'])} & "
                    f"{f(b['val_sMAPE'])} & {'default' if b['trial'] == 0 else b['trial']} \\\\")
    if s == "direct":
        rows.append("\\addlinespace")
write("r2_hpo_selected.tex", "\n".join(rows) + "\n")
summary["hpo"] = sel

# ---------------------------------------------------------------- tuned vs default (test)
d1, d2 = t1("results"), t2("results")
u1, u2 = t1("results_tuned"), t2("results_tuned")
rows, tv = [], {}
for s in ["direct", "recursive"]:
    for m in NEURAL:
        k = f"{m}-{s}"
        a = (lookup(d1, k), lookup(d1, k, col="sMAPE_sd"), moh(d2, k))
        b = (lookup(u1, k), lookup(u1, k, col="sMAPE_sd"), moh(u2, k))
        tv[k] = dict(default_h1=a[0], tuned_h1=b[0], default_full=a[2], tuned_full=b[2],
                     default_val=lookup(d1, k, "val"), tuned_val_rolling=lookup(u1, k, "val"))
        rows.append(f"{LAB[m]} ({s}) & {f(lookup(d1, k, 'val'))} & {f(lookup(u1, k, 'val'))} & "
                    f"{f(a[0])} $\\pm$ {f(a[1])} & {f(b[0])} $\\pm$ {f(b[1])} & {f(a[2])} & {f(b[2])} \\\\")
    rows.append("\\addlinespace")
for c in CLASSICAL:
    rows.append(f"{LAB[c]} & {f(lookup(d1, c, 'val'))} & -- & {f(lookup(d1, c))} & -- & {f(moh(d2, c))} & -- \\\\")
write("r2_tuned_vs_default.tex", "\n".join(rows) + "\n")
summary["tuned_vs_default"] = tv

# best tuned neural per horizon vs classical
p = u2.pivot_table(index="method", columns="h", values="sMAPE_mean")
pc = d2.pivot_table(index="method", columns="h", values="sMAPE_mean").loc[CLASSICAL]
neural = [f"{m}-{s}" for s in ["direct", "recursive"] for m in NEURAL]
summary["tuned_beats_best_classical_at_h"] = [int(h) for h in p.columns if p.loc[neural, h].min() < pc[h].min()]
summary["tuned_best_full"] = {k: moh(u2, k) for k in neural}
summary["classical_full"] = {c: moh(d2, c) for c in CLASSICAL}
summary["tuned_rank_h1_h12"] = {}
allp = pd.concat([p.loc[neural], pc])
r = allp.rank()
for k in neural:
    summary["tuned_rank_h1_h12"][k] = (int(r.loc[k, 1]), int(r.loc[k, 12]))
# DM tuned vs classical (rolling h1)
dm = pd.read_csv(ROOT / "results_tuned" / "T4_dm_tests.csv")
dm1 = dm[dm.evaluation == "rolling_h1_test_n19"]
sig = []
for k in neural:
    for c in ["sarima", "ets", "snaive"]:
        x = dm1[((dm1.method_A == k) & (dm1.method_B == c)) | ((dm1.method_A == c) & (dm1.method_B == k))]
        if len(x):
            st = x.dm_stat.iloc[0] if x.method_A.iloc[0] == k else -x.dm_stat.iloc[0]
            if x.p_value.iloc[0] < 0.05 and st < 0:
                sig.append((k, c, float(x.p_value.iloc[0])))
summary["tuned_significantly_better_than"] = sig

# ---------------------------------------------------------------- learning curves figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11})


def curves(folder):
    fs = glob.glob(str(ROOT / folder / "_curves" / "*.csv"))
    return pd.concat((pd.read_csv(x) for x in fs), ignore_index=True) if fs else pd.DataFrame()


cd, ct = curves("runs_curves_default"), curves("runs_tuned")
fig, axes = plt.subplots(2, 5, figsize=(17, 6.6), constrained_layout=True)
over = {}
for i, s in enumerate(["direct", "recursive"]):
    for j, m in enumerate(NEURAL):
        ax = axes[i, j]
        for data, ls, tag in [(cd, "--", "default"), (ct, "-", "tuned")]:
            g = data[(data.strategy == s) & (data.model == m)]
            if not len(g):
                continue
            q = g.groupby("epoch")[["train_loss", "val_loss"]].median()
            ax.plot(q.index, q.train_loss, ls, color="#1f77b4", lw=1.6, label=f"training ({tag})")
            ax.plot(q.index, q.val_loss, ls, color="#d62728", lw=1.6, label=f"validation ({tag})")
            vmin_ep = int(q.val_loss.idxmin())
            over[f"{m}-{s}-{tag}"] = dict(final_train=float(q.train_loss.iloc[-1]), final_val=float(q.val_loss.iloc[-1]),
                                          min_val=float(q.val_loss.min()), min_val_epoch=vmin_ep,
                                          epochs=int(q.index.max()))
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{LAB[m]} ({s})")
        if i == 1:
            ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("MSE (standardized scale)")
        ax.grid(alpha=0.3)
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=4, frameon=False)
fig.savefig(OUT / "r2_learning_curves.png", dpi=300)
plt.close(fig)
summary["curves"] = over

# ---------------------------------------------------------------- second series
x1, x2 = t1("results_tx"), t2("results_tx")
rows, tx = [], {}
methods = neural + CLASSICAL
for k in methods:
    tx[k] = dict(h1=lookup(x1, k), h1_sd=lookup(x1, k, col="sMAPE_sd"), full=moh(x2, k),
                 sales_h1=lookup(d1, k), sales_full=moh(d2, k))
rk = pd.DataFrame(tx).T
rk["rank_tx_full"] = rk.full.rank()
rk["rank_sales_full"] = rk.sales_full.rank()
rk["rank_tx_h1"] = rk.h1.rank()
rk["rank_sales_h1"] = rk.sales_h1.rank()
for k in methods:
    r_ = rk.loc[k]
    sd = "" if k in CLASSICAL else f" $\\pm$ {f(r_.h1_sd)}"
    rows.append(f"{label(k)} & {f(r_.h1)}{sd} & {int(r_.rank_tx_h1)} & {f(r_.full)} & {int(r_.rank_tx_full)} & "
                f"{int(r_.rank_sales_full)} \\\\")
    if k in ("patchtst-direct", "patchtst-recursive"):
        rows.append("\\addlinespace")
write("r2_second_series.tex", "\n".join(rows) + "\n")
summary["second_series"] = dict(
    table=rk.round(3).to_dict(orient="index"),
    spearman_full=float(stats.spearmanr(rk.rank_tx_full, rk.rank_sales_full)[0]),
    spearman_h1=float(stats.spearmanr(rk.rank_tx_h1, rk.rank_sales_h1)[0]),
    best_full=rk.full.idxmin(), best_h1=rk.h1.idxmin())
xp = x2.pivot_table(index="method", columns="h", values="sMAPE_mean")
summary["second_series"]["classical_best_at_h"] = [int(h) for h in xp.columns
                                                   if xp.loc[CLASSICAL, h].min() <= xp.loc[neural, h].min()]
xr = xp.loc[methods].rank()
summary["second_series"]["rank_h1_h12"] = {k: (int(xr.loc[k, 1]), int(xr.loc[k, 12])) for k in neural}

# ---------------------------------------------------------------- multivariate
m1, m2 = t1("results_mv"), t2("results_mv")
rows, mv = [], {}
for m in NEURAL:
    k = f"{m}-direct"
    a = (lookup(d1, k), lookup(d1, k, col="sMAPE_sd"), moh(d2, k))
    b = (lookup(m1, k), lookup(m1, k, col="sMAPE_sd"), moh(m2, k))
    # paired by seed: share of seeds in which the multivariate model is more accurate (rolling h=1)
    mv[k] = dict(uni_h1=a[0], mv_h1=b[0], uni_full=a[2], mv_full=b[2])
    rows.append(f"{LAB[m]} & {f(a[0])} $\\pm$ {f(a[1])} & {f(b[0])} $\\pm$ {f(b[1])} & {f(b[0] - a[0])} & "
                f"{f(a[2])} & {f(b[2])} & {f(b[2] - a[2])} \\\\")
write("r2_multivariate.tex", "\n".join(rows) + "\n")
summary["multivariate"] = mv

# ---------------------------------------------------------------- bootstrap table
bt = pd.read_csv(ROOT / "results" / "T13_bootstrap.csv")
rows = []
for _, r_ in bt.iterrows():
    rows.append(f"{label(r_.method_A)} -- {label(r_.method_B)} & {f(r_.h1_diff)} & [{f(r_.h1_ci_lo)}, {f(r_.h1_ci_hi)}] & "
                f"{f(r_.full_diff)} & [{f(r_.full_ci_lo)}, {f(r_.full_ci_hi)}] \\\\")
write("r2_bootstrap.tex", "\n".join(rows) + "\n")

(ROOT / "results_r2").mkdir(exist_ok=True)
(ROOT / "results_r2" / "r2_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str)[:6000])
