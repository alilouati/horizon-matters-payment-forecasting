"""
06_manuscript_assets.py — LaTeX tables and figures used in the manuscript,
generated only from runs/ and results/ (run 05_analyze.py first).

Output: manuscript_assets/tab_*.tex and manuscript_assets/fig_*.png
"""
import glob

import numpy as np
import pandas as pd

from common import RUNS_DIR, RESULTS_DIR, ROOT, ANOMALY_FILE, H, smape, load_series, TARGET, TRAIN_END, VAL_END

OUT = ROOT / "manuscript_assets"
OUT.mkdir(exist_ok=True)

NEURAL = ["lstm", "gru", "tcn", "nbeats", "patchtst"]
VARIANTS = ["tcn24", "patchtst_small"]
CLASSICAL = ["naive", "snaive", "ets", "sarima"]
LABEL = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "nbeats": "MLP", "patchtst": "Simplified patch Transformer",
         "tcn24": "TCN-24 (receptive field 31)", "patchtst_small": "Simplified patch Transformer (small)",
         "naive": "Naive", "snaive": "Seasonal Naive", "ets": "ETS", "sarima": "SARIMA"}


def label(method):
    if "-" in method:
        m, s = method.split("-")
        return f"{LABEL[m]} ({s})"
    return LABEL[method]


def f(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def write(name, s):
    (OUT / name).write_text(s)


# ---------------------------------------------------------------------------
t1 = pd.read_csv(RESULTS_DIR / "T1_rolling_h1.csv")
t2 = pd.read_csv(RESULTS_DIR / "T2_horizon_profile.csv")
t4 = pd.read_csv(RESULTS_DIR / "T4_dm_tests.csv")
t5 = pd.read_csv(RESULTS_DIR / "T5_bias.csv")
t8 = pd.read_csv(RESULTS_DIR / "T8_anomaly_sensitivity.csv")
t11 = pd.read_csv(RESULTS_DIR / "T11_backtest.csv")
t3 = pd.read_csv(RESULTS_DIR / "T3_win_probability.csv")
logs = sorted(glob.glob(str(RUNS_DIR / "_logs" / "*.csv")))
lg = pd.concat((pd.read_csv(x) for x in logs)) if logs else pd.read_csv(RUNS_DIR / "run_log.csv")
lg["cutoff"] = lg["cutoff"].astype(str)


def order(methods):
    key = {m: i for i, m in enumerate([f"{m}-{s}" for s in ["direct", "recursive"] for m in NEURAL + VARIANTS] + CLASSICAL)}
    return sorted(methods, key=lambda m: key.get(m, 99))


# ---- Table: rolling h=1 (val and test) ------------------------------------
rows = []
for m in order(t1.method.unique()):
    v = t1[(t1.method == m) & (t1.period == "val")].iloc[0]
    t = t1[(t1.method == m) & (t1.period == "test")].iloc[0]
    if v.n_seeds > 1:
        rows.append(f"{label(m)} & {f(v.sMAPE_mean)} $\\pm$ {f(v.sMAPE_sd)} & {f(t.sMAPE_mean)} $\\pm$ {f(t.sMAPE_sd)} & "
                    f"[{f(t.sMAPE_ci_lo)}, {f(t.sMAPE_ci_hi)}] & {f(t.MAE_mean)} & {f(t.sMAPE_seed_avg_forecast)} \\\\")
    else:
        rows.append(f"{label(m)} & {f(v.sMAPE_mean)} & {f(t.sMAPE_mean)} & -- & {f(t.MAE_mean)} & -- \\\\")
    if m in ("patchtst_small-direct", "patchtst_small-recursive"):
        rows.append("\\addlinespace")
write("tab_rolling_h1.tex", "\n".join(rows) + "\n")

# ---- Table: horizon profile (mean over seeds) -----------------------------
main_methods = [f"{m}-{s}" for s in ["direct", "recursive"] for m in NEURAL] + CLASSICAL
p = t2.pivot_table(index="method", columns="h", values="sMAPE_mean")
moh = t2.groupby("method").sMAPE_mean_over_h.first()
rows = []
for m in main_methods:
    vals = p.loc[m]
    best = {h: p.loc[main_methods, h].min() for h in range(1, H + 1)}
    cells = [("\\textbf{" + f(vals[h], 1) + "}") if abs(vals[h] - best[h]) < 1e-9 else f(vals[h], 1) for h in range(1, H + 1)]
    rows.append(f"{label(m)} & " + " & ".join(cells) + f" & {f(moh[m], 1)} \\\\")
    if m in ("patchtst-direct", "patchtst-recursive"):
        rows.append("\\addlinespace")
write("tab_horizon_profile.tex", "\n".join(rows) + "\n")

# ---- Table: rank by horizon -------------------------------------------------
r = p.loc[main_methods].rank()
rows = []
for m in main_methods:
    rows.append(f"{label(m)} & " + " & ".join(str(int(r.loc[m, h])) for h in range(1, H + 1)) +
                f" & {int(r.loc[m].min())}--{int(r.loc[m].max())} \\\\")
    if m in ("patchtst-direct", "patchtst-recursive"):
        rows.append("\\addlinespace")
write("tab_rank_by_h.tex", "\n".join(rows) + "\n")

# ---- Table: GRU vs TCN (direct) uncertainty --------------------------------
rows = []
g = t2[(t2.method == "gru-direct")].set_index("h")
tc = t2[(t2.method == "tcn-direct")].set_index("h")
wp = t3[(t3.strategy == "direct") & (t3.model_A == "gru") & (t3.model_B == "tcn") & (t3.evaluation == "complete_test_n8")].set_index("h")
dm = t4[(t4.evaluation == "complete_test_n8") & (t4.method_A == "gru-direct") & (t4.method_B == "tcn-direct")].set_index("h")
for h in range(1, H + 1):
    rows.append(f"{h} & {f(g.loc[h,'sMAPE_mean'])} [{f(g.loc[h,'sMAPE_ci_lo'])}, {f(g.loc[h,'sMAPE_ci_hi'])}] & "
                f"{f(tc.loc[h,'sMAPE_mean'])} [{f(tc.loc[h,'sMAPE_ci_lo'])}, {f(tc.loc[h,'sMAPE_ci_hi'])}] & "
                f"{f(wp.loc[h,'P_A_better'])} & {f(dm.loc[h,'dm_stat'])} & {f(dm.loc[h,'p_value'],3)} \\\\")
write("tab_gru_tcn_se.tex", "\n".join(rows) + "\n")

# ---- Table: DM tests, rolling h=1 test, neural (direct) vs classical ---------
d1 = t4[t4.evaluation == "rolling_h1_test_n19"]


def dm_lookup(a, b):
    x = d1[(d1.method_A == a) & (d1.method_B == b)]
    if len(x):
        return x.dm_stat.iloc[0], x.p_value.iloc[0]
    x = d1[(d1.method_A == b) & (d1.method_B == a)]
    return -x.dm_stat.iloc[0], x.p_value.iloc[0]


rows = []
for m in [f"{m}-{s}" for s in ["direct", "recursive"] for m in NEURAL]:
    cells = []
    for c in CLASSICAL:
        st, pv = dm_lookup(m, c)
        cells.append(f"{f(st)} ({f(pv,3)})")
    rows.append(f"{label(m)} & " + " & ".join(cells) + " \\\\")
    if m == "patchtst-direct":
        rows.append("\\addlinespace")
write("tab_dm_classical_se.tex", "\n".join(rows) + "\n")

# ---- Table: bias ------------------------------------------------------------
rows = []
for m in [x for x in order(t5.method.unique()) if not x.startswith(("tcn24", "patchtst_small"))]:
    b = t5[t5.method == m].iloc[0]
    sd = "--" if pd.isna(b.bias_seed_sd_bnSAR) else f(b.bias_seed_sd_bnSAR)
    pv = "$<$0.001" if b.sign_test_p < 0.001 else f(b.sign_test_p, 3)
    rows.append(f"{label(m)} & {f(b.bias_seed_mean_bnSAR)} & {sd} & {f(b.MAE_seed_avg_forecast_bnSAR)} & "
                f"{int(b.n_over)}/{int(b.n)} & {pv} \\\\")
    if m == "patchtst-recursive":
        rows.append("\\addlinespace")
write("tab_bias.tex", "\n".join(rows) + "\n")

# ---- Table: anomaly sensitivity --------------------------------------------
sel = ["gru-direct", "tcn-direct", "lstm-direct", "gru-recursive", "tcn-recursive", "lstm-recursive",
       "naive", "snaive", "ets", "sarima"]
cols = ["none", "2024-03", "2025-03", "both_March", "all_flagged_test"]
rows = []
for m in sel:
    a = t8[t8.method == m].set_index("excluded")
    rows.append(f"{label(m)} & " + " & ".join(f(a.loc[c, 'rolling_h1_sMAPE_mean']) for c in cols) + " & " +
                " & ".join(f(a.loc[c, 'complete_mean_over_h_sMAPE']) for c in ["none", "both_March", "all_flagged_test"]) + " \\\\")
write("tab_anomaly_sensitivity.tex", "\n".join(rows) + "\n")

# ---- Table: backtest ---------------------------------------------------------
cut = sorted(t11.cutoff.unique())
rows = []
bt = t11[t11.method.isin(main_methods)]
rk_full = bt.pivot_table(index="method", columns="cutoff", values="full_sMAPE_mean").rank()
rk_h1 = bt.pivot_table(index="method", columns="cutoff", values="h1_sMAPE_mean").rank()
for m in main_methods:
    a = bt[bt.method == m].set_index("cutoff")
    rows.append(f"{label(m)} & " + " & ".join(f(a.loc[c, 'full_sMAPE_mean'], 1) for c in cut) +
                f" & {f(rk_h1.loc[m].mean(),1)} & {f(rk_full.loc[m].mean(),1)} \\\\")
    if m in ("patchtst-direct", "patchtst-recursive"):
        rows.append("\\addlinespace")
write("tab_backtest.tex", "\n".join(rows) + "\n")

# ---- Table: parameters -------------------------------------------------------
pp = (lg[lg.cutoff.str.startswith("2022-12")].groupby(["strategy", "model"])
      .agg(n_params=("n_params", "first"), n_win=("n_train_windows", "first"), secs=("train_seconds", "mean")).reset_index())
rows = []
for m in NEURAL + VARIANTS:
    d_ = pp[(pp.model == m) & (pp.strategy == "direct")].iloc[0]
    r_ = pp[(pp.model == m) & (pp.strategy == "recursive")].iloc[0]
    rows.append(f"{LABEL[m]} & {int(d_.n_params):,} & {int(r_.n_params):,} & {d_.n_params / d_.n_win:.0f} & "
                f"{f(d_.secs,1)} & {f(r_.secs,1)} \\\\".replace(",", "{,}"))
write("tab_parameters.tex", "\n".join(rows) + "\n")

# ---- variance decomposition: horizon vs target month -------------------------
t7 = pd.read_csv(RESULTS_DIR / "T7_error_by_target_month.csv")


def r2(df, cols):
    X = pd.get_dummies(df[cols].astype(str), drop_first=True).astype(float)
    X.insert(0, "c", 1.0)
    y = df.ape.values
    b = np.linalg.lstsq(X.values, y, rcond=None)[0]
    return 1 - (y - X.values @ b).var() / y.var()


rows, csv = [], []
for m in main_methods:
    gdf = t7[t7.method == m]
    a, b, c = r2(gdf, ["h"]), r2(gdf, ["target"]), r2(gdf, ["h", "target"])
    csv.append(dict(method=m, R2_h=a, R2_target=b, R2_both=c, unique_h=c - b, unique_target=c - a))
    rows.append(f"{label(m)} & {f(a)} & {f(b)} & {f(c - b)} & {f(c - a)} \\\\")
    if m in ("patchtst-direct", "patchtst-recursive"):
        rows.append("\\addlinespace")
pd.DataFrame(csv).to_csv(RESULTS_DIR / "T12_horizon_vs_target_month.csv", index=False)
write("tab_variance_decomp.tex", "\n".join(rows) + "\n")

# ===========================================================================
# Figures
# ===========================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator


def clean_log(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12})
COL = {"lstm": "#1f77b4", "gru": "#d62728", "tcn": "#2ca02c", "nbeats": "#9467bd", "patchtst": "#ff7f0e",
       "naive": "#555555", "snaive": "#000000", "ets": "#8c564b", "sarima": "#e377c2"}
LS = {"naive": ":", "snaive": "--", "ets": "-.", "sarima": (0, (5, 1, 1, 1))}

# Fig: horizon profiles, two panels
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, constrained_layout=True)
for ax, s in zip(axes, ["direct", "recursive"]):
    for m in NEURAL:
        gg = t2[t2.method == f"{m}-{s}"].sort_values("h")
        ax.plot(gg.h, gg.sMAPE_mean, marker="o", ms=4, color=COL[m], label=LABEL[m])
        ax.fill_between(gg.h, gg.sMAPE_ci_lo, gg.sMAPE_ci_hi, color=COL[m], alpha=0.15)
    for c in CLASSICAL:
        gg = t2[t2.method == c].sort_values("h")
        ax.plot(gg.h, gg.sMAPE_mean, color=COL[c], ls=LS[c], lw=1.8, label=LABEL[c])
    ax.set_title(f"{s.capitalize()} multi-step strategy")
    ax.set_xlabel("Forecast horizon $h$ (months)")
    ax.set_xticks(range(1, H + 1))
    ax.set_yscale("log")
    ax.set_yticks([3, 5, 10, 20, 40])
    ax.set_yticklabels(["3", "5", "10", "20", "40"])
    clean_log(ax)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Test sMAPE (%), log scale")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=9, frameon=False)
fig.savefig(OUT / "fig_horizon_profile.png", dpi=300)
plt.close(fig)

# Fig: rolling h=1 test with CI
tt = t1[(t1.period == "test") & t1.method.isin(main_methods)].copy()
tt = tt.sort_values("sMAPE_mean")
fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
y = np.arange(len(tt))
for i, (_, rr) in enumerate(tt.iterrows()):
    base = rr.method.split("-")[0]
    if rr.n_seeds > 1:
        ax.errorbar(rr.sMAPE_mean, i, xerr=[[rr.sMAPE_mean - rr.sMAPE_ci_lo], [rr.sMAPE_ci_hi - rr.sMAPE_mean]],
                    fmt="o" if rr.strategy == "direct" else "s", color=COL[base], capsize=4, ms=7)
    else:
        ax.plot(rr.sMAPE_mean, i, "D", color=COL[base], ms=7)
ax.set_yticks(y)
ax.set_yticklabels([label(m) for m in tt.method])
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xticks([4, 5, 7, 10, 15, 20, 30])
ax.set_xticklabels(["4", "5", "7", "10", "15", "20", "30"])
clean_log(ax, "x")
ax.set_xlabel("Rolling $h=1$ test sMAPE (%), mean and 95% CI over 20 seeds (log scale)")
ax.grid(axis="x", alpha=0.3)
fig.savefig(OUT / "fig_rolling_h1.png", dpi=300)
plt.close(fig)

# Fig: backtest
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
for ax, s in zip(axes, ["direct", "recursive"]):
    for m in NEURAL:
        a = t11[t11.method == f"{m}-{s}"].sort_values("cutoff")
        ax.errorbar(a.cutoff, a.full_sMAPE_mean, yerr=a.full_sMAPE_sd, marker="o", ms=5, capsize=3,
                    color=COL[m], label=LABEL[m])
    for c in CLASSICAL:
        a = t11[t11.method == c].sort_values("cutoff")
        ax.plot(a.cutoff, a.full_sMAPE_mean, color=COL[c], ls=LS[c], lw=1.8, label=LABEL[c])
    ax.set_title(f"{s.capitalize()} strategy")
    ax.set_xlabel("Training cutoff")
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80])
    ax.set_yticklabels(["5", "10", "20", "40", "80"])
    clean_log(ax)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Mean sMAPE over $h=1,\\ldots,12$ (%), log scale")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=9, frameon=False)
fig.savefig(OUT / "fig_backtest.png", dpi=300)
plt.close(fig)

# Fig: signed errors (seed-averaged forecasts, rolling h=1 test)
files = glob.glob(str(RUNS_DIR / "2022-12" / "*" / "*" / "seed*.csv"))
pr = pd.concat((pd.read_csv(x) for x in files), ignore_index=True)
pr["method"] = np.where(pr.strategy == "classical", pr.model, pr.model + "-" + pr.strategy)
h1 = pr[(pr.h == 1) & (pr.origin_period == "test")]
ens = h1.groupby(["method", "target"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
ens["err"] = (ens.yhat - ens.y) / 1e9
mm = [m for m in main_methods]
fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
data = [ens[ens.method == m].err.values for m in mm]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
for patch, m in zip(bp["boxes"], mm):
    patch.set_facecolor(COL[m.split("-")[0]])
    patch.set_alpha(0.45)
for i, d_ in enumerate(data, 1):
    ax.scatter(np.full(len(d_), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d_)), d_, s=12, color="k", zorder=3)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(1, len(mm) + 1))
short = {"Simplified patch Transformer": "Simplified\npatch\nTransformer", "Seasonal Naive": "Seasonal\nNaive"}
ax.set_xticklabels([short.get(label(m).split(" (")[0], label(m).split(" (")[0]) +
                    ("\n(" + m.split("-")[1] + ")" if "-" in m else "") for m in mm], fontsize=10)
ax.set_ylabel("Forecast $-$ actual (billion SAR)")
ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig_signed_errors.png", dpi=300)
plt.close(fig)

# Fig: anomaly timeline + training maximum
ser = load_series()
an = pd.read_csv(ANOMALY_FILE, parse_dates=["period_start"])
fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
ax.plot(ser.period_start, ser[TARGET] / 1e9, color="#1f77b4", lw=1.4, label="POS sales")
hm = an[an.anom_hampel == 1]
st = an[(an.anom_stl == 1) & (an.anom_hampel == 0)]
ax.scatter(st.period_start, st[TARGET] / 1e9, s=22, marker="x", color="#7f7f7f", label="STL residual flag only", zorder=3)
ax.scatter(hm.period_start, hm[TARGET] / 1e9, s=40, color="#d62728", label="Hampel/MAD flag", zorder=4)
ax.axvspan(pd.Timestamp(TRAIN_END) + pd.offsets.MonthBegin(1), pd.Timestamp(VAL_END), color="#ffdd99", alpha=0.5, label="Validation")
ax.axvspan(pd.Timestamp(VAL_END) + pd.offsets.MonthBegin(1), ser.period_start.max(), color="#b3d9ff", alpha=0.5, label="Test")
tm = ser.loc[ser.period_start <= TRAIN_END, TARGET].max() / 1e9
ax.axhline(tm, color="k", ls="--", lw=1, label=f"Training maximum ({tm:.1f})")
ax.set_ylabel("Monthly POS sales (billion SAR)")
ax.set_xlim(pd.Timestamp("2014-01-01"), ser.period_start.max() + pd.offsets.MonthBegin(1))
ax.legend(loc="upper left", ncol=2, fontsize=10)
ax.grid(alpha=0.3)
fig.savefig(OUT / "fig_anomaly_timeline.png", dpi=300)
plt.close(fig)

# Fig: rolling h=1 trajectories (seed mean and range) for GRU/TCN/LSTM direct + SARIMA
fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
sub = pr[(pr.h == 1) & (pr.origin_period.isin(["val", "test"]))]
act = sub.groupby("target").y.first() / 1e9
x = pd.to_datetime(act.index)
ax.plot(x, act.values, "k-o", ms=4, lw=2, label="Actual")
for m in ["gru-direct", "tcn-direct", "lstm-direct"]:
    q = sub[sub.method == m].groupby("target").yhat
    lo, mid, hi = q.quantile(0.05) / 1e9, q.mean() / 1e9, q.quantile(0.95) / 1e9
    c = COL[m.split("-")[0]]
    ax.plot(x, mid.values, color=c, lw=1.6, label=f"{label(m)}, seed mean")
    ax.fill_between(x, lo.values, hi.values, color=c, alpha=0.15)
s_ = sub[sub.method == "sarima"].groupby("target").yhat.first() / 1e9
ax.plot(x, s_.values, color=COL["sarima"], ls=LS["sarima"], lw=1.8, label="SARIMA")
ax.axvline(pd.Timestamp("2024-01-01"), color="grey", lw=1)
ax.axhline(tm, color="k", ls="--", lw=1)
ax.set_ylabel("Billion SAR")
ax.legend(ncol=3, fontsize=10, loc="upper left")
ax.grid(alpha=0.3)
fig.savefig(OUT / "fig_trajectories.png", dpi=300)
plt.close(fig)

print("assets written to", OUT)
