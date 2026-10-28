"""
12_normalization_analysis.py — analysis, LaTeX tables and figures for the normalization experiments.
Run after run_normalization.py. Reads only stored forecasts (nothing is retrained).

Inputs
  runs/2022-12/...                 main experiment (global normalization, fixed 250 epochs; seeds 0-9 used here)
                                   and the classical forecasts
  runs_r3/<variant>/2022-12/...    normalization x early-stopping factorial and canonical models
  runs_r3/tuned_r3/2022-12/...     configurations selected by the search with the fixes (08_hpo.py --space r3)
Outputs
  results_r3/R3_factorial.csv      every configuration x variant: h=1 and complete-horizon test accuracy,
                                   validation sMAPE, early-stopping epoch, direction of h=1 errors
  results_r3/R3_tuned.csv          selected configurations + classical baselines, ranks, DM tests at h=1
  results_r3/R3_horizon_tuned.csv  test sMAPE by horizon (8 complete origins), ranks per horizon
  results_r3/R3_dm_horizon.csv     DM tests (HLN) of every selected configuration vs SARIMA and Seasonal Naive per h
  results_r3/R3_rank_agreement.csv Spearman correlation of configuration rankings between variants
  results_r3/r3_summary.json       numbers quoted in the manuscript
  manuscript_assets/r3_*.tex, r3_*.png
"""
import glob
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, H

spec = importlib.util.spec_from_file_location("an", ROOT / "05_analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
dm_hln = an.dm_hln


def dm_loss(d, h):
    """DM test (HLN correction, lag h-1 capped at n-2) on a given loss differential d."""
    d = np.asarray(d, float)
    n = len(d)
    lag = min(h - 1, n - 2)
    dc = d - d.mean()
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(lag + 1)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    var = var if var > 0 else gam[0] / n
    stat = d.mean() / np.sqrt(var)
    hh = lag + 1
    stat *= np.sqrt((n + 1 - 2 * hh + hh * (hh - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), n - 1)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m)
    out[o] = adj
    return out

RES, OUT = ROOT / "results_r3", ROOT / "manuscript_assets"
RES.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SEEDS = range(10)
MODELS = ["lstm", "gru", "tcn", "nbeats", "patchtst", "nbeats_c", "patchtst_c"]
STRATS = ["direct", "recursive"]
CLASSICAL = ["naive", "snaive", "ets", "sarima"]
VARIANTS = ["global_fixed", "global_es", "revin_fixed", "revin_es", "anchor_fixed", "anchor_es"]
VLAB = {"global_fixed": "Global, fixed", "global_es": "Global, ES", "revin_fixed": "RevIN, fixed",
        "revin_es": "RevIN, ES", "anchor_fixed": "Anchor, fixed", "anchor_es": "Anchor, ES"}
LAB = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "nbeats": "MLP",
       "patchtst": "Simplified patch Transformer", "nbeats_c": "N-BEATS (canonical)",
       "patchtst_c": "PatchTST (canonical)", "naive": "Naive", "snaive": "Seasonal Naive", "ets": "ETS",
       "sarima": "SARIMA"}
NLAB = {"global": "Global", "revin": "RevIN", "anchor": "Anchor"}
f = lambda x, d=2: "--" if x is None or pd.isna(x) else f"{x:.{d}f}"
fp = lambda p: "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def read_dir(pattern):
    files = glob.glob(str(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def read_logs(d):
    return read_dir(ROOT / d / "_logs" / "*.csv")


def load_variant(v):
    d = read_dir(ROOT / "runs_r3" / v / "2022-12" / "*" / "*" / "seed*.csv")
    if v == "global_fixed":
        old = read_dir(ROOT / "runs" / "2022-12" / "*" / "*" / "seed*.csv")
        old = old[(old.strategy != "classical") & old.seed.isin(SEEDS) & old.model.isin(MODELS)]
        d = pd.concat([old, d], ignore_index=True)
    if d.empty:
        return d
    d = d[d.seed.isin(SEEDS) & d.model.isin(MODELS)].copy()
    d["variant"] = v
    return d


def logs_variant(v):
    lg = read_logs(f"runs_r3/{v}")
    if v == "global_fixed":
        old = pd.read_csv(ROOT / "runs" / "run_log.csv") if (ROOT / "runs" / "run_log.csv").exists() else pd.DataFrame()
        if len(old):
            old = old[(old.get("cutoff", "2022-12").astype(str) == "2022-12") & old.seed.isin(SEEDS)]
            lg = pd.concat([lg, old], ignore_index=True)
    return lg


classical = read_dir(ROOT / "runs" / "2022-12" / "classical" / "*" / "seed*.csv")
test_all = classical[classical.origin_period == "test"]
cnt = test_all.groupby("origin").h.nunique()
COMPLETE = sorted(cnt[cnt == H].index)


def metrics(d):
    """d: forecasts of one configuration (all seeds). Returns a dict of test metrics."""
    t = d[d.origin_period == "test"]
    t1 = t[t.h == 1]
    tc = t[t.origin.isin(COMPLETE)]
    ape = lambda g: 200 * (g.yhat - g.y).abs() / (g.y.abs() + g.yhat.abs())
    s1 = t1.groupby("seed").apply(lambda g: ape(g).mean())
    sc = tc.groupby("seed").apply(lambda g: ape(g).mean())
    e1 = t1.groupby(["origin"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
    ec = tc.groupby(["origin", "h"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
    err1 = e1.yhat - e1.y
    over = int((err1 > 0).sum())
    n1 = len(e1)
    return dict(n_seeds=int(t.seed.nunique()), h1_mean=s1.mean(), h1_sd=s1.std(ddof=1) if len(s1) > 1 else np.nan,
                h1_ens=float(ape(e1).mean()), full_mean=sc.mean(), full_sd=sc.std(ddof=1) if len(sc) > 1 else np.nan,
                full_ens=float(ape(ec).mean()), mse_bn=float(err1.mean() / 1e9), mae_bn=float(err1.abs().mean() / 1e9),
                over=over, n_h1=n1, sign_p=float(stats.binomtest(over, n1, 0.5).pvalue) if n1 else np.nan,
                full_mse_bn=float((ec.yhat - ec.y).mean() / 1e9), full_over_share=float(((ec.yhat - ec.y) > 0).mean()))


# ======================================================================== factorial
rows = []
for v in VARIANTS:
    d = load_variant(v)
    if d.empty:
        continue
    lg = logs_variant(v)
    for (s, m), g in d.groupby(["strategy", "model"]):
        r = dict(variant=v, norm=v.split("_")[0], early_stopping=v.endswith("_es"), strategy=s, model=m)
        r.update(metrics(g))
        if len(lg) and "model" in lg:
            l = lg[(lg.strategy == s) & (lg.model == m)]
            r["val_sMAPE"] = l.val_sMAPE.mean() if "val_sMAPE" in l and len(l) else np.nan
            r["best_epoch_mean"] = l.best_epoch.mean() if "best_epoch" in l and len(l) else np.nan
            r["best_epoch_min"] = l.best_epoch.min() if "best_epoch" in l and len(l) else np.nan
            r["best_epoch_max"] = l.best_epoch.max() if "best_epoch" in l and len(l) else np.nan
        rows.append(r)
fac = pd.DataFrame(rows)
fac.to_csv(RES / "R3_factorial.csv", index=False)

cls_rows = []
for m in CLASSICAL:
    r = dict(model=m, strategy="classical")
    r.update(metrics(classical[classical.model == m]))
    cls_rows.append(r)
cls = pd.DataFrame(cls_rows)
CL = cls.set_index("model")
summary = {"classical": cls.set_index("model")[["h1_ens", "full_ens", "over", "mse_bn"]].round(3).to_dict("index")}

# rank agreement between variants (configurations present in both)
ra = []
for a in VARIANTS:
    for b in VARIANTS:
        if a >= b:
            continue
        x = fac[fac.variant == a].set_index(["strategy", "model"])
        y = fac[fac.variant == b].set_index(["strategy", "model"])
        idx = x.index.intersection(y.index)
        if len(idx) < 4:
            continue
        for col in ("h1_mean", "full_mean"):
            rho, p = stats.spearmanr(x.loc[idx, col], y.loc[idx, col])
            ra.append(dict(variant_a=a, variant_b=b, metric=col, n=len(idx), spearman=rho, p=p))
ra = pd.DataFrame(ra)
ra.to_csv(RES / "R3_rank_agreement.csv", index=False)

# ======================================================================== tuned with fixes
tuned = read_dir(ROOT / "runs_r3" / "tuned_r3" / "2022-12" / "*" / "*" / "seed*.csv")
best = json.loads((RES / "hpo_best_r3.json").read_text()) if (RES / "hpo_best_r3.json").exists() else {}
tl = read_logs("runs_r3/tuned_r3")
trows = []
if not tuned.empty:
    tuned = tuned[tuned.seed.isin(SEEDS)]
    cens = classical[(classical.origin_period == "test") & (classical.h == 1)].set_index(["model", "origin"])
    for (s, m), g in tuned.groupby(["strategy", "model"]):
        r = dict(strategy=s, model=m, method=f"{m}-{s}")
        r.update(metrics(g))
        b = best.get(f"{m}-{s}", {})
        r["norm"] = b.get("cfg", {}).get("norm", "global")
        r["val_sMAPE_sel"] = b.get("val_sMAPE", np.nan)
        r["val_sMAPE_trial0"] = b.get("default_val_sMAPE", np.nan)
        l = tl[(tl.strategy == s) & (tl.model == m)] if len(tl) else pd.DataFrame()
        r["best_epoch_mean"] = l.best_epoch.mean() if len(l) else np.nan
        r["n_params"] = int(l.n_params.iloc[0]) if len(l) else np.nan
        e = g[(g.origin_period == "test") & (g.h == 1)].groupby("origin").agg(y=("y", "first"), yhat=("yhat", "mean"))
        for c in CLASSICAL:
            ce = cens.loc[c].reindex(e.index)
            # sMAPE-consistent loss: difference in symmetric absolute percentage error
            sa = 200 * (e.yhat - e.y).abs() / (e.y.abs() + e.yhat.abs())
            sb = 200 * (ce.yhat - ce.y).abs() / (ce.y.abs() + ce.yhat.abs())
            stat, p = dm_loss((sa - sb).values, 1)
            r[f"dm_stat_{c}"], r[f"dm_p_{c}"] = stat, p
        trows.append(r)
tun = pd.DataFrame(trows)
if len(tun):
    pcols = [f"dm_p_{c}" for c in CLASSICAL]
    adj = holm(tun[pcols].values.ravel()).reshape(len(tun), len(CLASSICAL))
    for j, c in enumerate(CLASSICAL):
        tun[f"dm_p_holm_{c}"] = adj[:, j]
    allm = pd.concat([tun.assign(kind="neural"), cls.assign(kind="classical", method=cls.model)], ignore_index=True)
    allm["rank_h1"] = allm.h1_ens.rank(method="min").astype(int)
    allm["rank_full"] = allm.full_ens.rank(method="min").astype(int)
    allm.to_csv(RES / "R3_tuned.csv", index=False)

    # horizon profile (mean over seeds of per-seed sMAPE at each h; classical: single run)
    tc = pd.concat([tuned.assign(method=tuned.model + "-" + tuned.strategy),
                    classical.assign(method=classical.model)], ignore_index=True)
    tc = tc[(tc.origin_period == "test") & tc.origin.isin(COMPLETE)]
    tc["ape"] = 200 * (tc.yhat - tc.y).abs() / (tc.y.abs() + tc.yhat.abs())
    hp = tc.groupby(["method", "h", "seed"]).ape.mean().groupby(["method", "h"]).mean().unstack("h")
    hp.to_csv(RES / "R3_horizon_tuned.csv")
    hr = hp.rank(axis=0, method="min")
    best_by_h = {int(h): hp[h].idxmin() for h in hp.columns}

    # DM per horizon vs SARIMA and Seasonal Naive (seed-averaged forecasts, 8 origins)
    ens = tc.groupby(["method", "origin", "h"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
    ens["e"] = ens.yhat - ens.y
    dmr = []
    for m in tun.method:
        for c in ("sarima", "snaive", "ets", "naive"):
            for h in range(1, H + 1):
                a = ens[(ens.method == m) & (ens.h == h)].set_index("origin").e
                b = ens[(ens.method == c) & (ens.h == h)].set_index("origin").e.reindex(a.index)
                stat, p, lag = dm_hln(a.values, b.values, h)
                dmr.append(dict(method=m, baseline=c, h=h, stat=stat, p=p, lag=lag))
    dmr = pd.DataFrame(dmr)
    dmr.to_csv(RES / "R3_dm_horizon.csv", index=False)

# ======================================================================== stability of rankings across horizons
def kendall_w(M):
    """M: configurations x horizons (sMAPE). Kendall's coefficient of concordance of the rankings
    of the configurations over the horizons (1 = identical ranking at every horizon)."""
    R = M.rank(axis=0)
    n, m = R.shape
    S = ((R.sum(axis=1) - R.sum(axis=1).mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def horizon_matrix(d):
    t = d[(d.origin_period == "test") & d.origin.isin(COMPLETE)].copy()
    t["method"] = t.model + "-" + t.strategy
    t["ape"] = 200 * (t.yhat - t.y).abs() / (t.y.abs() + t.yhat.abs())
    return t.groupby(["method", "h", "seed"]).ape.mean().groupby(["method", "h"]).mean().unstack("h")


kw = []
for v in VARIANTS:
    d = load_variant(v)
    if d.empty:
        continue
    M = horizon_matrix(d)
    M5 = M.loc[[i for i in M.index if i.split("-")[0] in MODELS[:5]]]
    best_rank = M.rank(axis=0, method="min")
    kw.append(dict(set=v, n_configs=len(M), W_all=kendall_w(M), W_five=kendall_w(M5),
                   mean_rank_range=float((best_rank.max(axis=1) - best_rank.min(axis=1)).mean()),
                   n_distinct_best=int(M.idxmin(axis=0).nunique())))
if not tuned.empty:
    M = horizon_matrix(tuned)
    br = M.rank(axis=0, method="min")
    kw.append(dict(set="tuned_r3", n_configs=len(M), W_all=kendall_w(M),
                   W_five=kendall_w(M.loc[[i for i in M.index if i.split("-")[0] in MODELS[:5]]]),
                   mean_rank_range=float((br.max(axis=1) - br.min(axis=1)).mean()),
                   n_distinct_best=int(M.idxmin(axis=0).nunique())))
r2t = read_dir(ROOT / "runs_tuned" / "2022-12" / "*" / "*" / "seed*.csv")
if not r2t.empty:
    r2t = r2t[(r2t.strategy != "classical") & r2t.seed.isin(SEEDS)]
    M = horizon_matrix(r2t)
    br = M.rank(axis=0, method="min")
    kw.append(dict(set="tuned_r2", n_configs=len(M), W_all=kendall_w(M), W_five=kendall_w(M),
                   mean_rank_range=float((br.max(axis=1) - br.min(axis=1)).mean()),
                   n_distinct_best=int(M.idxmin(axis=0).nunique())))
kw = pd.DataFrame(kw)
kw.to_csv(RES / "R3_horizon_concordance.csv", index=False)

# ======================================================================== summary numbers
def fac_get(v, s, m, col):
    x = fac[(fac.variant == v) & (fac.strategy == s) & (fac.model == m)]
    return float(x[col].iloc[0]) if len(x) else np.nan


for v in VARIANTS:
    sub = fac[fac.variant == v]
    if sub.empty:
        continue
    one_sided = sub[(sub.over == 0) | (sub.over == sub.n_h1)]
    summary[v] = dict(
        n_configs=int(len(sub)),
        n_all_same_sign=int(len(one_sided)),
        all_same_sign=[f"{m}-{s}" for s, m in zip(one_sided.strategy, one_sided.model)],
        n_sign_sig=int((sub.sign_p < 0.05).sum()),
        best_h1=sub.sort_values("h1_mean").iloc[0][["model", "strategy", "h1_mean"]].to_dict(),
        best_full=sub.sort_values("full_mean").iloc[0][["model", "strategy", "full_mean"]].to_dict(),
        n_beat_sarima_h1=int((sub.h1_ens < CL.loc["sarima", "h1_ens"]).sum()),
        n_beat_bestcls_full=int((sub.full_ens < cls.full_ens.min()).sum()),
        median_h1=float(sub.h1_mean.median()), median_full=float(sub.full_mean.median()),
        max_h1=float(sub.h1_mean.max()), max_full=float(sub.full_mean.max()),
        mean_best_epoch=float(sub.best_epoch_mean.mean()) if "best_epoch_mean" in sub and v.endswith("_es") else None)
if len(tun):
    summary["tuned_r3"] = dict(
        rows=allm.set_index("method")[["h1_mean", "h1_sd", "h1_ens", "full_mean", "full_ens", "over", "sign_p",
                                       "rank_h1", "rank_full", "norm", "dm_p_sarima", "dm_p_ets", "dm_p_snaive",
                                       "dm_p_naive", "best_epoch_mean"]]
        .round(4).replace({np.nan: None}).to_dict("index"),
        n_dm_sig_better_h1={c: int(((tun[f"dm_p_{c}"] < 0.05) & (tun[f"dm_stat_{c}"] < 0)).sum()) for c in CLASSICAL},
        n_dm_sig_worse_h1={c: int(((tun[f"dm_p_{c}"] < 0.05) & (tun[f"dm_stat_{c}"] > 0)).sum()) for c in CLASSICAL},
        dm_horizon_sig_better=dmr[(dmr.p < 0.05) & (dmr.stat < 0)][["method", "baseline", "h"]].values.tolist(),
        dm_horizon_sig_worse=int(((dmr.p < 0.05) & (dmr.stat > 0)).sum()),
        dm_horizon_tests=int(dmr.p.notna().sum()),
        best_by_h=best_by_h,
        norm_selected={k: v.get("cfg", {}).get("norm") for k, v in best.items()},
        n_all_same_sign=int(((tun.over == 0) | (tun.over == tun.n_h1)).sum()),
        all_same_sign=[m for m, o, n in zip(tun.method, tun.over, tun.n_h1) if o in (0, n)],
        rank_by_h={m: hr.loc[m].astype(int).tolist() for m in hr.index},
        spearman_val_test=float(stats.spearmanr(tun.val_sMAPE_sel, tun.full_mean)[0]) if tun.val_sMAPE_sel.notna().sum() > 3 else None)
summary["rank_agreement"] = ra.round(3).to_dict("records") if len(ra) else []
summary["horizon_concordance"] = kw.round(3).to_dict("records")
(RES / "r3_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

# ======================================================================== LaTeX tables
def write(name, s):
    (OUT / name).write_text(s)


def panel(col, sdcol):
    lines = []
    for s in STRATS:
        for m in MODELS:
            vals = [fac_get(v, s, m, col) for v in VARIANTS]
            if all(pd.isna(vals)):
                continue
            mn = np.nanmin(vals)
            cells = ["--" if pd.isna(x) else (f"\\textbf{{{x:.2f}}}" if x == mn else f"{x:.2f}") for x in vals]
            lines.append(f"{LAB[m]} ({s}) & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    return lines


hdr = " & ".join(VLAB[v] for v in VARIANTS)
tab = ["\\begin{table}[htbp]", "\\centering",
       "\\caption{Test sMAPE (\\%) of the default architectures under three input normalizations and two training "
       "regimes (fixed 250 epochs; early stopping (ES) on the validation loss, patience 25, best weights restored); "
       "mean over 10 seeds. Global: train-only $\\log$ and $z$-score encoding (main experiment); RevIN: reversible "
       "instance normalization per input window; Anchor: inputs and forecasts relative to the last observed value. "
       "The canonical PatchTST contains RevIN and is therefore reported only under Global. Bold: lowest value in the row. "
       "Classical baselines, Panel A: SARIMA " + f(CL.loc['sarima', 'h1_ens']) + ", ETS " + f(CL.loc['ets', 'h1_ens'])
       + ", Naive " + f(CL.loc['naive', 'h1_ens']) + ", Seasonal Naive " + f(CL.loc['snaive', 'h1_ens'])
       + "; Panel B: SARIMA " + f(CL.loc['sarima', 'full_ens']) + ", ETS " + f(CL.loc['ets', 'full_ens'])
       + ", Naive " + f(CL.loc['naive', 'full_ens']) + ", Seasonal Naive " + f(CL.loc['snaive', 'full_ens']) + ".}",
       "\\label{tab:r3_factorial}", "\\vspace{4pt}", "\\small", "\\begin{adjustbox}{max width=\\textwidth}",
       "\\begin{tabular}{lcccccc}", "\\toprule", "\\textbf{Configuration} & " + hdr + " \\\\", "\\midrule",
       "\\multicolumn{7}{l}{\\textit{Panel A: rolling $h=1$ test sample ($n=19$)}} \\\\", "\\addlinespace"]
tab += panel("h1_mean", "h1_sd")
tab += ["\\midrule", "\\multicolumn{7}{l}{\\textit{Panel B: complete-horizon test sample (8 origins, $h=1,\\dots,12$)}} \\\\",
        "\\addlinespace"]
tab += panel("full_mean", "full_sd")
tab += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
write("r3_factorial.tex", "\n".join(tab) + "\n")

# direction of errors by variant
lines = []
for v in VARIANTS:
    sub = fac[fac.variant == v]
    if sub.empty:
        continue
    one = sub[(sub.over == 0) | (sub.over == sub.n_h1)]
    names = ", ".join(f"{LAB[m].split(' (')[0] if m in ('nbeats_c', 'patchtst_c') else LAB[m]}{' (c)' if m.endswith('_c') else ''}-{s[0]}"
                      for s, m in zip(one.strategy, one.model)) or "none"
    lines.append(f"{VLAB[v]} & {len(sub)} & {len(one)} & {(sub.sign_p < 0.05).sum()} & "
                 f"{f(sub.mae_bn.median())} & {f((sub.mse_bn.abs() / sub.mae_bn).median())} & {names} \\\\")
tab = ["\\begin{table}[htbp]", "\\centering",
       "\\caption{Direction of rolling $h=1$ test errors (seed-averaged forecasts, $n=19$) under each normalization "
       "and training regime. One-sided: configurations whose 19 errors all have the same sign; sign test: two-sided "
       "$p<0.05$; $|$Bias$|$/MAE: median ratio of the absolute mean signed error to the MAE (1 = every error has the "
       "same sign). Suffixes -d and -r denote the direct and recursive strategies; (c) the canonical implementations.}",
       "\\label{tab:r3_bias}", "\\vspace{4pt}", "\\small", "\\begin{adjustbox}{max width=\\textwidth}",
       "\\begin{tabular}{lccccc p{6.2cm}}", "\\toprule",
       "\\textbf{Variant} & \\textbf{Configs} & \\textbf{One-sided} & \\textbf{Sign test $p<0.05$} & "
       "\\textbf{Median MAE} & \\textbf{Median $|$Bias$|$/MAE} & \\textbf{One-sided configurations} \\\\", "\\midrule"]
tab += lines + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
write("r3_bias.tex", "\n".join(tab) + "\n")

if len(tun):
    lines = []
    for s in STRATS:
        for m in MODELS:
            x = allm[allm.method == f"{m}-{s}"]
            if x.empty:
                continue
            x = x.iloc[0]
            lines.append(f"{LAB[m]} ({s}) & {NLAB.get(x.norm, x.norm)} & {f(x.best_epoch_mean, 0)} & {f(x.val_sMAPE_sel)} & "
                         f"{f(x.h1_mean)} $\\pm$ {f(x.h1_sd)} & {f(x.h1_ens)} & {f(x.full_mean)} $\\pm$ {f(x.full_sd)} & {f(x.full_ens)} & "
                         f"{x.rank_h1} / {x.rank_full} & {int(x.over)}/{int(x.n_h1)} & {fp(x.dm_p_sarima)} ({fp(x.dm_p_holm_sarima)}) & {fp(x.dm_p_snaive)} ({fp(x.dm_p_holm_snaive)}) \\\\")
        lines.append("\\addlinespace")
    for c in CLASSICAL:
        x = allm[allm.method == c].iloc[0]
        lines.append(f"{LAB[c]} & -- & -- & -- & {f(x.h1_ens)} & {f(x.h1_ens)} & {f(x.full_ens)} & {f(x.full_ens)} & {x.rank_h1} / {x.rank_full} & "
                     f"{int(x.over)}/{int(x.n_h1)} & -- & -- \\\\")
    tab = ["\\begin{table}[htbp]", "\\centering",
           "\\caption{Configurations selected by the equal-budget search with the corrections (20 configurations "
           "$\\times$ 3 seeds per architecture and strategy; validation early stopping; normalization searched; "
           "selection on validation sMAPE only), evaluated with 10 new seeds. Epoch: mean early-stopping epoch; "
           "Mean: test sMAPE (\\%) of single runs, mean $\\pm$ SD over seeds; Avg.: test sMAPE (\\%) of the "
           "seed-averaged forecast (classical baselines: single deterministic forecast in both columns); rank among the 14 neural configurations and 4 classical "
           "baselines (seed-averaged forecasts, $h=1$ / complete horizon); Over: months overpredicted by the "
           "seed-averaged $h=1$ forecast; DM: two-sided HLN-corrected Diebold--Mariano $p$-value at $h=1$ "
           "(loss: symmetric absolute percentage error; seed-averaged forecasts) against SARIMA and Seasonal Naive, "
           "with the Holm-adjusted value over the 56 tests of the 14 configurations against the four classical "
           "baselines in parentheses.}",
           "\\label{tab:r3_tuned}", "\\vspace{4pt}", "\\small", "\\begin{adjustbox}{max width=\\textwidth}",
           "\\begin{tabular}{llcccccccccc}", "\\toprule",
           "& & & & \\multicolumn{2}{c}{\\textbf{Test $h=1$}} & \\multicolumn{2}{c}{\\textbf{Test, all $h$}} & & & & \\\\",
           "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
           "\\textbf{Configuration} & \\textbf{Norm.} & \\textbf{Epoch} & \\textbf{Val.} & \\textbf{Mean} & \\textbf{Avg.} & "
           "\\textbf{Mean} & \\textbf{Avg.} & \\textbf{Rank} & \\textbf{Over} & \\textbf{DM SARIMA} & \\textbf{DM SNaive} \\\\",
           "\\midrule"]
    tab += lines + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
    write("r3_tuned.tex", "\n".join(tab) + "\n")

# ======================================================================== figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

if len(tun):
    order = [f"{m}-{s}" for s in STRATS for m in MODELS if f"{m}-{s}" in hp.index] + CLASSICAL
    M = hp.loc[order]
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(M.values, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanpercentile(M.values, 95))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            rank = int(hr.loc[order[i]].iloc[j])
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanpercentile(M.values, 60) else "#222222",
                    fontweight="bold" if rank == 1 else "normal")
    lab = [f"{LAB[o.split('-')[0]]} ({o.split('-')[1][0]})" if "-" in o else LAB[o] for o in order]
    ax.set_yticks(range(len(order)), lab)
    ax.set_xticks(range(H), [str(h) for h in range(1, H + 1)])
    ax.set_xlabel("Forecast horizon h (months)")
    ax.axhline(len(order) - len(CLASSICAL) - 0.5, color="#333333", lw=0.8)
    ax.axhline(sum(o.endswith("-direct") for o in order) - 0.5, color="#999999", lw=0.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Test sMAPE (%)")
    ax.set_title("Selected configurations with corrections: test sMAPE by horizon (bold = lowest at h)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "r3_horizon_heatmap.png", dpi=200)
    plt.close(fig)

    # signed h=1 errors of the selected configurations
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    t1 = tuned[(tuned.origin_period == "test") & (tuned.h == 1)]
    e = t1.groupby(["model", "strategy", "origin"]).agg(y=("y", "first"), yhat=("yhat", "mean")).reset_index()
    e["err"] = (e.yhat - e.y) / 1e9
    c1 = classical[(classical.origin_period == "test") & (classical.h == 1)].copy()
    c1["err"] = (c1.yhat - c1.y) / 1e9
    groups = [(f"{LAB[m]} ({s[0]})", e[(e.model == m) & (e.strategy == s)].err.values)
              for s in STRATS for m in MODELS if len(e[(e.model == m) & (e.strategy == s)])]
    groups += [(LAB[c], c1[c1.model == c].err.values) for c in CLASSICAL]
    rng = np.random.default_rng(0)
    for i, (nm, v) in enumerate(groups):
        col = "#1f5fa8" if i < len(groups) - len(CLASSICAL) else "#777777"
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=col, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="#222222", lw=1.2)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Forecast minus actual (billion SAR)")
    ax.set_title("Signed rolling h=1 test errors, selected configurations with corrections (bar = mean)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "r3_signed_errors.png", dpi=200)
    plt.close(fig)

print(json.dumps({k: v for k, v in summary.items() if k in VARIANTS}, indent=1, default=str)[:4000])
if len(tun):
    print(allm[["method", "norm", "best_epoch_mean", "val_sMAPE_sel", "h1_mean", "h1_ens", "full_mean", "full_ens",
                "rank_h1", "rank_full", "over", "dm_p_sarima", "dm_p_snaive"]].round(2).to_string(index=False))
