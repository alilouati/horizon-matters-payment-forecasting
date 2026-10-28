"""
15_dm_tests_smape.py — Diebold–Mariano tests with an sMAPE-consistent loss and multiplicity adjustment.

For every comparison reported in the manuscript, the loss differential is the difference in the symmetric absolute
percentage error, d_t = sAPE_A(t) - sAPE_B(t) with sAPE = 200|y - yhat| / (|y| + |yhat|), whose mean is the sMAPE
difference. The DM statistic uses the long-run variance with h-1 autocovariances (capped at n-2) and the
Harvey–Leybourne–Newbold correction, compared with t(n-1). Squared-error (SE) DM tests on the same pairs are reported
for comparison. Holm adjustment is applied within each family of tests listed below.

Families (seed-averaged forecasts):
  F0  main experiment, rolling h=1 test (n=19): direct GRU vs direct TCN                          (Section 6.1)
  F1  main experiment, rolling h=1 test (n=19): 10 configurations x 4 classical baselines       (Table 11)
  F2  main experiment, direct GRU vs direct TCN, complete-horizon origins (n=8), h=1..12          (Table 9)
  F3  main experiment, seed-averaged configurations vs ETS and SARIMA, h=1..12 (n=8)               (Section 6.2)
  F4  tuned configurations (Section 6.10), rolling h=1: 10 x 4                                      (Section 6.10)
  F5  second series, direct GRU vs Naive, ETS, SARIMA, h=1..12 (n=8)                                (Section 6.11)
  F6  corrected configurations (Section 6.13), rolling h=1: 14 x 4                                  (Table 24)
  F7  corrected configurations, h=1..12 (n=8): 14 x 4 x 12                                          (Section 6.13)
Output: results_r3/R4_dm_smape.csv (all tests), results_r3/r4_summary.json, and the LaTeX tables
manuscript_assets/tab_gru_tcn.tex (Table 9) and tab_dm_classical.tex (Table 11).
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, H

CLS = ["naive", "snaive", "ets", "sarima"]
MAIN = ["lstm", "gru", "tcn", "nbeats", "patchtst"]


def dm(d, h):
    d = np.asarray(d, float)
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return np.nan, np.nan
    lag = min(h - 1, n - 2)
    dc = d - d.mean()
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(lag + 1)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        var = gam[0] / n
    stat = d.mean() / np.sqrt(var)
    hh = lag + 1
    corr = (n + 1 - 2 * hh + hh * (hh - 1) / n) / n
    if corr <= 0:
        return stat, np.nan
    stat *= np.sqrt(corr)
    return stat, 2 * stats.t.sf(abs(stat), n - 1)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def rd(pattern, seeds=None):
    fs = glob.glob(str(pattern))
    d = pd.concat((pd.read_csv(f) for f in fs), ignore_index=True)
    if seeds is not None:
        d = d[(d.strategy == "classical") | d.seed.isin(seeds)]
    d["method"] = np.where(d.strategy == "classical", d.model, d.model + "-" + d.strategy)
    return d


def ensemble(d):
    t = d[d.origin_period == "test"]
    e = t.groupby(["method", "origin", "h"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
    e["sape"] = 200 * (e.yhat - e.y).abs() / (e.y.abs() + e.yhat.abs())
    e["se"] = (e.yhat - e.y) ** 2
    cnt = t[t.strategy == "classical"].groupby("origin").h.nunique()
    return e.set_index(["method", "origin", "h"]), sorted(cnt[cnt == H].index)


rows = []


def test(fam, E, a, b, h, origins=None):
    A, B = E.loc[a], E.loc[b]
    A = A.xs(h, level="h")
    B = B.xs(h, level="h")
    if origins is not None:
        A, B = A.loc[origins], B.loc[origins]
    B = B.reindex(A.index)
    s1, p1 = dm(A.sape - B.sape, h)
    s2, p2 = dm(A.se - B.se, h)
    rows.append(dict(family=fam, a=a, b=b, h=h, n=len(A), diff_smape=float((A.sape - B.sape).mean()),
                     stat_sape=s1, p_sape=p1, stat_se=s2, p_se=p2))


W = ROOT
# main experiment (20 seeds) + classical
main = rd(W / "runs" / "2022-12" / "*" / "*" / "seed*.csv")
main = main[main.model.isin(MAIN + CLS + ["tcn24"])]
E, comp = ensemble(main)
cfg = [f"{m}-{s}" for s in ["direct", "recursive"] for m in MAIN]
test("F0", E, "gru-direct", "tcn-direct", 1)
for a in cfg:
    for b in CLS:
        test("F1", E, a, b, 1)
for h in range(1, H + 1):
    test("F2", E, "gru-direct", "tcn-direct", h, comp)
for a in cfg + ["tcn24-recursive"]:
    for b in ["ets", "sarima"]:
        for h in range(1, H + 1):
            test("F3", E, a, b, h, comp)
# tuned (Section 6.10)
tun = rd(W / "runs_tuned" / "2022-12" / "*" / "*" / "seed*.csv")
tun = pd.concat([tun[tun.strategy != "classical"], main[main.strategy == "classical"]], ignore_index=True)
Et, _ = ensemble(tun)
for a in cfg:
    for b in CLS:
        test("F4", Et, a, b, 1)
# second series
tx = rd(W / "runs_tx" / "2022-12" / "*" / "*" / "seed*.csv")
Ex, compx = ensemble(tx)
for b in ["naive", "ets", "sarima"]:
    for h in range(1, H + 1):
        test("F5", Ex, "gru-direct", b, h, compx)
# corrected configurations (Section 6.13)
r3 = rd(W / "runs_r3" / "tuned_r3" / "2022-12" / "*" / "*" / "seed*.csv", seeds=range(10))
r3 = pd.concat([r3, main[main.strategy == "classical"]], ignore_index=True)
Er, compr = ensemble(r3)
cfg3 = sorted(r3[r3.strategy != "classical"].method.unique())
for a in cfg3:
    for b in CLS:
        test("F6", Er, a, b, 1)
for a in cfg3:
    for b in CLS:
        for h in range(1, H + 1):
            test("F7", Er, a, b, h, compr)

R = pd.DataFrame(rows)
R["p_sape_holm"] = np.nan
R["p_se_holm"] = np.nan
for f, g in R.groupby("family"):
    R.loc[g.index, "p_sape_holm"] = holm(g.p_sape.values)
    R.loc[g.index, "p_se_holm"] = holm(g.p_se.values)
(ROOT / "results_r3").mkdir(exist_ok=True)
R.to_csv(ROOT / "results_r3" / "R4_dm_smape.csv", index=False)


def count(fam, col, sign, alpha=0.05):
    g = R[R.family == fam]
    s = "stat_sape" if "sape" in col else "stat_se"
    return int(((g[col] < alpha) & (np.sign(g[s]) == sign)).sum())


summary = {"n_tests": R.groupby("family").size().to_dict()}
for f in sorted(R.family.unique()):
    summary[f] = {c: {"better": count(f, c, -1), "worse": count(f, c, 1)} for c in ["p_sape", "p_sape_holm", "p_se", "p_se_holm"]}
(ROOT / "results_r3" / "r4_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- LaTeX tables
LAB = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "nbeats": "MLP", "patchtst": "Simplified patch Transformer"}
f = lambda x, d=2: "--" if pd.isna(x) else f"{x:.{d}f}"
fp = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
OUT = ROOT / "manuscript_assets"
OUT.mkdir(exist_ok=True)
t2 = pd.read_csv(ROOT / "results" / "T2_horizon_profile.csv")
t3 = pd.read_csv(ROOT / "results" / "T3_win_probability.csv")
g = t2[t2.method == "gru-direct"].set_index("h")
tc = t2[t2.method == "tcn-direct"].set_index("h")
wp = t3[(t3.strategy == "direct") & (t3.model_A == "gru") & (t3.model_B == "tcn") & (t3.evaluation == "complete_test_n8")].set_index("h")
F2 = R[R.family == "F2"].set_index("h")
lines = []
for h in range(1, H + 1):
    lines.append(f"{h} & {f(g.loc[h, 'sMAPE_mean'])} [{f(g.loc[h, 'sMAPE_ci_lo'])}, {f(g.loc[h, 'sMAPE_ci_hi'])}] & "
                 f"{f(tc.loc[h, 'sMAPE_mean'])} [{f(tc.loc[h, 'sMAPE_ci_lo'])}, {f(tc.loc[h, 'sMAPE_ci_hi'])}] & "
                 f"{f(wp.loc[h, 'P_A_better'])} & {f(F2.loc[h, 'stat_sape'])} & {fp(F2.loc[h, 'p_sape'])} & "
                 f"{fp(F2.loc[h, 'p_sape_holm'])} \\\\")
(OUT / "tab_gru_tcn.tex").write_text("\n".join(lines) + "\n")
F1 = R[R.family == "F1"].set_index(["a", "b"])
lines = []
for m in cfg:
    cells = [f"{f(F1.loc[(m, c), 'stat_sape'])} ({fp(F1.loc[(m, c), 'p_sape'])}; {fp(F1.loc[(m, c), 'p_sape_holm'])})" for c in CLS]
    a, st = m.split("-")
    lines.append(f"{LAB[a]} ({st}) & " + " & ".join(cells) + " \\\\")
    if m == "patchtst-direct":
        lines.append("\\addlinespace")
(OUT / "tab_dm_classical.tex").write_text("\n".join(lines) + "\n")
print("tables written")
