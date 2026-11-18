"""
15_dm_tests_smape.py — Diebold–Mariano tests with an sMAPE-consistent loss and multiplicity adjustment.

For every comparison reported in the manuscript, the loss differential is the difference in the symmetric absolute
percentage error, d_t = sAPE_A(t) - sAPE_B(t) with sAPE = 200|y - yhat| / (|y| + |yhat|), whose mean is the sMAPE
difference. The test is implemented in dmtest.py (exact formula there): truncated long-run variance with h-1
autocovariances, Harvey–Leybourne–Newbold factor with the actual forecast horizon h, t(n-1) reference. Tests with
h >= n (no defined HLN factor; horizons 8-12 on the n = 8 complete origins) and tests with a non-positive variance
estimate are reported as not available (no fallback) and are excluded from the Holm families. A Bartlett-weighted
(Newey–West) variance is computed for every test with h < n as a sensitivity analysis, and the procedure of the
previous package version (capped horizon in the HLN factor, gamma_0/n fallback) is recomputed for documentation.
Squared-error (SE) DM tests on the same pairs are reported for comparison. Holm adjustment is applied within each
family of tests listed below.

Families (seed-averaged forecasts):
  F0  main experiment, rolling h=1 test (n=19): direct GRU vs direct TCN                          (Section 6.1)
  F1  main experiment, rolling h=1 test (n=19): 10 configurations x 4 classical baselines       (Table 11)
  F2  main experiment, direct GRU vs direct TCN, complete-horizon origins (n=8), h=1..12          (Table 9)
  F3  main experiment, seed-averaged configurations vs ETS and SARIMA, h=1..12 (n=8)               (Section 6.2)
  F4  tuned configurations (Section 6.10), rolling h=1: 10 x 4                                      (Section 6.10)
  F5  second series, direct GRU vs Naive, ETS, SARIMA, h=1..12 (n=8)                                (Section 6.11)
  F6  corrected configurations (Section 6.13), rolling h=1: 14 x 4                                  (Table 24)
  F7  corrected configurations, h=1..12 (n=8): 14 x 4 x 12                                          (Section 6.13)
Output: results_r3/R4_dm_smape.csv (all tests, with status, sensitivity and legacy columns),
results_r3/r4_summary.json, results_r3/R4_dm_status_by_family_h.csv (availability by family and horizon), and the LaTeX tables
manuscript_assets/tab_gru_tcn.tex (Table 9) and tab_dm_classical.tex (Table 11).
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, H
from dmtest import dm_test, legacy_dm, holm

CLS = ["naive", "snaive", "ets", "sarima"]
MAIN = ["lstm", "gru", "tcn", "nbeats", "patchtst"]


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
    r = dict(family=fam, a=a, b=b, h=h, n=len(A), diff_smape=float((A.sape - B.sape).mean()))
    for k, d in [("sape", (A.sape - B.sape).values), ("se", (A.se - B.se).values)]:
        t = dm_test(d, h)
        ls, lp, lsub, lfb = legacy_dm(d, h)
        r.update({f"stat_{k}": t["stat"], f"p_{k}": t["p"], f"status_{k}": t["status"],
                  f"stat_{k}_bart": t["stat_bart"], f"p_{k}_bart": t["p_bart"],
                  f"stat_{k}_legacy": ls, f"p_{k}_legacy": lp, f"legacy_hsub_{k}": lsub, f"legacy_fallback_{k}": lfb})
    rows.append(r)


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
for f, g in R.groupby("family"):
    for c in ["p_sape", "p_se", "p_sape_bart", "p_se_bart", "p_sape_legacy", "p_se_legacy"]:
        R.loc[g.index, c + "_holm"] = holm(g[c].values)
(ROOT / "results_r3").mkdir(exist_ok=True)
R.to_csv(ROOT / "results_r3" / "R4_dm_smape.csv", index=False)


def count(fam, col, sign, alpha=0.05):
    g = R[R.family == fam]
    base = col.replace("_holm", "")
    s = base.replace("p_", "stat_", 1)
    return int(((g[col] < alpha) & (np.sign(g[s]) == sign)).sum())


summary = {"n_tests": R.groupby("family").size().to_dict(),
           "status_sape": R.groupby(["family", "status_sape"]).size().unstack(fill_value=0).to_dict("index"),
           "status_se": R.groupby(["family", "status_se"]).size().unstack(fill_value=0).to_dict("index"),
           "legacy_hsub_sape": R.groupby("family").legacy_hsub_sape.sum().astype(int).to_dict(),
           "legacy_fallback_sape": R.groupby("family").legacy_fallback_sape.sum().astype(int).to_dict()}
cols = ["p_sape", "p_sape_holm", "p_sape_bart", "p_sape_bart_holm", "p_sape_legacy", "p_sape_legacy_holm",
        "p_se", "p_se_holm", "p_se_bart", "p_se_bart_holm", "p_se_legacy", "p_se_legacy_holm"]
for f in sorted(R.family.unique()):
    summary[f] = {c: {"better": count(f, c, -1), "worse": count(f, c, 1)} for c in cols}
st = R.groupby(["family", "h"]).agg(tests=("status_sape", "size"),
                                    ok_sape=("status_sape", lambda x: int((x == "ok").sum())),
                                    na_horizon=("status_sape", lambda x: int((x == "na_horizon").sum())),
                                    na_variance_sape=("status_sape", lambda x: int((x == "na_variance").sum())),
                                    na_variance_se=("status_se", lambda x: int((x == "na_variance").sum())),
                                    legacy_hsub=("legacy_hsub_sape", "sum"),
                                    legacy_fallback_sape=("legacy_fallback_sape", "sum"),
                                    legacy_fallback_se=("legacy_fallback_se", "sum")).reset_index()
st.to_csv(ROOT / "results_r3" / "R4_dm_status_by_family_h.csv", index=False)
(ROOT / "results_r3" / "r4_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- LaTeX tables
LAB = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "nbeats": "MLP", "patchtst": "Simplified patch Transformer"}
f = lambda x, d=2: "n/a" if pd.isna(x) else f"{x:.{d}f}"
fp = lambda x: "n/a" if pd.isna(x) else ("$<$0.001" if x < 0.001 else f"{x:.3f}")
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

# ---------------------------------------------------------------- availability and sensitivity (Appendix)
FAMLAB = {"F0": "GRU vs TCN, $h=1$ (Sect.~\\ref{subsec:neural_results})",
          "F1": "Main experiment, $h=1$ (Table~\\ref{tab:dm_classical})",
          "F2": "GRU vs TCN by horizon (Table~\\ref{tab:gru_tcn})",
          "F3": "Main experiment by horizon (Sect.~\\ref{subsec:horizon_results})",
          "F4": "Tuned configurations, $h=1$ (Sect.~\\ref{subsec:hpo_results})",
          "F5": "Second series by horizon (Sect.~\\ref{subsec:second_series})",
          "F6": "Selected configurations, $h=1$ (Table~\\ref{tab:r3_tuned})",
          "F7": "Selected configurations by horizon (Sect.~\\ref{subsec:r3_results})"}
lines = []
for fam in ["F2", "F3", "F5", "F7"]:
    g = R[R.family == fam]
    cells = []
    for h in range(1, 8):
        x = g[g.h == h]
        cells.append(f"{int((x.status_sape == 'na_variance').sum())} ({int((x.status_se == 'na_variance').sum())})")
    nh = int((g.status_sape == "na_horizon").sum())
    ok = int((g.status_sape == "ok").sum())
    lines.append(f"{FAMLAB[fam]} & {len(g)} & {len(g[g.h == 1])} & " + " & ".join(cells) + f" & {nh} & {ok} \\\\")
(OUT / "tab_dm_availability.tex").write_text("\n".join(lines) + "\n")
lines = []
for fam in sorted(R.family.unique()):
    g = R[R.family == fam]
    def bw(col):
        return f"{count(fam, col, -1)} / {count(fam, col, 1)}"
    lines.append(f"{FAMLAB[fam]} & {len(g)} & {int((g.status_sape == 'ok').sum())} & {bw('p_sape')} & {bw('p_sape_holm')} & "
                 f"{bw('p_sape_bart_holm')} & {bw('p_sape_legacy_holm')} & {bw('p_se_holm')} \\\\")
(OUT / "tab_dm_sensitivity.tex").write_text("\n".join(lines) + "\n")
print("appendix tables written")
