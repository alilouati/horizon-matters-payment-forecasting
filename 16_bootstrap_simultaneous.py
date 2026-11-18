"""
16_bootstrap_simultaneous.py — traceable simultaneous (Bonferroni-adjusted) bootstrap intervals and the
origin-level loss differences behind every bootstrap statement of the manuscript.

All intervals are percentile intervals of the circular moving-block bootstrap of the mean difference in symmetric
absolute percentage error (sAPE) between two seed-averaged forecasts. They are conditional on the stored trained
forecasts and on the observed test period: they describe the variability of the mean difference over the forecast
origins of that period, not the variability from retraining on other data.

Samples (endpoints):
  h1    rolling one-step test sample: 19 monthly differences, sAPE_A - sAPE_B at h = 1
  full  complete-horizon sample: 8 forecast origins, per-origin mean over h = 1..12 of sAPE_A - sAPE_B

Families and adjustment (alpha = 0.05):
  T10   the 11 comparisons of Table 10; one family of 11 per endpoint (alpha/11 per interval); a joint family
        over both endpoints (alpha/22) is reported as sensitivity
  S613  Section 6.13: the 14 selected configurations x 4 baselines = 56 comparisons per endpoint (alpha/56), and the
        12 RevIN + early-stopping factorial configurations x 4 = 48 (alpha/48)

Block lengths: 3 months (h1) and 2 origins (full) as in the primary analysis (10_bootstrap.py,
14_normalization_bootstrap.py); sensitivity with block lengths 2, 3 and 4 for both endpoints.
Resamples: 100,000 per interval, vectorised; the resampling indices depend only on (n, block, seed), so all
comparisons of an endpoint share the same resamples.

Outputs (results_r3/):
  R5_origin_differences.csv      every origin-level difference used (set, A, B, endpoint, origin, d)
  R5_bootstrap_simultaneous.csv  point estimate, unadjusted 95% and simultaneous intervals, by block length,
                                 plus the number of origins with a negative difference
  r5_bootstrap_summary.json      counts used in the text
"""
import glob
import json

import numpy as np
import pandas as pd

from common import ROOT, H

REPS = 100_000
PAIRS10 = [("gru-direct", "tcn-direct"), ("gru-direct", "lstm-direct"), ("lstm-recursive", "gru-direct"),
           ("gru-direct", "sarima"), ("gru-direct", "naive"), ("tcn-direct", "sarima"), ("lstm-recursive", "sarima"),
           ("lstm-recursive", "naive"), ("lstm-direct", "ets"), ("gru-direct", "ets"), ("tcn-direct", "ets")]
CLS = ["naive", "snaive", "ets", "sarima"]
PRIMARY_BLOCK = {"h1": 3, "full": 2}


def rd(pattern):
    fs = glob.glob(str(pattern))
    d = pd.concat((pd.read_csv(f) for f in fs), ignore_index=True)
    d["method"] = np.where(d.strategy == "classical", d.model, d.model + "-" + d.strategy)
    return d


def ens(d, seeds=None):
    if seeds is not None:
        d = d[(d.strategy == "classical") | d.seed.isin(seeds)]
    t = d[d.origin_period == "test"]
    e = t.groupby(["method", "origin", "h"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
    e["sape"] = 200 * (e.yhat - e.y).abs() / (e.y.abs() + e.yhat.abs())
    cnt = t[t.strategy == "classical"].groupby("origin").h.nunique()
    return e.set_index(["method", "origin", "h"]).sape, sorted(cnt[cnt == H].index)


_IDX = {}


def idx(n, block, seed=0):
    key = (n, block, seed)
    if key not in _IDX:
        rng = np.random.default_rng(seed)
        k = int(np.ceil(n / block))
        starts = rng.integers(0, n, (REPS, k))
        _IDX[key] = ((starts[:, :, None] + np.arange(block)) % n).reshape(REPS, -1)[:, :n]
    return _IDX[key]


def interval(d, block, alpha):
    m = d[idx(len(d), block)].mean(1)
    return np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def diffs(S, comp, a, b):
    x = (S.loc[a] - S.loc[b]).rename("d").reset_index()
    d1 = x[x.h == 1].sort_values("origin")[["origin", "d"]]
    dc = x[x.origin.isin(comp)].groupby("origin", as_index=False).d.mean().sort_values("origin")
    return {"h1": d1, "full": dc}


rows, orows = [], []


def run(setname, S, comp, pairs, m_family):
    for a, b in pairs:
        for ep, dd in diffs(S, comp, a, b).items():
            for o, v in zip(dd.origin, dd.d):
                orows.append(dict(set=setname, a=a, b=b, endpoint=ep, origin=o, d=v))
            d = dd.d.values
            r = dict(set=setname, a=a, b=b, endpoint=ep, n=len(d), mean_diff=d.mean(), n_negative=int((d < 0).sum()),
                     family_size=m_family, primary_block=PRIMARY_BLOCK[ep])
            for blk in (2, 3, 4):
                lo, hi = interval(d, blk, 0.05)
                slo, shi = interval(d, blk, 0.05 / m_family)
                r.update({f"lo_95_b{blk}": lo, f"hi_95_b{blk}": hi, f"lo_sim_b{blk}": slo, f"hi_sim_b{blk}": shi})
                if setname == "T10":
                    jlo, jhi = interval(d, blk, 0.05 / (2 * m_family))
                    r.update({f"lo_joint_b{blk}": jlo, f"hi_joint_b{blk}": jhi})
            rows.append(r)


main = rd(ROOT / "runs" / "2022-12" / "*" / "*" / "seed*.csv")
S, comp = ens(main)
run("T10", S, comp, PAIRS10, len(PAIRS10))
cls = main[main.strategy == "classical"]
for setname in ["tuned_r3", "revin_es"]:
    nn = rd(ROOT / "runs_r3" / setname / "2022-12" / "*" / "*" / "seed*.csv")
    S3, comp3 = ens(pd.concat([nn, cls], ignore_index=True), seeds=range(10))
    cfg = sorted(nn.method.unique())
    run(setname, S3, comp3, [(a, b) for a in cfg for b in CLS], 4 * len(cfg))

B = pd.DataFrame(rows)
O = pd.DataFrame(orows)
(ROOT / "results_r3").mkdir(exist_ok=True)
B.to_csv(ROOT / "results_r3" / "R5_bootstrap_simultaneous.csv", index=False)
O.to_csv(ROOT / "results_r3" / "R5_origin_differences.csv", index=False)


def excl(g, side, kind, blk):
    lo, hi = g[f"lo_{kind}_b{blk}"], g[f"hi_{kind}_b{blk}"]
    return int((hi < 0).sum()) if side == "below" else int((lo > 0).sum())


summ = {}
for (s, ep), g in B.groupby(["set", "endpoint"]):
    k = f"{s}_{ep}"
    summ[k] = {"comparisons": len(g), "family_size": int(g.family_size.iloc[0]),
               "all_origins_negative": int((g.n_negative == g.n).sum())}
    for blk in (2, 3, 4):
        summ[k][f"b{blk}"] = {kind: {"below": excl(g, "below", kind, blk), "above": excl(g, "above", kind, blk)}
                              for kind in (["95", "sim", "joint"] if s == "T10" else ["95", "sim"])}
    if s != "T10":
        summ[k]["sim_below_by_baseline_primary"] = g.assign(x=g[f"hi_sim_b{PRIMARY_BLOCK[ep]}"] < 0).groupby("b").x.sum().astype(int).to_dict()
        summ[k]["sim_below_all_blocks_by_baseline"] = g.assign(
            x=(g[["hi_sim_b2", "hi_sim_b3", "hi_sim_b4"]] < 0).all(1)).groupby("b").x.sum().astype(int).to_dict()
t10 = B[B.set == "T10"].copy()
pb = t10.endpoint.map(PRIMARY_BLOCK)
t10["excl95"] = [(r[f"hi_95_b{b}"] < 0) or (r[f"lo_95_b{b}"] > 0) for (_, r), b in zip(t10.iterrows(), pb)]
t10["excl_sim"] = [(r[f"hi_sim_b{b}"] < 0) or (r[f"lo_sim_b{b}"] > 0) for (_, r), b in zip(t10.iterrows(), pb)]
t10["excl_joint"] = [(r[f"hi_joint_b{b}"] < 0) or (r[f"lo_joint_b{b}"] > 0) for (_, r), b in zip(t10.iterrows(), pb)]
summ["T10_primary_blocks"] = {
    "excluding_zero_95": [f"{a} - {b} ({e})" for a, b, e, x in zip(t10.a, t10.b, t10.endpoint, t10.excl95) if x],
    "lost_with_simultaneous": [f"{a} - {b} ({e})" for a, b, e, x, y in zip(t10.a, t10.b, t10.endpoint, t10.excl95, t10.excl_sim) if x and not y],
    "lost_with_joint": [f"{a} - {b} ({e})" for a, b, e, x, y in zip(t10.a, t10.b, t10.endpoint, t10.excl95, t10.excl_joint) if x and not y]}
(ROOT / "results_r3" / "r5_bootstrap_summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
