"""
14_normalization_bootstrap.py — uncertainty of the main comparison of the normalization experiments.
Circular moving-block bootstrap (10,000 resamples) of the mean difference in symmetric absolute percentage error
between each configuration and each classical baseline, using seed-averaged forecasts:
rolling h=1 test sample (19 months, block 3) and complete-horizon sample (8 origins, per-origin mean over h, block 2).
Sets: the configurations selected with the corrections (runs_r3/tuned_r3) and the RevIN + early-stopping
factorial cell (runs_r3/revin_es). Output: results_r3/R3_bootstrap.csv, r3_summary_bootstrap.json
"""
import glob
import json
import numpy as np
import pandas as pd
from common import ROOT, H

SEEDS = range(10)


ALPHA_BONF = None   # set below: Bonferroni over the comparisons of one set


def cbb_ci(d, block, reps=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(d)
    k = int(np.ceil(n / block))
    dd = np.concatenate([d, d[:block]])
    means = np.array([np.concatenate([dd[s:s + block] for s in rng.integers(0, n, k)])[:n].mean() for _ in range(reps)])
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def rd(p):
    fs = glob.glob(str(p))
    return pd.concat((pd.read_csv(f) for f in fs), ignore_index=True) if fs else pd.DataFrame()


cls = rd(ROOT / "runs" / "2022-12" / "classical" / "*" / "seed*.csv")
cls["method"] = cls.model
rows = []
for setname in ["tuned_r3", "revin_es"]:
    nn = rd(ROOT / "runs_r3" / setname / "2022-12" / "*" / "*" / "seed*.csv")
    nn = nn[nn.seed.isin(SEEDS)]
    nn["method"] = nn.model + "-" + nn.strategy
    d = pd.concat([nn, cls], ignore_index=True)
    d = d[d.origin_period == "test"]
    e = d.groupby(["method", "origin", "h"], as_index=False).agg(y=("y", "first"), yhat=("yhat", "mean"))
    e["ape"] = 200 * (e.yhat - e.y).abs() / (e.y.abs() + e.yhat.abs())
    cnt = cls[cls.origin_period == "test"].groupby("origin").h.nunique()
    comp = cnt[cnt == H].index
    A = e.set_index(["method", "origin", "h"]).ape
    for m in sorted(nn.method.unique()):
        for c in ["naive", "snaive", "ets", "sarima"]:
            diff = (A.loc[m] - A.loc[c]).rename("d").reset_index()
            d1 = diff[diff.h == 1].sort_values("origin").d.values
            dc = diff[diff.origin.isin(comp)].groupby("origin").d.mean().sort_index().values
            c1, cc = cbb_ci(d1, 3), cbb_ci(dc, 2)
            ab = 0.05 / (4 * nn.method.nunique())          # Bonferroni over all configuration-baseline pairs
            b1, bc = cbb_ci(d1, 3, reps=100000, alpha=ab), cbb_ci(dc, 2, reps=100000, alpha=ab)
            rows.append(dict(set=setname, method=m, baseline=c, h1_diff=d1.mean(), h1_lo=c1[0], h1_hi=c1[1],
                             full_diff=dc.mean(), full_lo=cc[0], full_hi=cc[1], h1_lo_bonf=b1[0], h1_hi_bonf=b1[1],
                             full_lo_bonf=bc[0], full_hi_bonf=bc[1]))
b = pd.DataFrame(rows)
b.to_csv(ROOT / "results_r3" / "R3_bootstrap.csv", index=False)
summ = {}
for s, g in b.groupby("set"):
    best = g.groupby("method").apply(lambda x: x.loc[x.full_diff.idxmax()])  # least favourable baseline per method
    summ[s] = {
        "n_methods": int(g.method.nunique()),
        "full_lower_than_all_baselines_ci": int(g.groupby("method").apply(lambda x: (x.full_hi < 0).all()).sum()),
        "full_ci_below_zero_by_baseline": g.groupby("baseline").apply(lambda x: int((x.full_hi < 0).sum())).to_dict(),
        "h1_ci_below_zero_by_baseline": g.groupby("baseline").apply(lambda x: int((x.h1_hi < 0).sum())).to_dict(),
        "h1_ci_above_zero_by_baseline": g.groupby("baseline").apply(lambda x: int((x.h1_lo > 0).sum())).to_dict(),
        "bonf_full_below_zero_by_baseline": g.groupby("baseline").apply(lambda x: int((x.full_hi_bonf < 0).sum())).to_dict(),
        "bonf_full_all_baselines": int(g.groupby("method").apply(lambda x: (x.full_hi_bonf < 0).all()).sum()),
        "bonf_h1_below_zero_by_baseline": g.groupby("baseline").apply(lambda x: int((x.h1_hi_bonf < 0).sum())).to_dict(),
    }
(ROOT / "results_r3" / "r3_summary_bootstrap.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
print(b[b.set == "tuned_r3"].round(2).to_string(index=False))
