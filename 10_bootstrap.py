"""
10_bootstrap.py — circular moving-block bootstrap confidence intervals for differences in
accuracy between pairs of methods (additional experiments). Uses seed-averaged forecasts.

Rolling h=1 test sample (19 months): per-month difference in symmetric absolute percentage
error, block length 3. Complete-horizon sample (8 origins): per-origin difference in the mean
sMAPE over h=1..12, block length 2. 10,000 resamples. Output: results/T13_bootstrap.csv
"""
import glob
import numpy as np
import pandas as pd
from common import RUNS_DIR, RESULTS_DIR, H

PAIRS = [("gru-direct", "tcn-direct"), ("gru-direct", "lstm-direct"), ("lstm-recursive", "gru-direct"),
         ("gru-direct", "sarima"), ("gru-direct", "naive"), ("tcn-direct", "sarima"), ("lstm-recursive", "sarima"),
         ("lstm-recursive", "naive"), ("lstm-direct", "ets"), ("gru-direct", "ets"), ("tcn-direct", "ets")]


def cbb_ci(d, block, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    k = int(np.ceil(n / block))
    means = np.empty(reps)
    dd = np.concatenate([d, d[:block]])
    for r in range(reps):
        starts = rng.integers(0, n, k)
        sample = np.concatenate([dd[s:s + block] for s in starts])[:n]
        means[r] = sample.mean()
    return np.percentile(means, [2.5, 97.5]), float((means > 0).mean())


def main():
    files = glob.glob(str(RUNS_DIR / "2022-12" / "*" / "*" / "seed*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["method"] = np.where(df.strategy == "classical", df.model, df.model + "-" + df.strategy)
    ens = df[df.origin_period == "test"].groupby(["method", "origin", "h"], as_index=False).agg(
        y=("y", "first"), yhat=("yhat", "mean"))
    ens["ape"] = 200 * (ens.yhat - ens.y).abs() / (ens.y.abs() + ens.yhat.abs())
    cnt = ens.groupby("origin").h.nunique()
    comp = cnt[cnt == H].index
    rows = []
    for a, b in PAIRS:
        A = ens[ens.method == a].set_index(["origin", "h"]).ape
        B = ens[ens.method == b].set_index(["origin", "h"]).ape
        d1 = (A - B).xs(1, level="h").sort_index().values
        ci1, p1 = cbb_ci(d1, 3)
        dc = (A - B).rename("d").reset_index()
        dc = dc[dc.origin.isin(comp)].groupby("origin")["d"].mean().sort_index().values
        cic, pc = cbb_ci(dc, 2)
        rows.append(dict(method_A=a, method_B=b, h1_diff=d1.mean(), h1_ci_lo=ci1[0], h1_ci_hi=ci1[1], h1_share_pos=p1,
                         full_diff=dc.mean(), full_ci_lo=cic[0], full_ci_hi=cic[1], full_share_pos=pc,
                         n_h1=len(d1), n_origins=len(dc)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "T13_bootstrap.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
