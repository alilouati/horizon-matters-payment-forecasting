"""
02_anomaly_screening.py — retrospective anomaly screening (manuscript Section 4.7).

Input : data/monthly_series.csv
Output: data/anomaly_flags.csv            one row per month with both flags
        results/table_anomaly_by_split.csv

Detector 1 (Hampel/MAD): centred 7-month window t-3..t+3,
    m_t = median(window), s_t = 1.4826 * median(|y_j - m_t|),
    flag if |y_t - m_t| > 3 s_t. The first and last 3 months have no complete
    centred window and are not screened.
Detector 2 (robust STL): STL(period=12, robust=True) on log1p(y); the residual
    r_t is standardised robustly, z_t = (r_t - median r) / (1.4826 MAD(r)),
    flag if |z_t| > 3.
Combined flag: Hampel OR STL.

Flags are used only for interpretation and sensitivity analysis; the forecasting
series is never modified.
"""
import numpy as np
import pandas as pd
from common import (load_series, ANOMALY_FILE, RESULTS_DIR, TARGET,
                    HAMPEL_WINDOW, HAMPEL_K, MAD_SCALE, STL_PERIOD, STL_K)


def hampel_flags(y, window=HAMPEL_WINDOW, k=HAMPEL_K):
    y = np.asarray(y, float)
    half = window // 2
    flags = np.zeros(len(y), dtype=int)
    for t in range(half, len(y) - half):
        w = y[t - half:t + half + 1]
        m = np.median(w)
        s = MAD_SCALE * np.median(np.abs(w - m))
        flags[t] = int(s > 0 and abs(y[t] - m) > k * s)
    return flags


def stl_flags(y, period=STL_PERIOD, k=STL_K):
    from statsmodels.tsa.seasonal import STL
    res = STL(np.log1p(np.asarray(y, float)), period=period, robust=True).fit()
    r = res.resid
    med = np.median(r)
    s = MAD_SCALE * np.median(np.abs(r - med))
    z = (r - med) / s
    return (np.abs(z) > k).astype(int), z


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_series()
    y = df[TARGET].values

    df["anom_hampel"] = hampel_flags(y)
    df["anom_stl"], df["stl_robust_z"] = stl_flags(y)
    df["anom_any"] = (df["anom_hampel"] | df["anom_stl"]).astype(int)
    df["anom_stl_only"] = ((df["anom_stl"] == 1) & (df["anom_hampel"] == 0)).astype(int)
    df.to_csv(ANOMALY_FILE, index=False)

    summ = (df.groupby("split")[["anom_hampel", "anom_stl_only", "anom_any"]].sum()
              .reindex(["train", "val", "test"]))
    summ.insert(0, "n_months", df.groupby("split").size().reindex(summ.index))
    summ.loc["total"] = summ.sum()
    summ["pct_any"] = 100 * summ["anom_any"] / summ["n_months"]
    summ.to_csv(RESULTS_DIR / "table_anomaly_by_split.csv")
    print(summ.to_string())
    flagged = df.loc[df["anom_any"] == 1, ["period_start", "split", TARGET, "anom_hampel", "anom_stl"]]
    print("\nflagged months:")
    print(flagged.assign(period_start=flagged["period_start"].dt.strftime("%Y-%m")).to_string(index=False))


if __name__ == "__main__":
    main()
