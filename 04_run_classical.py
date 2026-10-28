"""
04_run_classical.py — classical reference forecasts on EXACTLY the same origins,
horizons and target months as the neural experiments (answers R1 comment 1).

Models, re-fitted at every origin t on all observations before t (expanding window):
  naive    : y_hat(t+h-1) = y(t-1)
  snaive   : y_hat(t+h-1) = y(t+h-1-12)          (valid for h <= 12)
  ets      : Holt-Winters on log1p(y), additive damped trend, additive seasonality (12)
  sarima   : SARIMA(1,1,1)(0,1,1)_12 on log1p(y)
Forecasts on the log scale are returned with expm1.

Output: runs/<cutoff>/classical/<model>/seed00.csv  (same schema as the neural runs)
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from common import (load_series, cutoff_index, period_label, RUNS_DIR, ROOT, TARGET, H,
                    MAIN_CUTOFF, EXTRA_CUTOFFS, SARIMA_ORDER, SARIMA_SEASONAL_ORDER)

CLASSICAL = ["naive", "snaive", "ets", "sarima"]


def forecast(name, hist):
    """12-step forecast from the observed history (numpy array, original scale)."""
    if name == "naive":
        return np.repeat(hist[-1], H)
    if name == "snaive":
        return np.array([hist[len(hist) - 12 + (h - 1)] for h in range(1, H + 1)])
    logy = np.log1p(hist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "ets":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(logy, trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=12).fit(optimized=True)
            return np.expm1(fit.forecast(H))
        if name == "sarima":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            fit = SARIMAX(logy, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            return np.expm1(fit.forecast(H))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CLASSICAL)
    ap.add_argument("--cutoffs", default="main", help="'main', 'extra', or 'all'")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--runs-dir", default=None, help="output folder (default: runs)")
    args = ap.parse_args()
    runs_dir = ROOT / args.runs_dir if args.runs_dir else RUNS_DIR
    cutoffs = {"main": [MAIN_CUTOFF], "extra": EXTRA_CUTOFFS,
               "all": [MAIN_CUTOFF] + EXTRA_CUTOFFS}[args.cutoffs]

    df = load_series()
    dates = df["period_start"].values
    y = df[args.target].values.astype(float)

    for c in cutoffs:
        ci = cutoff_index(dates, c)
        for name in args.models:
            out = runs_dir / pd.Timestamp(c).strftime("%Y-%m") / "classical" / name / "seed00.csv"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for t in range(ci + 1, len(y)):
                yhat = forecast(name, y[:t])
                for h in range(1, H + 1):
                    tgt = t + h - 1
                    if tgt >= len(y):
                        break
                    rows.append({
                        "cutoff": pd.Timestamp(c).strftime("%Y-%m"), "strategy": "classical",
                        "model": name, "seed": 0,
                        "origin": pd.Timestamp(dates[t]).strftime("%Y-%m"),
                        "origin_period": period_label(dates[t]) if c == MAIN_CUTOFF else "backtest",
                        "h": h, "target": pd.Timestamp(dates[tgt]).strftime("%Y-%m"),
                        "y": y[tgt], "yhat": float(yhat[h - 1]),
                    })
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{pd.Timestamp(c):%Y-%m} {name}: {len(rows)} forecasts saved")


if __name__ == "__main__":
    main()
