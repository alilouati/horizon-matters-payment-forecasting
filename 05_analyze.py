"""
05_analyze.py — every table, test and figure for the manuscript, computed
only from the saved per-origin predictions in runs/ (nothing is retrained here).

Outputs in results/ (see README for the mapping to manuscript tables):
  T1_rolling_h1.csv              rolling h=1 metrics, val (n=12) and test (n=19): mean, SD, 95% CI over seeds
  T2_horizon_profile.csv         complete-horizon test metrics per h (8 origins), mean/SD/CI over seeds
  T3_win_probability.csv         P(model A < model B) over all seed pairs, per horizon
  T4_dm_tests.csv                Diebold-Mariano tests (Harvey-Leybourne-Newbold correction) on seed-averaged forecasts
  T5_bias.csv                    mean signed error over seeds; share of over-predictions; sign test
  T6_skill_vs_snaive.csv         sMAPE relative to Seasonal Naive on identical (origin, h) pairs
  T7_error_by_target_month.csv   absolute % error per (origin, h, target month) — separates horizon from target month
  T8_anomaly_sensitivity.csv     results with flagged test months excluded
  T9_extrapolation.csv           test values above the training maximum; share of under-predictions
  T10_parameters.csv             trainable parameters, training windows and training time
  T11_backtest.csv               results for each training cutoff (reproducibility across periods)
  fig_*.png
"""
import glob
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import (RUNS_DIR, RESULTS_DIR, ANOMALY_FILE, H, smape, mae, rmse, bias,
                    load_series, TARGET, TRAIN_END, MAIN_CUTOFF)

MAIN = pd.Timestamp(MAIN_CUTOFF).strftime("%Y-%m")
KEY = ["strategy", "model"]


# ---------------------------------------------------------------------------
def load_runs():
    files = glob.glob(str(RUNS_DIR / "*" / "*" / "*" / "seed*.csv"))
    if not files:
        raise SystemExit("no prediction files found in runs/ — run 03 and 04 first")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["method"] = np.where(df["strategy"] == "classical", df["model"],
                            df["model"] + "-" + df["strategy"])
    df["err"] = df["yhat"] - df["y"]
    df["ape"] = 200 * np.abs(df["err"]) / (np.abs(df["y"]) + np.abs(df["yhat"]))
    return df


def complete_origins(df, period="test"):
    """Origins of the main experiment with all H targets available (8 for the test period)."""
    sub = df[(df["cutoff"] == MAIN) & (df["origin_period"] == period)]
    counts = sub.groupby("origin")["h"].nunique()
    return sorted(counts[counts == H].index)


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - half, x.mean() + half


def per_seed_metrics(sub, by):
    rows = []
    for k, g in sub.groupby(by + ["seed"]):
        rows.append(dict(zip(by + ["seed"], k), sMAPE=smape(g.y, g.yhat), MAE=mae(g.y, g.yhat) / 1e9,
                         RMSE=rmse(g.y, g.yhat) / 1e9, bias=bias(g.y, g.yhat) / 1e9, n=len(g)))
    return pd.DataFrame(rows)


def summarise(ps, by, metrics=("sMAPE", "MAE", "RMSE", "bias")):
    rows = []
    for k, g in ps.groupby(by):
        r = dict(zip(by, k if isinstance(k, tuple) else (k,)), n_seeds=g["seed"].nunique(), n=int(g["n"].iloc[0]))
        for m in metrics:
            lo, hi = ci95(g[m])
            r.update({f"{m}_mean": g[m].mean(), f"{m}_sd": g[m].std(ddof=1), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def ensemble(sub, by):
    """Seed-averaged forecast for every (method, origin, h)."""
    return (sub.groupby(by + ["origin", "h", "target"], as_index=False)
               .agg(y=("y", "first"), yhat=("yhat", "mean")))


def dm_hln(e1, e2, h):
    """Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
    Loss: squared error. Negative statistic -> forecast 1 more accurate.
    The autocovariance lag is h-1, capped at n-2 when the sample is shorter;
    returns NaN p-value when the HLN correction is undefined."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return np.nan, np.nan, 0
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
        return stat, np.nan, lag
    stat *= np.sqrt(corr)
    return stat, 2 * stats.t.sf(abs(stat), n - 1), lag


# ---------------------------------------------------------------------------
def main():
    import argparse
    global RUNS_DIR, RESULTS_DIR, TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--target", default=TARGET)
    a = ap.parse_args()
    from common import ROOT
    if a.runs_dir:
        RUNS_DIR = ROOT / a.runs_dir
    if a.results_dir:
        RESULTS_DIR = ROOT / a.results_dir
    TARGET = a.target
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_runs()
    main_df = df[df["cutoff"] == MAIN]
    neural = main_df[main_df["strategy"] != "classical"]
    by = ["method"] + KEY
    comp = complete_origins(main_df, "test")
    print(f"complete test origins: {len(comp)} ({comp[0]} .. {comp[-1]})" if comp else "no complete origins")

    # ---- T1 rolling h=1 -------------------------------------------------
    t1 = []
    for period in ["val", "test"]:
        sub = main_df[(main_df["h"] == 1) & (main_df["origin_period"] == period)]
        s = summarise(per_seed_metrics(sub, by), by)
        ens = ensemble(sub, by)
        s["sMAPE_seed_avg_forecast"] = s["method"].map(ens.groupby("method").apply(lambda g: smape(g.y, g.yhat)))
        s.insert(0, "period", period)
        t1.append(s)
    t1 = pd.concat(t1).sort_values(["period", "sMAPE_mean"])
    t1.to_csv(RESULTS_DIR / "T1_rolling_h1.csv", index=False)

    # ---- T2 complete-horizon profile ------------------------------------
    cdf = main_df[main_df["origin"].isin(comp) & (main_df["origin_period"] == "test")]
    ps_h = per_seed_metrics(cdf, by + ["h"])
    t2 = summarise(ps_h, by + ["h"])
    mean_over_h = (ps_h.groupby(by + ["seed"])["sMAPE"].mean().reset_index()
                     .groupby(by)["sMAPE"].agg(["mean", "std"]).rename(
                         columns={"mean": "sMAPE_mean_over_h", "std": "sMAPE_mean_over_h_sd"}).reset_index())
    t2 = t2.merge(mean_over_h, on=by)
    t2.sort_values(["strategy", "model", "h"]).to_csv(RESULTS_DIR / "T2_horizon_profile.csv", index=False)

    # ---- T3 win probabilities (all seed pairs) --------------------------
    rows = []
    nps = ps_h[ps_h["strategy"] != "classical"]
    h1 = per_seed_metrics(neural[(neural["h"] == 1) & (neural["origin_period"] == "test")], by)
    for strat in nps["strategy"].unique():
        models = sorted(nps.loc[nps["strategy"] == strat, "model"].unique())
        for a, b in itertools.combinations(models, 2):
            def wp(xa, xb):
                xa, xb = np.asarray(xa), np.asarray(xb)
                return float(np.mean(xa[:, None] < xb[None, :])) if len(xa) and len(xb) else np.nan
            ga = h1[(h1.strategy == strat) & (h1.model == a)]["sMAPE"]
            gb = h1[(h1.strategy == strat) & (h1.model == b)]["sMAPE"]
            rows.append(dict(strategy=strat, model_A=a, model_B=b, evaluation="rolling_h1_test_n19",
                             h=1, P_A_better=wp(ga, gb)))
            for h in range(1, H + 1):
                ga = nps[(nps.strategy == strat) & (nps.model == a) & (nps.h == h)]["sMAPE"]
                gb = nps[(nps.strategy == strat) & (nps.model == b) & (nps.h == h)]["sMAPE"]
                rows.append(dict(strategy=strat, model_A=a, model_B=b, evaluation="complete_test_n8",
                                 h=h, P_A_better=wp(ga, gb)))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "T3_win_probability.csv", index=False)

    # ---- T4 Diebold-Mariano on seed-averaged forecasts ------------------
    rows = []
    for label, sub, hs in [("rolling_h1_test_n19", main_df[(main_df.h == 1) & (main_df.origin_period == "test")], [1]),
                           ("complete_test_n8", cdf, range(1, H + 1))]:
        ens = ensemble(sub, ["method"])
        methods = sorted(ens["method"].unique())
        for h in hs:
            eh = ens[ens.h == h].pivot(index="origin", columns="method", values="yhat")
            yv = ens[ens.h == h].groupby("origin")["y"].first().reindex(eh.index)
            for a, b in itertools.combinations(methods, 2):
                if a not in eh or b not in eh:
                    continue
                st, p, lag = dm_hln(eh[a] - yv, eh[b] - yv, h)
                rows.append(dict(evaluation=label, h=h, method_A=a, method_B=b, n=len(yv),
                                 dm_stat=st, p_value=p, nw_lag=lag))
    t4 = pd.DataFrame(rows)
    t4.to_csv(RESULTS_DIR / "T4_dm_tests.csv", index=False)

    # ---- T5 bias -------------------------------------------------------
    sub = main_df[(main_df.h == 1) & (main_df.origin_period == "test")]
    ps = per_seed_metrics(sub, by)
    ens = ensemble(sub, by)
    rows = []
    for (m, s_, mo), g in ens.groupby(by):
        e = g.yhat - g.y
        npos = int((e > 0).sum())
        rows.append(dict(method=m, strategy=s_, model=mo, n=len(e),
                         bias_seed_mean_bnSAR=ps.loc[ps.method == m, "bias"].mean(),
                         bias_seed_sd_bnSAR=ps.loc[ps.method == m, "bias"].std(ddof=1),
                         bias_seed_avg_forecast_bnSAR=e.mean() / 1e9,
                         MAE_seed_avg_forecast_bnSAR=e.abs().mean() / 1e9,
                         n_over=npos, share_over=npos / len(e),
                         sign_test_p=stats.binomtest(npos, len(e), 0.5).pvalue,
                         t_test_p=stats.ttest_1samp(e, 0).pvalue))
    pd.DataFrame(rows).sort_values("MAE_seed_avg_forecast_bnSAR").to_csv(RESULTS_DIR / "T5_bias.csv", index=False)

    # ---- T6 skill relative to seasonal naive ----------------------------
    ens = ensemble(cdf, ["method"])
    if "snaive" in ens["method"].unique():
        base = ens[ens.method == "snaive"].groupby("h").apply(lambda g: smape(g.y, g.yhat))
        t6 = (ens.groupby(["method", "h"]).apply(lambda g: smape(g.y, g.yhat)).rename("sMAPE").reset_index())
        t6["sMAPE_snaive"] = t6["h"].map(base)
        t6["relative_sMAPE"] = t6["sMAPE"] / t6["sMAPE_snaive"]
        t6.to_csv(RESULTS_DIR / "T6_skill_vs_snaive.csv", index=False)

    # ---- T7 error by target month ---------------------------------------
    t7 = ens[["method", "origin", "h", "target", "y", "yhat"]].copy()
    t7["ape"] = 200 * (t7.yhat - t7.y).abs() / (t7.y.abs() + t7.yhat.abs())
    t7.to_csv(RESULTS_DIR / "T7_error_by_target_month.csv", index=False)
    (t7.pivot_table(index="target", columns="method", values="ape", aggfunc="mean")
       .to_csv(RESULTS_DIR / "T7b_mean_ape_by_target_month.csv"))

    # ---- T8 anomaly sensitivity -----------------------------------------
    excl_sets = {"none": [], "2024-03": ["2024-03"], "2025-03": ["2025-03"],
                 "both_March": ["2024-03", "2025-03"]}
    if ANOMALY_FILE.exists() and TARGET == "pos_sales_SAR":
        an = pd.read_csv(ANOMALY_FILE, parse_dates=["period_start"])
        flagged_test = an.loc[(an.anom_any == 1) & (an.split == "test"), "period_start"].dt.strftime("%Y-%m").tolist()
        excl_sets["all_flagged_test"] = flagged_test
    rows = []
    for name, ex in excl_sets.items():
        r1 = main_df[(main_df.h == 1) & (main_df.origin_period == "test") & ~main_df.target.isin(ex)]
        r2 = cdf[~cdf.target.isin(ex)]
        a = per_seed_metrics(r1, by).groupby("method")["sMAPE"].agg(["mean", "std"])
        b = (per_seed_metrics(r2, by + ["h"]).groupby(["method", "seed"])["sMAPE"].mean()
               .groupby("method").agg(["mean", "std"]))
        for m in a.index.union(b.index):
            rows.append(dict(excluded=name, excluded_months=";".join(ex), method=m,
                             rolling_h1_sMAPE_mean=a["mean"].get(m), rolling_h1_sMAPE_sd=a["std"].get(m),
                             complete_mean_over_h_sMAPE=b["mean"].get(m), complete_sd=b["std"].get(m)))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "T8_anomaly_sensitivity.csv", index=False)

    # ---- T9 extrapolation ------------------------------------------------
    ser = load_series()
    train_max = ser.loc[ser.period_start <= TRAIN_END, TARGET].max()
    rows = [dict(item="training maximum (bn SAR)", value=train_max / 1e9)]
    for p in ["val", "test"]:
        v = ser.loc[ser.split == p, TARGET]
        rows.append(dict(item=f"{p}: months above training maximum", value=f"{int((v > train_max).sum())} of {len(v)}"))
    sub = ensemble(main_df[(main_df.h == 1) & (main_df.origin_period == "test")], ["method"])
    for m, g in sub.groupby("method"):
        rows.append(dict(item=f"{m}: share of test months under-predicted (rolling h=1)",
                         value=round(float((g.yhat < g.y).mean()), 3)))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "T9_extrapolation.csv", index=False)

    # ---- T10 parameters --------------------------------------------------
    log_files = glob.glob(str(RUNS_DIR / "_logs" / "*.csv")) or glob.glob(str(RUNS_DIR / "run_log.csv"))
    if log_files:
        lg = pd.concat((pd.read_csv(f) for f in log_files), ignore_index=True)
        lg.to_csv(RESULTS_DIR / "run_log_all.csv", index=False)
        lg["cutoff"] = lg["cutoff"].astype(str)
        lg = lg[lg.cutoff == MAIN]
        (lg.groupby(["strategy", "model"]).agg(n_params=("n_params", "first"),
                                               n_train_windows=("n_train_windows", "first"),
                                               train_seconds_mean=("train_seconds", "mean"),
                                               n_seeds=("seed", "nunique"))
           .reset_index().to_csv(RESULTS_DIR / "T10_parameters.csv", index=False))

    # ---- T11 backtest across cutoffs ------------------------------------
    rows = []
    for c, g in df.groupby("cutoff"):
        c0 = pd.Period(c, "M")
        window = [str(c0 + k) for k in range(1, 32)]        # 31 months after the cutoff (= val + test length)
        g = g[g.origin.isin(window)]
        cnt = g.groupby("origin")["h"].nunique()
        comp_c = cnt[cnt == H].index
        h1 = per_seed_metrics(g[g.h == 1], by).groupby(by)["sMAPE"].agg(["mean", "std"])
        full = (per_seed_metrics(g[g.origin.isin(comp_c)], by + ["h"]).groupby(by + ["seed"])["sMAPE"].mean()
                  .groupby(by).agg(["mean", "std"]))
        for k in h1.index:
            rows.append(dict(cutoff=c, method=k[0], strategy=k[1], model=k[2],
                             n_origins_h1=int(g.origin.nunique()), n_complete_origins=len(comp_c),
                             h1_sMAPE_mean=h1.loc[k, "mean"], h1_sMAPE_sd=h1.loc[k, "std"],
                             full_sMAPE_mean=full["mean"].get(k), full_sMAPE_sd=full["std"].get(k)))
    t11 = pd.DataFrame(rows)
    if len(t11):
        t11["h1_rank"] = t11.groupby("cutoff")["h1_sMAPE_mean"].rank()
        t11["full_rank"] = t11.groupby("cutoff")["full_sMAPE_mean"].rank()
    t11.to_csv(RESULTS_DIR / "T11_backtest.csv", index=False)

    make_figures(t1, t2, t11)
    print(f"results written to {RESULTS_DIR}")


# ---------------------------------------------------------------------------
def make_figures(t1, t2, t11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10})

    # Figure: horizon profile with 95% CI band, one panel per neural strategy
    strategies = [s for s in ["direct", "recursive"] if s in t2.strategy.unique()]
    if strategies:
        fig, axes = plt.subplots(1, len(strategies), figsize=(6.5 * len(strategies), 4.5),
                                 sharey=True, squeeze=False, constrained_layout=True)
        for ax, s in zip(axes[0], strategies):
            for m, g in t2[t2.strategy == s].groupby("model"):
                g = g.sort_values("h")
                ax.plot(g.h, g.sMAPE_mean, marker="o", label=m)
                ax.fill_between(g.h, g.sMAPE_ci_lo, g.sMAPE_ci_hi, alpha=0.2)
            sn = t2[(t2.strategy == "classical") & (t2.model == "snaive")].sort_values("h")
            if len(sn):
                ax.plot(sn.h, sn.sMAPE_mean, "k--", label="Seasonal Naive")
            ax.set_title(f"{s} strategy")
            ax.set_xlabel("Forecast horizon h (months)")
            ax.set_xticks(range(1, H + 1))
        axes[0][0].set_ylabel("Test sMAPE (%), mean and 95% CI over seeds")
        axes[0][-1].legend(loc="upper left")
        fig.savefig(RESULTS_DIR / "fig_horizon_profile.png", dpi=300)
        plt.close(fig)

    # Figure: rolling h=1 test sMAPE with 95% CI
    t = t1[t1.period == "test"].sort_values("sMAPE_mean")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.5), constrained_layout=True)
    yv = np.arange(len(t))
    err = np.vstack([t.sMAPE_mean - t.sMAPE_ci_lo, t.sMAPE_ci_hi - t.sMAPE_mean]).astype(float)
    err = np.nan_to_num(err)
    ax.errorbar(t.sMAPE_mean, yv, xerr=err, fmt="o", capsize=3)
    ax.set_yticks(yv)
    ax.set_yticklabels(t.method)
    ax.invert_yaxis()
    ax.set_xlabel("Rolling h=1 test sMAPE (%), mean and 95% CI over seeds (n=19 months)")
    fig.savefig(RESULTS_DIR / "fig_rolling_h1_test.png", dpi=300)
    plt.close(fig)

    # Figure: backtest across cutoffs
    if len(t11) and t11.cutoff.nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for m, g in t11.groupby("method"):
            g = g.sort_values("cutoff")
            ax.plot(g.cutoff, g.full_sMAPE_mean, marker="o", label=m)
        ax.set_xlabel("Training cutoff")
        ax.set_ylabel("Mean sMAPE over h=1..12 (%)")
        ax.legend(ncol=2, fontsize=9)
        fig.savefig(RESULTS_DIR / "fig_backtest.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
