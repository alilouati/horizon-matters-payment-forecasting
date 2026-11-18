"""
dmtest.py — Diebold–Mariano test used throughout the package (05_analyze.py, 12_normalization_analysis.py,
15_dm_tests_smape.py).

Given a loss differential d_1..d_n for forecast horizon h:

    dbar      = mean(d)
    gamma_k   = (1/n) * sum_{t=k+1}^{n} (d_t - dbar)(d_{t-k} - dbar),   k = 0..h-1
    V_rect    = (gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k) / n               (truncated, untapered; Diebold and Mariano 1995)
    V_bart    = (gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) gamma_k) / n     (Bartlett / Newey–West weights; sensitivity only)
    DM        = dbar / sqrt(V)
    HLN       = sqrt( (n + 1 - 2h + h(h-1)/n) / n )                       (Harvey, Leybourne and Newbold 1997)
    stat      = HLN * DM,  p = 2 * P(t_{n-1} > |stat|)

The forecast horizon h enters both the autocovariance bandwidth (h-1 lags) and the HLN factor; no substitute horizon
is used. A test is reported only when it is defined:

    status "ok"           test computed
    "na_horizon"          h >= n: the bandwidth h-1 would exceed n-2 and the HLN factor is not positive
                          (with n = 8 complete origins: horizons 8-12). No p-value.
    "na_variance"         V_rect <= 0: the truncated long-run variance estimate is unusable. No p-value in the
                          primary analysis (no fallback); the Bartlett-weighted estimate is reported as sensitivity.
    "na_constant"         the loss differential is constant (no test possible).

Earlier versions of this package replaced h by min(h-1, n-2)+1 in the HLN factor and replaced V_rect <= 0 by gamma_0/n.
`legacy_dm` reproduces that procedure so that the effect of the change can be documented.
"""
import numpy as np
from scipy import stats


def _gammas(d, lags):
    n = len(d)
    dc = d - d.mean()
    return np.array([np.sum(dc[k:] * dc[:n - k]) / n for k in range(lags + 1)])


def dm_test(d, h):
    """Return dict with stat, p (primary, rectangular kernel), status, and Bartlett sensitivity stat_bart, p_bart."""
    d = np.asarray(d, float)
    n = len(d)
    out = dict(n=n, h=h, stat=np.nan, p=np.nan, status="ok", stat_bart=np.nan, p_bart=np.nan,
               var_rect=np.nan, var_bart=np.nan, hln=np.nan)
    if n < 3 or np.allclose(d, d[0]):
        out["status"] = "na_constant"
        return out
    hln2 = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if h - 1 > n - 2 or hln2 <= 0:
        out["status"] = "na_horizon"
        return out
    g = _gammas(d, h - 1)
    w = 1 - np.arange(1, h) / h
    v_rect = (g[0] + 2 * g[1:].sum()) / n
    v_bart = (g[0] + 2 * (w * g[1:]).sum()) / n
    out.update(var_rect=v_rect, var_bart=v_bart, hln=np.sqrt(hln2))
    if v_bart > 0:
        sb = np.sqrt(hln2) * d.mean() / np.sqrt(v_bart)
        out.update(stat_bart=sb, p_bart=2 * stats.t.sf(abs(sb), n - 1))
    if v_rect <= 0:
        out["status"] = "na_variance"
        return out
    s = np.sqrt(hln2) * d.mean() / np.sqrt(v_rect)
    out.update(stat=s, p=2 * stats.t.sf(abs(s), n - 1))
    return out


def legacy_dm(d, h):
    """Procedure of the previous package version (capped horizon in HLN, gamma_0/n fallback). Documentation only."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return np.nan, np.nan, False, False
    lag = min(h - 1, n - 2)
    g = _gammas(d, lag)
    var = (g[0] + 2 * g[1:].sum()) / n
    fallback = var <= 0
    if fallback:
        var = g[0] / n
    hh = lag + 1
    corr = (n + 1 - 2 * hh + hh * (hh - 1) / n) / n
    stat = d.mean() / np.sqrt(var)
    if corr <= 0:
        return stat, np.nan, hh != h, fallback
    stat *= np.sqrt(corr)
    return stat, 2 * stats.t.sf(abs(stat), n - 1), hh != h, fallback


def holm(p):
    """Holm-adjusted p-values; NaN entries (tests not available) are excluded from the family."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out
