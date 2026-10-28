"""
engine.py — training/forecasting engine for configurable models (additional experiments).

Supports: any target column, optional covariates (direct strategy only), configurable
architecture and training settings, per-epoch training/validation loss curves, and the
validation score used for hyperparameter selection (validation data only).

Normalization experiments: cfg["norm"] selects the normalisation wrapper (cfgmodels.build), and
cfg["train"]["patience"] enables early stopping on the validation loss (masked MSE over all
validation (origin, horizon) pairs, encoded scale); the weights of the best epoch are restored.
"""
import copy
import os
import random
import time

import numpy as np
import pandas as pd

from common import (load_series, cutoff_index, TrainOnlyEncoder, period_label, smape,
                    LOOKBACK, H, GRAD_CLIP, MAIN_CUTOFF, VAL_END)
import cfgmodels


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare(df, cutoff, target, covariates):
    """Encoded matrix Z (T x C; target in column 0), raw target y, cutoff index, target encoder."""
    dates = df["period_start"].values
    ci = cutoff_index(dates, cutoff)
    cols = [target] + list(covariates)
    encs = [TrainOnlyEncoder(df[c].values[:ci + 1]) for c in cols]
    Z = np.stack([e.encode(df[c].values) for e, c in zip(encs, cols)], axis=1).astype(np.float32)
    return Z, df[target].values.astype(float), ci, encs[0], dates


def windows(Z, ci, out_len):
    X, Y = [], []
    for t in range(LOOKBACK, ci - out_len + 2):
        X.append(Z[t - LOOKBACK:t])
        Y.append(Z[t:t + out_len, 0])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def val_pairs(Z, ci, val_end_idx, out_len):
    """Validation inputs/targets with a mask for targets inside the validation period."""
    X, Y, M = [], [], []
    for t in range(ci + 1, val_end_idx + 1):
        X.append(Z[t - LOOKBACK:t])
        tgt = np.zeros(out_len, np.float32)
        msk = np.zeros(out_len, np.float32)
        for k in range(out_len):
            if t + k <= val_end_idx:
                tgt[k], msk[k] = Z[t + k, 0], 1.0
        Y.append(tgt)
        M.append(msk)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(M, np.float32)


def forecast_path(model, Zhist, strategy, device):
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.tensor(Zhist[-LOOKBACK:][None], dtype=torch.float32, device=device)
        if strategy == "direct":
            return model(x).cpu().numpy().reshape(-1)[:H]
        if Zhist.shape[1] != 1:
            raise ValueError("recursive strategy requires a univariate input")
        hist = list(map(float, Zhist[-LOOKBACK:, 0]))
        out = []
        for _ in range(H):
            xx = torch.tensor(np.asarray(hist[-LOOKBACK:], np.float32)[None, :, None], device=device)
            z = float(model(xx).cpu().numpy().reshape(-1)[0])
            out.append(z)
            hist.append(z)
        return np.asarray(out)


def train_eval(df, *, cutoff=MAIN_CUTOFF, target="pos_sales_SAR", covariates=(), strategy="direct",
               name="gru", cfg=None, seed=0, device="cpu", curves=False, predict=True):
    import torch
    import torch.nn as nn
    cfg = cfg or cfgmodels.default_cfg(name)
    tr = cfg["train"]
    Z, y, ci, enc, dates = prepare(df, cutoff, target, covariates)
    out_len = H if strategy == "direct" else 1
    X, Y = windows(Z, ci, out_len)
    vend = int(np.where(pd.to_datetime(dates) == pd.Timestamp(VAL_END))[0][0])
    has_val = cutoff == MAIN_CUTOFF
    if has_val:
        Xv, Yv, Mv = val_pairs(Z, ci, vend, out_len)

    set_seed(seed)
    norm = cfg.get("norm", "global")
    patience = tr.get("patience")
    if patience and not has_val:
        raise ValueError("early stopping requires the validation period (main cutoff)")
    model = cfgmodels.build(name, out_len, cfg["arch"], in_ch=Z.shape[1], lookback=LOOKBACK, norm=norm).to(device)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(X, device=device), torch.tensor(Y, device=device)
    if has_val:
        Xvt, Yvt, Mvt = (torch.tensor(a, device=device) for a in (Xv, Yv, Mv))
    g = torch.Generator().manual_seed(seed)
    n, curve = len(Xt), []
    best_loss, best_ep, best_state, epochs_run = np.inf, 0, None, 0
    t0 = time.time()
    for ep in range(1, tr["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, tr["batch"]):
            b = perm[i:i + tr["batch"]].to(device)
            opt.zero_grad()
            loss = mse(model(Xt[b]), Yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += float(loss.item()) * len(b)
        epochs_run = ep
        vloss = np.nan
        if has_val and (curves or patience):
            model.eval()
            with torch.no_grad():
                e = (model(Xvt) - Yvt) ** 2 * Mvt
                vloss = float(e.sum() / Mvt.sum())
        if curves:
            row = {"epoch": ep, "train_loss": tot / n}
            if has_val:
                row["val_loss"] = vloss
            curve.append(row)
        if patience:
            if vloss < best_loss:
                best_loss, best_ep, best_state = vloss, ep, copy.deepcopy(model.state_dict())
            elif ep - best_ep >= patience:
                break
    if patience and best_state is not None:
        model.load_state_dict(best_state)
    secs = time.time() - t0

    # validation score for model selection: sMAPE over all validation (origin, h) pairs
    val_smape = np.nan
    if has_val:
        yy, yh = [], []
        for t in range(ci + 1, vend + 1):
            p = enc.decode(forecast_path(model, Z[:t], strategy, device))
            for k in range(H):
                if t + k <= vend:
                    yy.append(y[t + k]); yh.append(p[k])
        val_smape = smape(yy, yh)

    rows = []
    if predict:
        for t in range(ci + 1, len(y)):
            p = enc.decode(forecast_path(model, Z[:t], strategy, device))
            for h in range(1, H + 1):
                k = t + h - 1
                if k >= len(y):
                    break
                rows.append({"cutoff": pd.Timestamp(cutoff).strftime("%Y-%m"), "strategy": strategy, "model": name,
                             "seed": seed, "origin": pd.Timestamp(dates[t]).strftime("%Y-%m"),
                             "origin_period": period_label(dates[t]) if cutoff == MAIN_CUTOFF else "backtest",
                             "h": h, "target": pd.Timestamp(dates[k]).strftime("%Y-%m"),
                             "y": y[k], "yhat": float(p[h - 1])})
    log = {"cutoff": pd.Timestamp(cutoff).strftime("%Y-%m"), "strategy": strategy, "model": name, "seed": seed,
           "target": target, "covariates": "+".join(covariates) or "none", "n_train_windows": len(X),
           "n_params": n_params, "train_seconds": round(secs, 2), "val_sMAPE": val_smape, "device": device,
           "norm": norm, "patience": patience or 0, "epochs_run": epochs_run,
           "best_epoch": best_ep if patience else epochs_run}
    return pd.DataFrame(rows), log, pd.DataFrame(curve)
