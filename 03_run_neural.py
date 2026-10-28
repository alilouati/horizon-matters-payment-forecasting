"""
03_run_neural.py — train every (cutoff, strategy, model, seed) combination and save
ALL per-origin, per-horizon forecasts.

Strategies
  direct    : network outputs the 12-month vector jointly (manuscript protocol),
              trained on 301 windows for the main cutoff.
  recursive : network outputs one month; the 12-month path is produced by feeding
              predictions back as inputs. Trained on 312 one-step windows.

Training (identical for all models): Adam, lr 1e-3, batch 32 (shuffled), 250 epochs,
MSE loss on the encoded scale, gradient-norm clipping at 1.0. No dropout except the
PyTorch default inside the original PatchTST-style encoder layer (0.1).

Evaluation: weights are fixed after training. For EVERY month t after the cutoff,
the model receives the 24 observed months before t and returns 12 forecasts for
t .. t+11. Rows are written for every target month that exists in the data.
All downstream tables (rolling h=1, complete-horizon, bias, DM tests, sensitivity
analyses) are computed from these saved predictions by 05_analyze.py.

Output: runs/<cutoff>/<strategy>/<model>/seed<NN>.csv
        runs/_logs/*.csv    (parameters, training time, final loss for every run)

Usage examples
  python 03_run_neural.py                               # main cutoff, 5 models, 2 strategies, 20 seeds
  python 03_run_neural.py --seeds 30
  python 03_run_neural.py --cutoffs extra --seeds 10    # backtesting cutoffs
  python 03_run_neural.py --models tcn24 patchtst_small --seeds 20   # capacity / receptive-field variants
Finished runs are skipped, so the script can be interrupted and restarted.
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd

from common import (load_series, cutoff_index, TrainOnlyEncoder, make_windows, build_model,
                    count_parameters, period_label, RUNS_DIR, TARGET, LOOKBACK, H, EPOCHS,
                    BATCH_SIZE, LR, GRAD_CLIP, MAIN_CUTOFF, EXTRA_CUTOFFS, MODEL_NAMES, STRATEGIES)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(model, X, Y, seed, device):
    import torch
    import torch.nn as nn
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)
    g = torch.Generator().manual_seed(seed)
    n = len(Xt)
    model.train()
    epoch_loss = float("nan")
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE].to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), Yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss.item()) * len(b)
        epoch_loss = total / n
    return model, epoch_loss


def forecast_path(model, z_hist, strategy, device):
    """12-month forecast (encoded scale) from the last LOOKBACK observed values."""
    import torch
    model.eval()
    with torch.no_grad():
        if strategy == "direct":
            x = torch.tensor(z_hist[-LOOKBACK:], dtype=torch.float32, device=device).unsqueeze(0)
            return model(x).cpu().numpy().reshape(-1)[:H]
        hist = list(map(float, z_hist[-LOOKBACK:]))
        out = []
        for _ in range(H):
            x = torch.tensor(hist[-LOOKBACK:], dtype=torch.float32, device=device).unsqueeze(0)
            zhat = float(model(x).cpu().numpy().reshape(-1)[0])
            out.append(zhat)
            hist.append(zhat)
        return np.asarray(out)


def run_one(df, cutoff, strategy, name, seed, device):
    dates = df["period_start"].values
    y = df[TARGET].values.astype(float)
    ci = cutoff_index(dates, cutoff)
    enc = TrainOnlyEncoder(y[:ci + 1])
    z = enc.encode(y)

    out_len = H if strategy == "direct" else 1
    X, Y = make_windows(z, ci, LOOKBACK, out_len)

    set_seed(seed)
    model = build_model(name, out_len)
    n_params = count_parameters(model)
    t0 = time.time()
    model, final_loss = train(model, X, Y, seed, device)
    secs = time.time() - t0

    rows = []
    for t in range(ci + 1, len(y)):           # every origin after the cutoff
        yhat = enc.decode(forecast_path(model, z[:t], strategy, device))
        for h in range(1, H + 1):
            tgt = t + h - 1
            if tgt >= len(y):
                break
            rows.append({
                "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m"),
                "strategy": strategy, "model": name, "seed": seed,
                "origin": pd.Timestamp(dates[t]).strftime("%Y-%m"),
                "origin_period": period_label(dates[t]) if cutoff == MAIN_CUTOFF else "backtest",
                "h": h,
                "target": pd.Timestamp(dates[tgt]).strftime("%Y-%m"),
                "y": y[tgt], "yhat": float(yhat[h - 1]),
            })
    log = {"cutoff": pd.Timestamp(cutoff).strftime("%Y-%m"), "strategy": strategy, "model": name,
           "seed": seed, "n_train_windows": len(X), "n_params": n_params,
           "train_seconds": round(secs, 2), "final_train_loss": final_loss, "device": device}
    return pd.DataFrame(rows), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds (0..N-1)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--cutoffs", default="main", help="'main', 'extra', 'all', or explicit dates")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (optional)")
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.cutoffs == "main":
        cutoffs = [MAIN_CUTOFF]
    elif args.cutoffs == "extra":
        cutoffs = EXTRA_CUTOFFS
    elif args.cutoffs == "all":
        cutoffs = [MAIN_CUTOFF] + EXTRA_CUTOFFS
    else:
        cutoffs = args.cutoffs.split(",")

    df = load_series()
    RUNS_DIR.mkdir(exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    jobs = [(c, s, m, sd) for c in cutoffs for s in args.strategies for m in args.models for sd in seeds]
    print(f"{len(jobs)} runs on {device}")

    for i, (c, strat, name, seed) in enumerate(jobs, 1):
        out = RUNS_DIR / pd.Timestamp(c).strftime("%Y-%m") / strat / name / f"seed{seed:02d}.csv"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        preds, log = run_one(df, c, strat, name, seed, device)
        # write the per-run log first, then the predictions (the predictions file marks completion)
        log_dir = RUNS_DIR / "_logs"
        log_dir.mkdir(exist_ok=True)
        pd.DataFrame([log]).to_csv(log_dir / f"{log['cutoff']}_{strat}_{name}_seed{seed:02d}.csv", index=False)
        tmp = out.with_suffix(".tmp")
        preds.to_csv(tmp, index=False)
        tmp.replace(out)
        print(f"[{i}/{len(jobs)}] {log['cutoff']} {strat:9s} {name:14s} seed {seed:2d} "
              f"params {log['n_params']:>7d}  {log['train_seconds']:6.1f}s  loss {log['final_train_loss']:.5f}")


if __name__ == "__main__":
    main()
