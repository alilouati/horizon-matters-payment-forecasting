"""
08_hpo.py — architecture-specific hyperparameter search (additional experiments, group B).

Equal budget for every architecture and strategy: random search with N trials
(trial 0 = the default configuration), each trial trained with S seeds on the
training period only. Selection criterion: mean validation sMAPE over all
(origin, horizon) pairs whose target lies in the validation period (2023).
The test period is never used for selection.

  python 08_hpo.py --strategy direct --model gru --trials 20 --seeds 3     # run (resumable)
  python 08_hpo.py --summarize                                             # -> results_r2/hpo_*.csv/json

Search for the normalization experiments (--space r3): same budget, validation early stopping always on, normalisation
(global / RevIN / last-value anchoring) searched, canonical N-BEATS and PatchTST included.
  python 08_hpo.py --space r3 --strategy direct --model patchtst_c --trial 3
  python 08_hpo.py --space r3 --summarize                                  # -> results_r3/hpo_*_r3.csv/json
"""
import argparse
import json

import pandas as pd

from common import ROOT, load_series
import cfgmodels

HPO_DIR = ROOT / "runs_hpo"
OUT = ROOT / "results_r2"
SFX = ""
SAMPLER = cfgmodels.sample_cfgs


def use_r3():
    global HPO_DIR, OUT, SFX, SAMPLER
    HPO_DIR, OUT, SFX, SAMPLER = ROOT / "runs_hpo_r3", ROOT / "results_r3", "_r3", cfgmodels.sample_cfgs_r3


def run(strategy, model, trials, seeds, threads, only=None):
    import torch
    torch.set_num_threads(threads)
    from engine import train_eval
    df = load_series()
    for k, cfg in enumerate(SAMPLER(model, strategy, trials)):
        if only is not None and k != only:
            continue
        f = HPO_DIR / strategy / model / f"trial{k:02d}.csv"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in range(seeds):
            _, log, _ = train_eval(df, strategy=strategy, name=model, cfg=cfg, seed=1000 + s, predict=False)
            rows.append({"strategy": strategy, "model": model, "trial": k, "seed": 1000 + s,
                         "cfg": json.dumps(cfg), "val_sMAPE": log["val_sMAPE"], "n_params": log["n_params"],
                         "train_seconds": log["train_seconds"], "best_epoch": log["best_epoch"]})
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"{strategy} {model} trial {k}: val sMAPE {pd.DataFrame(rows).val_sMAPE.mean():.2f}", flush=True)


def summarize():
    OUT.mkdir(exist_ok=True)
    files = list(HPO_DIR.rglob("trial*.csv"))
    d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    d.to_csv(OUT / f"hpo_all_runs{SFX}.csv", index=False)
    t = (d.groupby(["strategy", "model", "trial"])
           .agg(cfg=("cfg", "first"), val_sMAPE_mean=("val_sMAPE", "mean"), val_sMAPE_sd=("val_sMAPE", "std"),
                n_params=("n_params", "first"), train_seconds=("train_seconds", "mean"), n_seeds=("seed", "nunique"),
                best_epoch=("best_epoch", "mean") if "best_epoch" in d else ("seed", "size"))
           .reset_index())
    t.to_csv(OUT / f"hpo_trials{SFX}.csv", index=False)
    best = {}
    for (s, m), g in t.groupby(["strategy", "model"]):
        b = g.sort_values("val_sMAPE_mean").iloc[0]
        best[f"{m}-{s}"] = {"cfg": json.loads(b.cfg), "trial": int(b.trial), "val_sMAPE": float(b.val_sMAPE_mean),
                            "default_val_sMAPE": float(g[g.trial == 0].val_sMAPE_mean.iloc[0]),
                            "n_params": int(b.n_params), "n_trials": int(g.trial.nunique())}
    (OUT / f"hpo_best{SFX}.json").write_text(json.dumps(best, indent=1))
    print(json.dumps({k: (v["trial"], round(v["default_val_sMAPE"], 2), round(v["val_sMAPE"], 2)) for k, v in best.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy")
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--trial", type=int, default=None, help="run only this trial index")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--space", choices=["r2", "r3"], default="r2")
    a = ap.parse_args()
    if a.space == "r3":
        use_r3()
    if a.summarize:
        summarize()
    else:
        run(a.strategy, a.model, a.trials, a.seeds, a.threads, a.trial)
