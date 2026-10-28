"""
09_run_configurable.py — seeded runs of configurable models with full prediction storage
(additional experiments: tuned configurations, learning curves, second series, multivariate).

  # tuned configurations (from 08_hpo.py), 20 seeds, with learning curves
  python 09_run_configurable.py --runs-dir runs_tuned --config results_r2/hpo_best.json --seeds 20 --curves
  # learning curves of the default configurations
  python 09_run_configurable.py --runs-dir runs_curves_default --seeds 5 --curves
  # second series (POS transaction counts)
  python 09_run_configurable.py --runs-dir runs_tx --target pos_transactions --seeds 10
  # multivariate inputs (direct strategy only)
  python 09_run_configurable.py --runs-dir runs_mv --covariates pos_transactions pos_terminals --strategies direct --seeds 20
  # normalization experiments: normalisation and validation early stopping (default architectures), canonical models
  python 09_run_configurable.py --runs-dir runs_r3/revin_es --norm revin --patience 25 --seeds 10
  python 09_run_configurable.py --runs-dir runs_r3/global_fixed --models nbeats_c patchtst_c --seeds 10
"""
import argparse
import json

import pandas as pd

from common import ROOT, load_series, MODEL_NAMES, MAIN_CUTOFF
import cfgmodels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--target", default="pos_sales_SAR")
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--strategies", nargs="+", default=["direct", "recursive"])
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--config", default="default", help="'default' or a JSON file from 08_hpo.py")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--norm", choices=["global", "revin", "anchor"], default=None,
                    help="override the normalisation of the configuration")
    ap.add_argument("--patience", type=int, default=None,
                    help="validation early stopping with this patience (0 = off); overrides the configuration")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cutoff", default=MAIN_CUTOFF, help="training cutoff (e.g. 2018-12-01); early stopping needs the main cutoff")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from engine import train_eval

    best = json.loads((ROOT / a.config).read_text()) if a.config != "default" else None
    runs = ROOT / a.runs_dir
    df = load_series()
    for s in a.strategies:
        for m in a.models:
            cfg = json.loads(json.dumps(best[f"{m}-{s}"]["cfg"])) if best else cfgmodels.default_cfg(m)
            if a.norm is not None:
                cfg["norm"] = a.norm
            if a.patience is not None:
                if a.patience > 0:
                    cfg["train"]["patience"] = a.patience
                else:
                    cfg["train"].pop("patience", None)
            for seed in range(a.seed_start, a.seed_start + a.seeds):
                ctag = pd.Timestamp(a.cutoff).strftime("%Y-%m")
                out = runs / ctag / s / m / f"seed{seed:02d}.csv"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                preds, log, curve = train_eval(df, cutoff=a.cutoff, target=a.target, covariates=a.covariates, strategy=s,
                                               name=m, cfg=cfg, seed=seed, curves=a.curves)
                log["config"] = json.dumps(cfg)
                (runs / "_logs").mkdir(exist_ok=True)
                pd.DataFrame([log]).to_csv(runs / "_logs" / f"{ctag}_{s}_{m}_seed{seed:02d}.csv", index=False)
                if a.curves:
                    (runs / "_curves").mkdir(exist_ok=True)
                    curve.assign(strategy=s, model=m, seed=seed).to_csv(
                        runs / "_curves" / f"{s}_{m}_seed{seed:02d}.csv", index=False)
                tmp = out.with_suffix(".tmp")
                preds.to_csv(tmp, index=False)
                tmp.replace(out)
                print(f"{a.runs_dir} {s} {m} seed {seed}: params {log['n_params']} val sMAPE {log['val_sMAPE']:.2f} "
                      f"({log['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
