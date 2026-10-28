"""
run_normalization.py — one-click launcher for the normalization experiments (group C) (open in Spyder and press Run,
or: python run_normalization.py). Runs in the background; progress in logs/progress_r3.log,
completion marker logs/DONE_R3, results in results_r3_bundle.zip. Resumable.

  0. smoke test of every new component (3 epochs)
  1. normalisation x early-stopping factorial with the default architectures, 10 seeds:
       norm in {global, revin, anchor} x {fixed 250 epochs, validation early stopping}
       (global/fixed = the main experiment, seeds 0-9, not re-run for the five original models)
     plus canonical N-BEATS (all six cells) and canonical PatchTST (built-in RevIN; fixed / early stopping)
  2. equal-budget hyperparameter search with the fixes (08_hpo.py --space r3):
       7 architectures x 2 strategies x 20 trials x 3 seeds, early stopping always on,
       normalisation searched, selection on validation sMAPE only
  3. selected configurations, 10 seeds
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__" and "--worker" not in sys.argv:
    (HERE / "logs").mkdir(exist_ok=True)
    out = open(HERE / "logs" / "progress_r3.log", "a")
    p = subprocess.Popen([sys.executable, str(HERE / "run_normalization.py"), "--worker"], cwd=HERE,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Normalization experiments started in the background (pid {p.pid}).")
    sys.exit(0)

import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

OLD = ["lstm", "gru", "tcn", "nbeats", "patchtst"]
CANON = ["nbeats_c", "patchtst_c"]
STRATS = ["direct", "recursive"]
TRIALS, HPO_SEEDS, SEEDS = 20, 3, 10
VARIANTS = {"global_fixed": ("global", 0), "global_es": ("global", 25), "revin_fixed": ("revin", 0),
            "revin_es": ("revin", 25), "anchor_fixed": ("anchor", 0), "anchor_es": ("anchor", 25)}

SMOKE = r'''
import itertools, cfgmodels
from common import load_series
from engine import train_eval
df = load_series()
for name, norm, strat in itertools.product(["lstm","gru","tcn","nbeats","patchtst","nbeats_c","patchtst_c"],
                                           ["global","revin","anchor"], ["direct","recursive"]):
    if name == "patchtst_c" and norm != "global":
        continue
    cfg = cfgmodels.default_cfg(name); cfg["norm"] = norm
    cfg["train"]["epochs"] = 3; cfg["train"]["patience"] = 2
    p, log, _ = train_eval(df, strategy=strat, name=name, cfg=cfg, seed=0)
    assert len(p) > 0 and p.yhat.notna().all(), (name, norm, strat)
    print(name, norm, strat, log["n_params"], round(log["val_sMAPE"], 2), log["best_epoch"], flush=True)
cfg = cfgmodels.default_cfg("nbeats_c"); cfg["arch"]["basis"] = "interpretable"; cfg["train"]["epochs"] = 3
for strat in ["direct", "recursive"]:
    p, log, _ = train_eval(df, strategy=strat, name="nbeats_c", cfg=cfg, seed=0)
    print("nbeats_c interpretable", strat, log["n_params"], round(log["val_sMAPE"], 2), flush=True)
print("SMOKE OK")
'''


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    return cmd, r.returncode, r.stdout[-3000:], r.stderr[-3000:]


def pool(jobs, workers, label):
    t0, failed = time.time(), 0
    log(f"{label}: {len(jobs)} jobs on {workers} workers")
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            cmd, rc, so, se = f.result()
            if rc != 0:
                failed += 1
                log(f"[{label} {i}/{len(jobs)}] FAILED {' '.join(cmd)}\n{se}")
            elif i % 10 == 0 or i == len(jobs):
                log(f"[{label} {i}/{len(jobs)}] ({(time.time() - t0) / 60:.1f} min)")
    return failed


def heavy(c):
    return 0 if any(k in c for k in ("patchtst", "patchtst_c", "nbeats_c")) else 1


def main():
    log("=" * 70)
    import torch
    log(f"python {sys.version.split()[0]} | torch {torch.__version__} | cores {os.cpu_count()}")
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    if not (HERE / "data" / "monthly_series.csv").exists():
        for step in (["01_prepare_data.py"], ["02_anomaly_screening.py"]):
            cmd, rc, so, se = run(step)
            log(f"{step[0]} rc={rc} {se[-300:]}")
            if rc:
                return

    cmd, rc, so, se = run(["-c", SMOKE])
    log(f"smoke test rc={rc}\n{so}\n{se[-2500:]}")
    if rc:
        return

    T = ["--threads", "1"]
    jobs = []
    # 1. factorial with default architectures + canonical models
    for v, (norm, pat) in VARIANTS.items():
        models = (OLD if v != "global_fixed" else []) + (CANON if norm == "global" else ["nbeats_c"])
        for s in STRATS:
            for m in models:
                for s0 in (0, 5):
                    jobs.append(["09_run_configurable.py", "--runs-dir", f"runs_r3/{v}", "--strategies", s,
                                 "--models", m, "--norm", norm, "--patience", str(pat), "--seeds", "5",
                                 "--seed-start", str(s0)] + T)
    # 2. hyperparameter search with the fixes
    for s in STRATS:
        for m in OLD + CANON:
            for k in range(TRIALS):
                jobs.append(["08_hpo.py", "--space", "r3", "--strategy", s, "--model", m, "--trials", str(TRIALS),
                             "--seeds", str(HPO_SEEDS), "--trial", str(k)] + T)
    jobs.sort(key=heavy)
    pool(jobs, workers, "stage A")

    cmd, rc, so, se = run(["08_hpo.py", "--space", "r3", "--summarize"])
    log(f"HPO r3 summary rc={rc}\n{so}\n{se[-1000:]}")
    if rc:
        return

    # 3. selected configurations, 10 seeds, with learning curves
    jobs = [["09_run_configurable.py", "--runs-dir", "runs_r3/tuned_r3", "--config", "results_r3/hpo_best_r3.json",
             "--strategies", s, "--models", m, "--seeds", "5", "--seed-start", str(s0), "--curves"] + T
            for s in STRATS for m in OLD + CANON for s0 in (0, 5)]
    jobs.sort(key=heavy)
    pool(jobs, workers, "stage B")

    bundle = HERE / "results_r3_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["results_r3", "runs_r3", "runs_hpo_r3", "logs"]:
            for p in (HERE / folder).rglob("*"):
                if p.is_file() and p.suffix != ".zip":
                    z.write(p, p.relative_to(HERE))
    log(f"bundle written: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    (HERE / "logs" / "DONE_R3").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FATAL:\n" + traceback.format_exc())
