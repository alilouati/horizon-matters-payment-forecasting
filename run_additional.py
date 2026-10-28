"""
run_additional.py — one-click launcher for the additional experiments (group B) (open in Spyder and press Run,
or: python run_additional.py). Runs in the background; progress in logs/progress_r2.log,
completion marker logs/DONE_R2, results in results_r2_bundle.zip. Resumable.

  1. hyperparameter search (08_hpo.py): 5 models x 2 strategies x 20 trials x 3 seeds, validation only
  2. tuned configurations: 20 seeds each, with learning curves (09_run_configurable.py)
  3. default configurations: learning curves, 5 seeds
  4. second series (POS transaction counts): default configurations, 10 seeds + classical baselines
  5. multivariate inputs (sales + transactions + terminals), direct strategy, 20 seeds
  6. analysis of every experiment (05_analyze.py)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__" and "--worker" not in sys.argv:
    (HERE / "logs").mkdir(exist_ok=True)
    out = open(HERE / "logs" / "progress_r2.log", "a")
    p = subprocess.Popen([sys.executable, str(HERE / "run_additional.py"), "--worker"], cwd=HERE,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Additional experiments started in the background (pid {p.pid}).")
    sys.exit(0)

import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS = ["lstm", "gru", "tcn", "nbeats", "patchtst"]
STRATS = ["direct", "recursive"]
TRIALS, HPO_SEEDS = 20, 3


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    return cmd, r.returncode, r.stdout[-2000:], r.stderr[-3000:]


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


def main():
    log("=" * 70)
    import torch
    log(f"python {sys.version.split()[0]} | torch {torch.__version__} | cores {os.cpu_count()}")
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    for step in (["01_prepare_data.py"], ["02_anomaly_screening.py"]):
        cmd, rc, so, se = run(step)
        log(f"{step[0]} rc={rc} {se[-300:]}")
        if rc:
            return

    # smoke test of the configurable engine
    cmd, rc, so, se = run(["09_run_configurable.py", "--runs-dir", "runs_smoke", "--models", "gru", "patchtst",
                           "--strategies", "direct", "--seeds", "1", "--curves"])
    log(f"smoke test rc={rc}\n{so}\n{se[-1500:]}")
    if rc:
        return

    T = ["--threads", "1"]
    # stage A: hyperparameter search + experiments that do not depend on it
    jobs = [["08_hpo.py", "--strategy", s, "--model", m, "--trials", str(TRIALS), "--seeds", str(HPO_SEEDS),
             "--trial", str(k)] + T for s in STRATS for m in MODELS for k in range(TRIALS)]
    jobs.sort(key=lambda c: 0 if "patchtst" in c else 1)
    for s in STRATS:
        for m in MODELS:
            jobs.append(["09_run_configurable.py", "--runs-dir", "runs_curves_default", "--strategies", s,
                         "--models", m, "--seeds", "5", "--curves"] + T)
            for s0 in (0, 5):
                jobs.append(["09_run_configurable.py", "--runs-dir", "runs_tx", "--target", "pos_transactions",
                             "--strategies", s, "--models", m, "--seeds", "5", "--seed-start", str(s0)] + T)
    for m in MODELS:
        for s0 in range(0, 20, 5):
            jobs.append(["09_run_configurable.py", "--runs-dir", "runs_mv", "--covariates", "pos_transactions",
                         "pos_terminals", "--strategies", "direct", "--models", m, "--seeds", "5",
                         "--seed-start", str(s0)] + T)
    jobs.append(["04_run_classical.py", "--target", "pos_transactions", "--runs-dir", "runs_tx"])
    pool(jobs, workers, "stage A")

    cmd, rc, so, se = run(["08_hpo.py", "--summarize"])
    log(f"HPO summary rc={rc}\n{so}\n{se[-1000:]}")
    if rc:
        return

    # stage B: tuned configurations, 20 seeds, with learning curves
    jobs = [["09_run_configurable.py", "--runs-dir", "runs_tuned", "--config", "results_r2/hpo_best.json",
             "--strategies", s, "--models", m, "--seeds", "5", "--seed-start", str(s0), "--curves"] + T
            for s in STRATS for m in MODELS for s0 in range(0, 20, 5)]
    jobs.sort(key=lambda c: 0 if "patchtst" in c else 1)
    pool(jobs, workers, "stage B")

    # classical baselines next to the tuned and multivariate runs (same forecasts as the main experiment)
    for d in ["runs_tuned", "runs_mv"]:
        src, dst = HERE / "runs" / "2022-12" / "classical", HERE / d / "2022-12" / "classical"
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)

    for args in (["--runs-dir", "runs_tuned", "--results-dir", "results_tuned"],
                 ["--runs-dir", "runs_mv", "--results-dir", "results_mv"],
                 ["--runs-dir", "runs_tx", "--results-dir", "results_tx", "--target", "pos_transactions"]):
        cmd, rc, so, se = run(["05_analyze.py"] + args)
        log(f"05_analyze {' '.join(args)} rc={rc} {se[-500:]}")

    bundle = HERE / "results_r2_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["results_r2", "results_tuned", "results_mv", "results_tx", "runs_hpo", "runs_tuned",
                       "runs_mv", "runs_tx", "runs_curves_default", "logs"]:
            for p in (HERE / folder).rglob("*"):
                if p.is_file() and p.suffix != ".zip":
                    z.write(p, p.relative_to(HERE))
    log(f"bundle written: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    (HERE / "logs" / "DONE_R2").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FATAL:\n" + traceback.format_exc())
