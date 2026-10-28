"""
run_all.py — one-click launcher for the complete experiment.

Open this file in Spyder and press Run (or: python run_all.py).
It starts the whole pipeline in a BACKGROUND process and returns immediately,
so Spyder stays usable and closing Spyder does not stop the experiment.

Progress:   logs/progress.log   (updated continuously)
Finished:   logs/DONE           (created at the end)
Results:    results_bundle.zip  (everything needed for the manuscript)

Running it again resumes: finished runs are skipped.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__" and "--worker" not in sys.argv:
    (HERE / "logs").mkdir(exist_ok=True)
    out = open(HERE / "logs" / "progress.log", "a")
    p = subprocess.Popen([sys.executable, str(HERE / "run_all.py"), "--worker"], cwd=HERE,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Experiment started in the background (pid {p.pid}).")
    print(f"Progress: {HERE / 'logs' / 'progress.log'}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# background worker
# ---------------------------------------------------------------------------
import platform
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    return cmd, r.returncode, r.stdout[-3000:], r.stderr[-3000:]


def ensure(pkg, pip_name=None):
    try:
        __import__(pkg)
        return True
    except ImportError:
        log(f"{pkg} not found, installing {pip_name or pkg} ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", pip_name or pkg], capture_output=True, text=True)
        log(r.stdout[-500:] + r.stderr[-500:])
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False


def main():
    log("=" * 70)
    log(f"python {sys.version.split()[0]} at {sys.executable}")
    log(f"platform {platform.platform()}  cores {os.cpu_count()}")
    for pkg, pipn in [("numpy", None), ("pandas", None), ("scipy", None), ("matplotlib", None),
                      ("statsmodels", None), ("torch", None)]:
        if not ensure(pkg, pipn):
            log(f"FATAL: could not import or install {pkg}")
            return
    import torch, statsmodels, numpy, pandas
    log(f"torch {torch.__version__} | statsmodels {statsmodels.__version__} | numpy {numpy.__version__} | pandas {pandas.__version__}")

    for step in (["01_prepare_data.py"], ["02_anomaly_screening.py"]):
        cmd, rc, so, se = run(step)
        log(f"{step[0]} rc={rc}\n{so}\n{se}")
        if rc != 0:
            log("FATAL: stopping")
            return

    # quick smoke test before the long run
    t0 = time.time()
    cmd, rc, so, se = run(["03_run_neural.py", "--seeds", "1", "--seed-start", "0", "--models", "gru", "tcn",
                           "nbeats", "lstm", "patchtst", "--strategies", "direct", "--device", "cpu", "--threads", "1"])
    log(f"smoke test rc={rc} ({time.time() - t0:.0f}s)\n{so}\n{se}")
    if rc != 0:
        log("FATAL: smoke test failed")
        return

    # job list: each job = one (cutoff set, strategy, model, seed range)
    jobs = []
    models = ["lstm", "gru", "tcn", "nbeats", "patchtst"]
    for strat in ["direct", "recursive"]:
        for m in models + ["tcn24", "patchtst_small"]:
            for s0 in range(0, 20, 5):                      # main cutoff, 20 seeds
                jobs.append(["03_run_neural.py", "--cutoffs", "main", "--strategies", strat, "--models", m,
                             "--seeds", "5", "--seed-start", str(s0), "--device", "cpu", "--threads", "1"])
        for m in models:
            for s0 in range(0, 10, 5):                      # 4 backtest cutoffs, 10 seeds
                jobs.append(["03_run_neural.py", "--cutoffs", "extra", "--strategies", strat, "--models", m,
                             "--seeds", "5", "--seed-start", str(s0), "--device", "cpu", "--threads", "1"])
    # put the slow PatchTST jobs first so they do not finish last
    jobs.sort(key=lambda c: 0 if "patchtst" in c else 1)
    jobs.append(["04_run_classical.py", "--cutoffs", "all"])

    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    log(f"{len(jobs)} jobs on {workers} parallel workers")
    t0 = time.time()
    failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            cmd, rc, so, se = f.result()
            tag = " ".join(cmd[1:8])
            if rc != 0:
                failed += 1
                log(f"[{i}/{len(jobs)}] FAILED {cmd}\n{se}")
            else:
                log(f"[{i}/{len(jobs)}] done {tag}  (elapsed {(time.time() - t0) / 60:.1f} min)")
    log(f"all jobs finished, {failed} failed")

    cmd, rc, so, se = run(["05_analyze.py"])
    log(f"05_analyze.py rc={rc}\n{so}\n{se}")

    bundle = HERE / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["results", "data", "logs"]:
            for p in (HERE / folder).rglob("*"):
                if p.is_file() and p.suffix != ".zip":
                    z.write(p, p.relative_to(HERE))
        for p in (HERE / "runs").rglob("*.csv"):           # all saved predictions
            z.write(p, p.relative_to(HERE))
    log(f"bundle written: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    (HERE / "logs" / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FATAL:\n" + traceback.format_exc())
