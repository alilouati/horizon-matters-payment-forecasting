"""
run_normalization_backtest.py — the per-window normalisation (RevIN) with the DEFAULT architectures and the
fixed training budget (no early stopping, no tuning, hence no use of later data for any choice), retrained
for the four additional training cutoffs (December 2018-2021), 10 seeds; canonical PatchTST (built-in RevIN)
alongside. Forecasts in runs_r3/revin_fixed/<cutoff>/ and runs_r3/global_fixed/<cutoff>/.
Run in Spyder (press Run) or: python run_normalization_backtest.py   -> logs/progress_bt_r3.log, logs/DONE_BT_R3
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__" and "--worker" not in sys.argv:
    (HERE / "logs").mkdir(exist_ok=True)
    out = open(HERE / "logs" / "progress_bt_r3.log", "a")
    p = subprocess.Popen([sys.executable, str(HERE / "run_normalization_backtest.py"), "--worker"], cwd=HERE,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Normalization backtest started in the background (pid {p.pid}).")
    sys.exit(0)

import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

CUTOFFS = ["2018-12-01", "2019-12-01", "2020-12-01", "2021-12-01"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    return cmd, r.returncode, r.stderr[-2000:]


jobs = []
for c in CUTOFFS:
    for s in ["direct", "recursive"]:
        for m in ["lstm", "gru", "tcn", "nbeats", "patchtst", "nbeats_c"]:
            for s0 in (0, 5):
                jobs.append(["09_run_configurable.py", "--runs-dir", "runs_r3/revin_fixed", "--cutoff", c, "--strategies", s,
                             "--models", m, "--norm", "revin", "--patience", "0", "--seeds", "5", "--seed-start", str(s0),
                             "--threads", "1"])
        for s0 in (0, 5):
            jobs.append(["09_run_configurable.py", "--runs-dir", "runs_r3/global_fixed", "--cutoff", c, "--strategies", s,
                         "--models", "patchtst_c", "--norm", "global", "--patience", "0", "--seeds", "5",
                         "--seed-start", str(s0), "--threads", "1"])
jobs.sort(key=lambda j: 0 if ("patchtst" in j or "nbeats_c" in j) else 1)
workers = max(1, min(8, (os.cpu_count() or 2) - 2))
log(f"backtest: {len(jobs)} jobs on {workers} workers")
t0 = time.time()
with ThreadPoolExecutor(workers) as ex:
    for i, f in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
        cmd, rc, se = f.result()
        if rc:
            log(f"FAILED {' '.join(cmd)}\n{se}")
        elif i % 10 == 0 or i == len(jobs):
            log(f"[{i}/{len(jobs)}] ({(time.time() - t0) / 60:.1f} min)")
bundle = HERE / "results_bt_r3_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for v in ["revin_fixed", "global_fixed"]:
        for p in (HERE / "runs_r3" / v).rglob("*"):
            if p.is_file() and "2022-12" not in str(p.relative_to(HERE / "runs_r3" / v)).split("/")[0]:
                z.write(p, p.relative_to(HERE))
log(f"bundle written {bundle}")
(HERE / "logs" / "DONE_BT_R3").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
