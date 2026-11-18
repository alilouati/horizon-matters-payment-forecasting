"""
reproduce_results.py — regenerate every result table and figure of the manuscript from the source data and
the stored forecasts, without retraining any model.

    python reproduce_results.py            # full regeneration (about 5-10 minutes on a laptop)
    python reproduce_results.py --check    # additionally compare the regenerated LaTeX tables with
                                           # the copies shipped in manuscript_assets_reference/
Every run writes audit/reproduction_log.txt: date, platform, Python and package versions, the duration and
status of each step, the result of the check, and the SHA-256 checksum of every generated file.

Steps (each step only reads files written by the steps before it, or the stored forecasts in runs*/):
  1  01_prepare_data.py          raw SAMA export -> data/monthly_series.csv, data/monthly_all_indicators.csv
  2  02_anomaly_screening.py     -> data/anomaly_flags.csv                                   (needs statsmodels)
  3  05_analyze.py               main experiment  -> results/T1..T11                (Tables 6-8, 12-14, 16-17)
  4  05_analyze.py               tuned / multivariate / second series -> results_tuned/, results_mv/, results_tx/
  4b 08_hpo.py --summarize       search logs runs_hpo*/ -> results_r2/hpo_best.json, results_r3/hpo_best_r3.json
                                 (the selected configurations; Tables 18 and 24, Section 6.13)
  5  10_bootstrap.py             -> results/T13_bootstrap.csv                        (input to Table 10)
  6  06_manuscript_assets.py     -> manuscript_assets/tab_*.tex and fig_*.png  (Tables 6-8, 12-14, 16-17, Figs 13-18)
  7  07_method_figures.py        -> manuscript_assets/fig_method_*.png                      (Figures 4-12)
  8  11_additional_assets.py     -> manuscript_assets/r2_*             (Tables 5, 10, 18-21, Figure 19)
  9  12_normalization_analysis.py -> results_r3/, manuscript_assets/r3_*  (Tables 22-24, Figures 20-21)
 10  13_normalization_backtest.py -> manuscript_assets/r3_backtest.tex                         (Table 25)
 11  14_normalization_bootstrap.py -> results_r3/R3_bootstrap.csv                     (Section 6.13)
 12  15_dm_tests_smape.py        -> results_r3/R4_dm_smape.csv, R4_dm_status_by_family_h.csv,
                                    manuscript_assets/tab_gru_tcn.tex, tab_dm_classical.tex
                                    (Tables 9 and 11, all DM tests, availability of each test, sensitivity)
 13  16_bootstrap_simultaneous.py -> results_r3/R5_bootstrap_simultaneous.csv, R5_origin_differences.csv
                                    (simultaneous intervals of Table 10 and Section 6.13, block-length
                                    sensitivity, origin-level loss differences)
Training the models is done by run_all.py, run_additional.py, run_normalization.py and
run_normalization_backtest.py (see README.md); this script does not need PyTorch.
"""
import argparse
import filecmp
import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ["01_prepare_data.py"],
    ["02_anomaly_screening.py"],
    ["05_analyze.py"],
    ["05_analyze.py", "--runs-dir", "runs_tuned", "--results-dir", "results_tuned"],
    ["05_analyze.py", "--runs-dir", "runs_mv", "--results-dir", "results_mv"],
    ["05_analyze.py", "--runs-dir", "runs_tx", "--results-dir", "results_tx", "--target", "pos_transactions"],
    ["08_hpo.py", "--summarize"],
    ["08_hpo.py", "--space", "r3", "--summarize"],
    ["10_bootstrap.py"],
    ["06_manuscript_assets.py"],
    ["07_method_figures.py"],
    ["11_additional_assets.py"],
    ["12_normalization_analysis.py"],
    ["13_normalization_backtest.py"],
    ["14_normalization_bootstrap.py"],
    ["15_dm_tests_smape.py"],
    ["16_bootstrap_simultaneous.py"],
]


def versions():
    out = []
    for m in ["numpy", "pandas", "scipy", "matplotlib", "statsmodels", "torch"]:
        try:
            out.append(f"{m} {__import__(m).__version__}")
        except Exception:
            out.append(f"{m} not installed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare regenerated tables with manuscript_assets_reference/")
    ap.add_argument("--skip", nargs="*", default=[], help="script names to skip (e.g. 02_anomaly_screening.py)")
    a = ap.parse_args()
    log = [f"reproduce_results.py run at {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
           f"platform: {platform.platform()}", f"python {sys.version.split()[0]}"] + versions() + [""]
    t0 = time.time()
    for step in STEPS:
        if step[0] in a.skip:
            print(f"--  skipped {' '.join(step)}")
            log.append(f"skipped  {' '.join(step)}")
            continue
        t = time.time()
        r = subprocess.run([sys.executable] + step, cwd=HERE, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAILED"
        line = f"{status:6s} {' '.join(step)} ({time.time() - t:.0f} s)"
        print(line, flush=True)
        log.append(line)
        if r.returncode:
            print(r.stderr[-3000:])
            log.append(r.stderr[-3000:])
            write_log(log)
            sys.exit(1)
    log.append(f"all steps finished in {(time.time() - t0) / 60:.1f} min")
    print(log[-1])
    code = 0
    if a.check:
        ref, new = HERE / "manuscript_assets_reference", HERE / "manuscript_assets"
        tex = sorted(p.name for p in ref.glob("*.tex"))
        diff = [n for n in tex if not filecmp.cmp(ref / n, new / n, shallow=False)]
        line = f"LaTeX tables compared: {len(tex)}; identical: {len(tex) - len(diff)}; different: {diff}"
        print(line)
        log.append(line)
        code = 1 if diff else 0
    log += ["", "SHA-256 of generated files:"]
    for d in ["results", "results_tuned", "results_mv", "results_tx", "results_r2", "results_r3", "manuscript_assets"]:
        for f in sorted((HERE / d).rglob("*")) if (HERE / d).exists() else []:
            if f.is_file():
                log.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(HERE)}")
    write_log(log)
    sys.exit(code)


def write_log(lines):
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "reproduction_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
