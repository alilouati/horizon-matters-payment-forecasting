"""
reproduce_results.py — regenerate every result table and figure of the manuscript from the source data and
the stored forecasts, without retraining any model.

    python reproduce_results.py            # full regeneration (about 5-10 minutes on a laptop)
    python reproduce_results.py --check    # additionally compare the regenerated LaTeX tables with
                                           # the copies shipped in manuscript_assets_reference/

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
 12  15_dm_tests_smape.py        -> results_r3/R4_dm_smape.csv, manuscript_assets/tab_gru_tcn.tex,
                                    tab_dm_classical.tex                  (Tables 9 and 11, all DM tests)
Training the models is done by run_all.py, run_additional.py, run_normalization.py and
run_normalization_backtest.py (see README.md); this script does not need PyTorch.
"""
import argparse
import filecmp
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
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare regenerated tables with manuscript_assets_reference/")
    ap.add_argument("--skip", nargs="*", default=[], help="script names to skip (e.g. 02_anomaly_screening.py)")
    a = ap.parse_args()
    t0 = time.time()
    for step in STEPS:
        if step[0] in a.skip:
            print(f"--  skipped {' '.join(step)}")
            continue
        t = time.time()
        r = subprocess.run([sys.executable] + step, cwd=HERE, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"{status:6s} {' '.join(step)} ({time.time() - t:.0f} s)", flush=True)
        if r.returncode:
            print(r.stderr[-3000:])
            sys.exit(1)
    print(f"all steps finished in {(time.time() - t0) / 60:.1f} min")
    if a.check:
        ref, new = HERE / "manuscript_assets_reference", HERE / "manuscript_assets"
        tex = sorted(p.name for p in ref.glob("*.tex"))
        diff = [n for n in tex if not filecmp.cmp(ref / n, new / n, shallow=False)]
        print(f"LaTeX tables compared: {len(tex)}; identical: {len(tex) - len(diff)}; different: {diff}")
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
