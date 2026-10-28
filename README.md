# Reproduction package: deep multi-horizon forecasting of Saudi POS sales

This package contains the code, input data, stored forecasts and run logs behind every number, table and figure
of the manuscript. `INVENTORY.md` lists every file and maps each script to the tables and figures it produces.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (`pip install -r requirements.txt`). PyTorch is needed only to
retrain the networks and statsmodels only for the classical baselines and the anomaly screening; regenerating the
tables and figures from the stored forecasts needs NumPy, pandas, SciPy and Matplotlib only. A GPU is not needed.

## Regenerating the tables and figures (no training)

```bash
python reproduce_results.py            # about 5 minutes on a laptop CPU
python reproduce_results.py --check    # also compares the regenerated LaTeX tables with manuscript_assets_reference/
python reproduce_results.py --skip 02_anomaly_screening.py   # if statsmodels is not installed (uses data/anomaly_flags.csv)
```

The script runs the analysis steps in order (01, 02, 05, 08 `--summarize`, 10, 06, 07, 11, 12, 13, 14, 15) and
writes `results*/` and `manuscript_assets/`. Every step reads only the input data, the stored forecasts in
`runs*/` or the outputs of earlier steps.

## Retraining all models

```bash
python run_all.py                       # data, main experiment (680 neural runs, 5 cutoffs), classical baselines, analysis
python run_additional.py                # first hyperparameter search (600 runs) and 450 training runs
python run_normalization.py             # normalization x early-stopping factorial, canonical models, second search (840 runs)
python run_normalization_backtest.py    # RevIN with a fixed budget at the four earlier cutoffs
python reproduce_results.py
```

Each launcher runs in the background (open it in Spyder and press Run, or call it from a terminal), logs its
progress in `logs/`, skips runs that already exist and can be restarted. The reported results were produced on
macOS (Apple silicon, CPU, one thread per run) with Python 3.11, PyTorch 2.10, statsmodels 0.14, NumPy 1.24 and
pandas 2.1; the main experiment took about 25 minutes on 8 parallel workers.

## Data

`data/pos-transactions.csv` is the complete SAMA export (4,689 records: monthly, quarterly and annual values for
nine indicators) and the only external input. The monthly Total POS Sales series runs from January 1995 to July
2025 (367 months); sales are reported in thousand SAR and converted to SAR. Partition: training 1995-01 to 2022-12
(336 months), validation 2023 (12), test 2024-01 to 2025-07 (19).

## Design notes

**Forecasts.** `runs/<cutoff>/<strategy>/<model>/seedNN.csv` holds one row per (origin, horizon) with the realised
value and the forecast. The network weights are fixed after training at the cutoff; at every later origin the
network receives the 24 preceding observed months and returns 12 forecasts. The classical models are re-estimated at
every origin on all earlier observations and evaluated on the same (origin, h) pairs.

**Strategies.** `direct`: one network outputs the 12-month vector (301 training windows). `recursive`: the same
architecture with one output (312 one-step windows), iterated over 12 steps.

**Models.** In the manuscript the code identifiers `nbeats` and `patchtst` are called the MLP and the simplified
patch Transformer; `nbeats_c` and `patchtst_c` are canonical N-BEATS and canonical PatchTST.

**Seeds.** Python, NumPy and PyTorch seeds and the mini-batch order are fixed per run; deterministic algorithms are
requested. Results are reported as distributions over seeds.

**Statistical tests.** `15_dm_tests_smape.py` computes all Diebold–Mariano tests reported in the manuscript with the
symmetric absolute percentage error as loss (its mean is the sMAPE), the Harvey–Leybourne–Newbold correction and
Newey–West lag h−1 (capped at n−2), with Holm-adjusted p-values within each family of tests; the squared-error
version is reported alongside as a robustness check (`tab_*_se.tex`, columns `*_se` of `R4_dm_smape.csv`).
`05_analyze.py` (`results/T4_dm_tests.csv`) keeps the squared-error tests. Bootstrap intervals use the circular
moving-block bootstrap (`10_bootstrap.py`, `14_normalization_bootstrap.py`).

**Original scripts.** The original exploratory scripts trained unseeded models and did not save per-origin
forecasts; they are superseded by this package and are not needed to reproduce the results.

## License and data source

The code is released under the MIT License (`LICENSE`). The raw data in `data/pos-transactions.csv` are public
payment statistics of the Saudi Central Bank (SAMA), Monthly Statistical Bulletin; please cite SAMA as the source
when reusing them. The derived series, stored forecasts and logs were produced with the code in this repository.

## Citation

If you use this code or the stored forecasts, please cite: Louati, A.; Louati, H.; Kariri, E. When the Horizon
Matters for Deep Forecasting of Payment Time Series (manuscript under review).
