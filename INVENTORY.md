# Inventory of the reproduction package

The package has five kinds of content: Python code, configuration files, input data, stored forecasts and run
logs. Every result table and figure of the manuscript can be regenerated from the input data and the stored
forecasts with one command (`python reproduce_results.py`, no PyTorch needed, about 5 minutes on a laptop CPU).
Retraining the models from the input data is done with the four training launchers (Section 1c).

## 1. Python code (23 files)

### 1a. Shared modules (3)

| File | Role |
|---|---|
| `common.py` | All fixed settings (window L=24, horizon H=12, data split, epochs, learning rate, seeds), data loading, metrics |
| `cfgmodels.py` | Configurable versions of the seven architectures (LSTM, GRU, TCN, MLP, simplified patch Transformer, canonical N-BEATS, canonical PatchTST), the normalization wrappers (global, RevIN, last-value anchoring) and the two search spaces |
| `engine.py` | Common training and forecasting loop for `cfgmodels.py` (early stopping, learning curves, direct and recursive forecasting) |

### 1b. Numbered scripts (15)

Scripts 03, 04, 08 and 09 train or fit models and write stored forecasts; the other eleven only read stored
forecasts and data. Table and figure numbers refer to the manuscript.

| Script | Reads | Writes | Manuscript output |
|---|---|---|---|
| `01_prepare_data.py` | `data/pos-transactions.csv` | `data/monthly_series.csv`, `data/monthly_all_indicators.csv` | Section 3 data; Figure 2 |
| `02_anomaly_screening.py` (needs statsmodels) | `data/monthly_series.csv` | `data/anomaly_flags.csv` | flags listed in Table 15 |
| `03_run_neural.py` (needs PyTorch) | data | `runs/<cutoff>/<strategy>/<model>/seedNN.csv`, `runs/run_log.csv` | main experiment: 680 neural runs |
| `04_run_classical.py` (needs statsmodels) | data | `runs/<cutoff>/classical/<model>/seed00.csv` | Naive, Seasonal Naive, ETS, SARIMA |
| `05_analyze.py` | `runs*/` | `results*/T1..T12*.csv` | inputs to Tables 6-8, 12-14, 16-17 |
| `06_manuscript_assets.py` | `results/` | `manuscript_assets/tab_*.tex`, `fig_*.png` | Tables 6-8, 12-14, 16-17; Figures 13-18 |
| `07_method_figures.py` | — | `manuscript_assets/fig_method_*.png` | Figures 4-12 |
| `08_hpo.py` (search needs PyTorch; `--summarize` does not) | data / `runs_hpo*/` | `runs_hpo*/`, `results_r2/hpo_*`, `results_r3/hpo_*_r3*` | Tables 5, 18; Section 6.13 search |
| `09_run_configurable.py` (needs PyTorch) | data, configuration files | `runs_tuned/`, `runs_curves_default/`, `runs_tx/`, `runs_mv/`, `runs_r3/` | additional experiments |
| `10_bootstrap.py` | `runs/` | `results/T13_bootstrap.csv` | input to Table 10 |
| `11_additional_assets.py` | `results*/`, `runs*/` | `manuscript_assets/r2_*` | Tables 5, 10, 18-21; Figure 19 |
| `12_normalization_analysis.py` | `runs_r3/`, `runs/` | `results_r3/R3_*.csv`, `manuscript_assets/r3_*` | Tables 22-24; Figures 20-21 |
| `13_normalization_backtest.py` | `runs_r3/`, `runs/` | `results_r3/R3_backtest.csv`, `manuscript_assets/r3_backtest.tex` | Table 25 |
| `14_normalization_bootstrap.py` | `runs_r3/`, `runs/` | `results_r3/R3_bootstrap.csv` | bootstrap intervals, Section 6.13 |
| `15_dm_tests_smape.py` | `runs*/` | `results_r3/R4_dm_smape.csv`, `manuscript_assets/tab_gru_tcn.tex`, `tab_dm_classical.tex` | Tables 9 and 11; all DM p-values (sAPE loss, Holm-adjusted; squared-error loss as robustness) |

Figures 1 and 3 are drawn in LaTeX (TikZ) in the manuscript source; Figure 2 is a static excerpt of `data/monthly_series.csv`. Tables 1-4 are descriptive and written by hand; Table 15 lists the months flagged in `data/anomaly_flags.csv`.

### 1c. Launchers (5)

| File | What it runs | Needs PyTorch |
|---|---|---|
| `run_all.py` | 01-07: data, main experiment (680 neural runs + classical baselines at five cutoffs), analysis | yes |
| `run_additional.py` | 08-11: search (200 configurations x 3 seeds = 600 runs), 450 training runs (selected configurations, learning curves, second series, multivariate) | yes |
| `run_normalization.py` | normalization x early-stopping factorial, canonical models, search with the corrections (280 configurations x 3 seeds = 840 runs), selected configurations | yes |
| `run_normalization_backtest.py` | RevIN with the default architectures and a fixed budget at the four earlier cutoffs (Table 25) | yes |
| `reproduce_results.py` | every analysis step (01, 02, 05, 08 `--summarize`, 10, 06, 07, 11-15) from the stored forecasts; `--check` compares the regenerated LaTeX tables with `manuscript_assets_reference/` | no |

The training launchers write 1,360 neural training runs in `runs_r3/` and 450 in `runs_tuned/`,
`runs_curves_default/`, `runs_tx/` and `runs_mv/`; with the 680 runs of the main experiment and the 1,440 search
runs this gives all runs reported in the manuscript.

## 2. Configuration files (4)

| File | Content |
|---|---|
| `common.py` | fixed settings of every experiment (see above); no other file defines a default hyperparameter |
| `cfgmodels.py` | default configurations (identical to the main experiment) and both search spaces |
| `results_r2/hpo_best.json` | configurations selected on validation sMAPE by the first search (Table 18) |
| `results_r3/hpo_best_r3.json` | configurations selected by the search with normalization and early stopping (Table 24) |

The two JSON files are regenerated by `08_hpo.py --summarize` from the stored search runs (identical up to the
last floating-point digit) and are read by `09_run_configurable.py --config`.

## 3. Input data (4 files in `data/`)

| File | Content |
|---|---|
| `pos-transactions.csv` | complete SAMA export (4,689 records; monthly, quarterly and annual values of nine indicators). The only external input |
| `monthly_series.csv` | monthly Total POS Sales in SAR, 1995-01 to 2025-07 (367 months), with the train/validation/test label (from 01) |
| `monthly_all_indicators.csv` | all nine monthly indicators, incl. POS transactions and terminals (second series and multivariate experiment; from 01) |
| `anomaly_flags.csv` | Hampel and robust-STL flags (from 02; included because statsmodels may not be available) |

## 4. Stored forecasts (2,522 CSV files) and search logs (480 CSV files)

Every forecast of every training run is stored: one row per (origin, horizon) with columns
`cutoff, strategy, model, seed, origin, origin_period (val/test), h, target, y, yhat`.

| Folder | Files | Content |
|---|---|---|
| `runs/` | 680 + 20 | main experiment: `2022-12/` (5 models + `tcn24`, `patchtst_small`; 2 strategies; 20 seeds) and `2018-12/` ... `2021-12/` (5 models, 2 strategies, 10 seeds); classical baselines at each cutoff |
| `runs_hpo/` | 200 | search log of the first search: one file per configuration, with the configuration and the validation sMAPE of each of its 3 seeds |
| `runs_tuned/` | 200 + 4 | selected configurations of the first search, 20 seeds |
| `runs_curves_default/` | 50 | default configurations retrained with learning curves (5 models, 2 strategies, 5 seeds) |
| `runs_tx/` | 100 + 4 | second series (POS transaction counts), neural and classical |
| `runs_mv/` | 100 + 4 | multivariate inputs (sales, transactions, terminals) |
| `runs_hpo_r3/` | 280 | search log of the search with the corrections (as `runs_hpo/`) |
| `runs_r3/<variant>/` | 1,360 | normalization experiments: `{global,revin,anchor}_{fixed,es}` factorial (10 seeds), `tuned_r3` (selected configurations, 10 seeds), and `revin_fixed/`, `global_fixed/` at the four earlier cutoffs (Table 25) |

The folder suffixes `_r2`/`_r3` mark the second and third group of experiments: `results_r2/`, `runs_hpo/`,
`runs_tuned/`, `runs_curves_default/`, `runs_tx/` and `runs_mv/` belong to the additional experiments
(Sections 6.7-6.12); `results_r3/`, `runs_r3/` and `runs_hpo_r3/` to the normalization experiments (Section 6.13).

## 5. Run logs

| Location | Content |
|---|---|
| `runs/run_log.csv` | one row per neural run of the main experiment: trainable parameters, training windows, training time, final loss, device |
| `runs_tuned/_logs/`, `runs_tx/_logs/`, `runs_mv/_logs/`, `runs_curves_default/_logs/`, `runs_r3/*/_logs/` | one CSV file per run (1,810 files): configuration actually used (as JSON), trainable parameters, training windows, validation sMAPE, training time |
| `runs_tuned/_curves/`, `runs_curves_default/_curves/`, `runs_r3/tuned_r3/_curves/` | per-epoch training and validation loss, 390 files (Figure 19) |

## 6. Reference outputs

`manuscript_assets_reference/` holds the 22 LaTeX table files generated for the manuscript (20 used in the
manuscript and the squared-error versions of Tables 9 and 11). With `python reproduce_results.py --check`, all 22
regenerated files were byte-identical to these copies. The figures are regenerated in `manuscript_assets/`.
