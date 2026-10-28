"""
common.py — shared configuration and utilities for the POS-sales forecasting pipeline.

Every numerical setting reported in the manuscript is defined HERE and only here.
All other scripts import from this file, so the code and the paper cannot drift apart.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"          # one CSV of predictions per (cutoff, strategy, model, seed)
RESULTS_DIR = ROOT / "results"    # tables (CSV) and figures (PNG) for the manuscript

RAW_FILE = DATA_DIR / "pos-transactions.csv"         # SAMA export, 4,689 records
SERIES_FILE = DATA_DIR / "monthly_series.csv"        # produced by 01_prepare_data.py
WIDE_FILE = DATA_DIR / "monthly_all_indicators.csv"  # produced by 01_prepare_data.py
ANOMALY_FILE = DATA_DIR / "anomaly_flags.csv"        # produced by 02_anomaly_screening.py

# ---------------------------------------------------------------------------
# Experimental settings (must match the manuscript)
# ---------------------------------------------------------------------------
TARGET = "pos_sales_SAR"
LOOKBACK = 24          # L
H = 12                 # forecast horizon
EPOCHS = 250
BATCH_SIZE = 32
LR = 1e-3
GRAD_CLIP = 1.0

# Chronological partition used in the manuscript
TRAIN_END = "2022-12-01"   # training: 1995-01 .. 2022-12 (336 months)
VAL_END = "2023-12-01"     # validation: 2023-01 .. 2023-12 (12 months)
#                            test: 2024-01 .. 2025-07 (19 months)

# Additional training cutoffs for the backtesting (reproducibility) analysis.
# Each cutoff retrains every model on data up to and including that month and
# evaluates all later origins. The first entry is the main experiment.
MAIN_CUTOFF = TRAIN_END
EXTRA_CUTOFFS = ["2018-12-01", "2019-12-01", "2020-12-01", "2021-12-01"]

# Anomaly screening (Section 4.7)
HAMPEL_WINDOW = 7        # centred window (t-3 .. t+3)
HAMPEL_K = 3.0           # threshold in robust SDs
MAD_SCALE = 1.4826
STL_PERIOD = 12
STL_K = 3.0              # robust z threshold on STL residuals

# Classical reference models (fixed specifications, re-fitted at every origin)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)

MODEL_NAMES = ["lstm", "gru", "tcn", "nbeats", "patchtst"]
# Optional capacity/receptive-field variants (sensitivity analysis, see README)
VARIANT_NAMES = ["tcn24", "patchtst_small"]
STRATEGIES = ["direct", "recursive"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(yhat - y) / denom) * 100.0)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(yhat, float) - np.asarray(y, float))))


def rmse(y, yhat):
    e = np.asarray(yhat, float) - np.asarray(y, float)
    return float(np.sqrt(np.mean(e * e)))


def bias(y, yhat):
    """Mean signed error, forecast minus actual (positive = overprediction)."""
    return float(np.mean(np.asarray(yhat, float) - np.asarray(y, float)))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_series():
    df = pd.read_csv(SERIES_FILE, parse_dates=["period_start"])
    return df.sort_values("period_start").reset_index(drop=True)


def cutoff_index(dates, cutoff):
    """Index of the last training month for a given cutoff date."""
    idx = np.where(pd.to_datetime(dates) == pd.Timestamp(cutoff))[0]
    if len(idx) != 1:
        raise ValueError(f"cutoff {cutoff} not found in series")
    return int(idx[0])


class TrainOnlyEncoder:
    """log1p + standardisation with parameters estimated on training data only."""

    def __init__(self, y_train):
        ylog = np.log1p(np.maximum(np.asarray(y_train, float), 0.0))
        self.mu = float(ylog.mean())
        sd = float(ylog.std())
        self.sd = sd if sd > 1e-9 else 1.0

    def encode(self, y):
        return ((np.log1p(np.maximum(np.asarray(y, float), 0.0)) - self.mu) / self.sd).astype(np.float32)

    def decode(self, z):
        return np.expm1(np.asarray(z, float) * self.sd + self.mu)


def make_windows(z, last_train_idx, lookback, out_len):
    """
    Training windows whose targets lie entirely inside the training period.
    Anchor t uses z[t-L : t] as input and z[t : t+out_len] as target.
    For out_len=12 and 336 training months this yields 301 windows;
    for out_len=1 it yields 312 windows.
    """
    X, Y = [], []
    for t in range(lookback, last_train_idx - out_len + 2):
        X.append(z[t - lookback:t])
        Y.append(z[t:t + out_len])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def period_label(origin_date):
    """Which partition of the MAIN experiment an origin date belongs to."""
    d = pd.Timestamp(origin_date)
    if d <= pd.Timestamp(TRAIN_END):
        return "train"
    if d <= pd.Timestamp(VAL_END):
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Models (imported lazily so that non-neural scripts do not need torch)
# ---------------------------------------------------------------------------
def build_model(name, out_len):
    """
    Architectures identical to the original scripts (DL_12_months_predict.py).
    out_len = 12 for the direct strategy, 1 for the recursive strategy.
    """
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.rnn = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, out_len)

        def forward(self, x):
            o, _ = self.rnn(x.unsqueeze(-1))
            return self.fc(o[:, -1, :])

    class GRUModel(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.rnn = nn.GRU(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, out_len)

        def forward(self, x):
            o, _ = self.rnn(x.unsqueeze(-1))
            return self.fc(o[:, -1, :])

    class TCNModel(nn.Module):
        """Causal dilated convolutions. dilations=(1,2) -> receptive field 7 (original);
        dilations=(1,2,4,8) -> receptive field 31, covering the full 24-month window."""

        def __init__(self, channels=32, k=3, dilations=(1, 2)):
            super().__init__()
            self.inp = nn.Conv1d(1, channels, 1)
            self.convs = nn.ModuleList(
                [nn.Conv1d(channels, channels, k, dilation=d, padding=(k - 1) * d) for d in dilations])
            self.trims = [(k - 1) * d for d in dilations]
            self.act = nn.ReLU()
            self.fc = nn.Linear(channels, out_len)

        def forward(self, x):
            y = self.inp(x.unsqueeze(1))
            for conv, trim in zip(self.convs, self.trims):
                y = self.act(conv(y))[:, :, :-trim]
            return self.fc(y[:, :, -1])

    class NBeatsModel(nn.Module):
        def __init__(self, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(LOOKBACK, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_len))

        def forward(self, x):
            return self.net(x)

    class PatchTSTModel(nn.Module):
        """Original: PyTorch defaults dim_feedforward=2048, dropout=0.1.
        'patchtst_small': dim_feedforward=128, dropout=0.0."""

        def __init__(self, patch_len=6, d_model=64, nhead=4, layers=2, dim_ff=2048, dropout=0.1):
            super().__init__()
            self.patch_len = patch_len
            self.n_patches = LOOKBACK // patch_len
            self.proj = nn.Linear(patch_len, d_model)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                             dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
            self.head = nn.Linear(d_model, out_len)

        def forward(self, x):
            B = x.shape[0]
            n, P = self.n_patches, self.patch_len
            h = self.proj(x[:, :n * P].reshape(B, n, P)) + self.pos
            h = self.encoder(h)
            return self.head(h[:, -1, :])

    builders = {
        "lstm": lambda: LSTMModel(),
        "gru": lambda: GRUModel(),
        "tcn": lambda: TCNModel(dilations=(1, 2)),
        "tcn24": lambda: TCNModel(dilations=(1, 2, 4, 8)),
        "nbeats": lambda: NBeatsModel(),
        "patchtst": lambda: PatchTSTModel(),
        "patchtst_small": lambda: PatchTSTModel(dim_ff=128, dropout=0.0),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name}")
    return builders[name]()


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
