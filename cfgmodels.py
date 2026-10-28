"""
cfgmodels.py — configurable versions of the five architectures (additional experiments):
hyperparameter search, learning curves, second series and multivariate inputs.

Normalization experiments:
  * input/output normalisation wrappers applied to any model (cfg["norm"]):
      "global"  - no further normalisation (train-only log/z-score encoding only; default)
      "revin"   - reversible instance normalisation per input window (Kim et al., 2022)
      "anchor"  - inputs and forecasts expressed relative to the last observed value
  * canonical N-BEATS ("nbeats_c"): doubly residual stacks of blocks with backcast and
    forecast outputs; generic or interpretable (trend + seasonality) basis
  * canonical PatchTST ("patchtst_c"): RevIN, overlapping patches with end padding,
    channel independence, BatchNorm Transformer encoder and flatten head

All models take x with shape (batch, L, C) — C input channels (C = 1 univariate).
With the DEFAULT configurations and C = 1 the architectures (and parameter counts)
are identical to the original implementations in common.build_model.
"""
import copy
import random

import numpy as np

# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------
TRAIN_DEFAULT = {"lr": 1e-3, "batch": 32, "epochs": 250, "weight_decay": 0.0}
ARCH_DEFAULT = {
    "lstm": {"hidden": 64, "layers": 1, "dropout": 0.0},
    "gru": {"hidden": 64, "layers": 1, "dropout": 0.0},
    "tcn": {"channels": 32, "kernel": 3, "dilations": [1, 2], "dropout": 0.0},
    "nbeats": {"hidden": 256, "layers": 2, "dropout": 0.0},
    "patchtst": {"patch_len": 6, "d_model": 64, "nhead": 4, "layers": 2, "dim_ff": 2048, "dropout": 0.1},
    # canonical models (normalization experiments); PatchTST defaults follow the small-dataset setting of Nie et al. (2023)
    "nbeats_c": {"basis": "generic", "blocks": 6, "layers": 4, "width": 128, "dropout": 0.0},
    "patchtst_c": {"patch_len": 4, "stride": 2, "d_model": 16, "nhead": 4, "layers": 3, "dim_ff": 128,
                   "dropout": 0.3, "head_dropout": 0.0},
}

SEARCH_TRAIN = {"lr": [3e-4, 1e-3, 3e-3], "batch": [16, 32, 64], "epochs": [100, 250, 500],
                "weight_decay": [0.0, 1e-4, 1e-3]}
SEARCH_ARCH = {
    "lstm": {"hidden": [16, 32, 64, 128], "layers": [1, 2], "dropout": [0.0, 0.1, 0.2]},
    "gru": {"hidden": [16, 32, 64, 128], "layers": [1, 2], "dropout": [0.0, 0.1, 0.2]},
    "tcn": {"channels": [16, 32, 64], "kernel": [2, 3], "dilations": [[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            "dropout": [0.0, 0.1, 0.2]},
    "nbeats": {"hidden": [64, 128, 256, 512], "layers": [2, 3, 4], "dropout": [0.0, 0.1, 0.2]},
    "patchtst": {"patch_len": [3, 4, 6, 8], "d_model": [16, 32, 64], "nhead": [2, 4], "layers": [1, 2],
                 "dim_ff": [64, 128, 256, 2048], "dropout": [0.0, 0.1, 0.2]},
}


SEARCH_ARCH.update({
    "nbeats_c": {"basis": ["generic", "interpretable"], "blocks": [3, 6, 12], "layers": [2, 4],
                 "width": [64, 128, 256, 512], "dropout": [0.0, 0.1, 0.2]},
    "patchtst_c": {"patch_len": [3, 4, 6, 8], "stride": [1, 2, 3], "d_model": [16, 32, 64], "nhead": [2, 4],
                   "layers": [1, 2, 3], "dim_ff": [64, 128, 256], "dropout": [0.0, 0.1, 0.2, 0.3],
                   "head_dropout": [0.0, 0.1]},
})
# Search for the normalization experiments: validation early stopping is always used (patience 25, best weights restored);
# the maximum number of epochs and the normalisation are searched together with the other settings.
SEARCH_TRAIN_R3 = {"lr": [3e-4, 1e-3, 3e-3], "batch": [16, 32, 64], "epochs": [250, 500],
                   "weight_decay": [0.0, 1e-4, 1e-3], "patience": [25]}
NORMS = ["global", "revin", "anchor"]
PATIENCE = 25


def norms_for(name):
    """patchtst_c contains RevIN internally, so no further wrapper is applied to it."""
    return ["global"] if name == "patchtst_c" else NORMS


def default_cfg(name):
    return {"arch": copy.deepcopy(ARCH_DEFAULT[name]), "train": dict(TRAIN_DEFAULT)}


def sample_cfgs(name, strategy, n_trials, seed=12345):
    """Trial 0 is the default configuration; trials 1..n-1 are random draws (without duplicates)."""
    rng = random.Random(f"{seed}-{name}-{strategy}")
    cfgs, seen = [default_cfg(name)], set()
    seen.add(repr(cfgs[0]))
    while len(cfgs) < n_trials:
        c = {"arch": {k: rng.choice(v) for k, v in SEARCH_ARCH[name].items()},
             "train": {k: rng.choice(v) for k, v in SEARCH_TRAIN.items()}}
        if repr(c) not in seen:
            seen.add(repr(c))
            cfgs.append(c)
    return cfgs


def sample_cfgs_r3(name, strategy, n_trials, seed=54321):
    """Trial 0: default architecture + early stopping, global normalisation (RevIN for patchtst_c is built in);
    trials 1..n-1: random draws over architecture, training settings and normalisation."""
    rng = random.Random(f"r3-{seed}-{name}-{strategy}")
    c0 = default_cfg(name)
    c0["train"]["patience"] = PATIENCE
    c0["norm"] = "global"
    cfgs, seen = [c0], {repr(c0)}
    while len(cfgs) < n_trials:
        c = {"arch": {k: rng.choice(v) for k, v in SEARCH_ARCH[name].items()},
             "train": {k: rng.choice(v) for k, v in SEARCH_TRAIN_R3.items()},
             "norm": rng.choice(norms_for(name))}
        if name == "patchtst_c" and c["arch"]["stride"] > c["arch"]["patch_len"]:
            continue
        if repr(c) not in seen:
            seen.add(repr(c))
            cfgs.append(c)
    return cfgs


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def build(name, out_len, arch, in_ch=1, lookback=24, norm="global"):
    base = _build(name, out_len, arch, in_ch, lookback)
    if norm == "global" or name == "patchtst_c":
        return base
    return _wrap(base, norm, in_ch)


def _wrap(base, norm, in_ch):
    import torch
    import torch.nn as nn

    class RevIN(nn.Module):
        """Reversible instance normalisation: each input window is standardised by its own mean and
        standard deviation (per channel, learnable affine map); forecasts are mapped back with the
        statistics of the target channel of the same window."""

        def __init__(self):
            super().__init__()
            self.base = base
            self.w = nn.Parameter(torch.ones(in_ch))
            self.b = nn.Parameter(torch.zeros(in_ch))

        def forward(self, x):
            m = x.mean(1, keepdim=True)
            s = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            out = self.base((x - m) / s * self.w + self.b)
            out = (out - self.b[0]) / (self.w[0] + 1e-8)
            return out * s[:, 0, :1] + m[:, 0, :1]

    class Anchor(nn.Module):
        """Inputs relative to the last observed value of each channel; forecasts are changes
        relative to the last observed value of the target."""

        def __init__(self):
            super().__init__()
            self.base = base

        def forward(self, x):
            last = x[:, -1:, :]
            return self.base(x - last) + last[:, 0, :1]

    if norm == "revin":
        return RevIN()
    if norm == "anchor":
        return Anchor()
    raise ValueError(norm)


def _build(name, out_len, arch, in_ch=1, lookback=24):
    import math
    import torch
    import torch.nn as nn

    class RNN(nn.Module):
        def __init__(self, cell):
            super().__init__()
            L = arch["layers"]
            self.rnn = cell(in_ch, arch["hidden"], num_layers=L, batch_first=True,
                            dropout=arch["dropout"] if L > 1 else 0.0)
            self.drop = nn.Dropout(arch["dropout"])
            self.fc = nn.Linear(arch["hidden"], out_len)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(self.drop(o[:, -1, :]))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            ch, k = arch["channels"], arch["kernel"]
            self.inp = nn.Conv1d(in_ch, ch, 1)
            self.convs = nn.ModuleList([nn.Conv1d(ch, ch, k, dilation=d, padding=(k - 1) * d) for d in arch["dilations"]])
            self.trims = [(k - 1) * d for d in arch["dilations"]]
            self.act = nn.ReLU()
            self.drop = nn.Dropout(arch["dropout"])
            self.fc = nn.Linear(ch, out_len)

        def forward(self, x):
            y = self.inp(x.transpose(1, 2))
            for conv, trim in zip(self.convs, self.trims):
                y = self.drop(self.act(conv(y))[:, :, :-trim])
            return self.fc(y[:, :, -1])

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            h, layers, d = arch["hidden"], [], lookback * in_ch
            for _ in range(arch["layers"]):
                layers += [nn.Linear(d, h), nn.ReLU()]
                if arch["dropout"] > 0:
                    layers.append(nn.Dropout(arch["dropout"]))
                d = h
            layers.append(nn.Linear(d, out_len))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.reshape(x.shape[0], -1))

    class Patch(nn.Module):
        def __init__(self):
            super().__init__()
            P = arch["patch_len"]
            self.P, self.n = P, lookback // P
            self.proj = nn.Linear(P * in_ch, arch["d_model"])
            enc = nn.TransformerEncoderLayer(d_model=arch["d_model"], nhead=arch["nhead"],
                                             dim_feedforward=arch["dim_ff"], dropout=arch["dropout"],
                                             batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=arch["layers"])
            self.pos = nn.Parameter(torch.zeros(1, self.n, arch["d_model"]))
            self.head = nn.Linear(arch["d_model"], out_len)

        def forward(self, x):
            B = x.shape[0]
            start = x.shape[1] - self.n * self.P            # use the most recent n*P months
            h = x[:, start:, :].reshape(B, self.n, self.P * x.shape[2])
            h = self.encoder(self.proj(h) + self.pos)
            return self.head(h[:, -1, :])

    class NBeatsBlock(nn.Module):
        def __init__(self, kind):
            super().__init__()
            w, d = arch["width"], lookback * in_ch
            layers = [nn.Linear(d, w), nn.ReLU()]
            for _ in range(arch["layers"] - 1):
                layers += [nn.Linear(w, w), nn.ReLU()]
                if arch["dropout"] > 0:
                    layers.append(nn.Dropout(arch["dropout"]))
            self.fc = nn.Sequential(*layers)
            self.kind = kind
            if kind == "generic":
                self.back = nn.Linear(w, d)
                self.fore = nn.Linear(w, out_len)
            else:
                # one coefficient vector defines a continuous curve over the input window and the forecast
                # period (t = 0..L-1 backcast, t = L..L+out_len-1 forecast); only the target channel is fitted
                t = torch.arange(lookback + out_len, dtype=torch.float32)
                if kind == "trend":                       # polynomial of degree 3
                    tt = t / lookback
                    basis = torch.stack([tt ** p for p in range(4)])
                else:                                      # monthly Fourier harmonics (period 12)
                    rows = [torch.ones_like(t)]
                    for i in range(1, 7):
                        rows.append(torch.cos(2 * math.pi * i * t / 12))
                        if i < 6:
                            rows.append(torch.sin(2 * math.pi * i * t / 12))
                    basis = torch.stack(rows)
                self.register_buffer("basis", basis)
                self.theta = nn.Linear(w, basis.shape[0], bias=False)

        def forward(self, x):
            h = self.fc(x)
            if self.kind == "generic":
                return self.back(h), self.fore(h)
            c = self.theta(h) @ self.basis                  # (B, L + out_len)
            back = c[:, :lookback]                          # univariate: input vector = target history
            return back, c[:, lookback:]

    class NBeats(nn.Module):
        """Doubly residual stacking: each block removes its backcast from the input passed to the next
        block and adds its forecast to the total forecast (Oreshkin et al., 2020)."""

        def __init__(self):
            super().__init__()
            n = arch["blocks"]
            if arch["basis"] == "generic":
                kinds = ["generic"] * n
            else:
                kinds = ["trend"] * max(1, n // 2) + ["seasonality"] * max(1, n - n // 2)
            self.blocks = nn.ModuleList([NBeatsBlock(k) for k in kinds])

        def forward(self, x):
            r = x.reshape(x.shape[0], -1)
            f = 0
            for b in self.blocks:
                back, fore = b(r)
                r = r - back
                f = f + fore
            return f

    class EncLayer(nn.Module):
        """Transformer encoder layer with BatchNorm, as in PatchTST."""

        def __init__(self, d, nhead, ff, p):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, nhead, dropout=p, batch_first=True)
            self.bn1, self.bn2 = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
            self.drop = nn.Dropout(p)

        def forward(self, x):
            x = x + self.drop(self.attn(x, x, x, need_weights=False)[0])
            x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.drop(self.ff(x))
            return self.bn2(x.transpose(1, 2)).transpose(1, 2)

    class PatchTSTc(nn.Module):
        """PatchTST (Nie et al., 2023): RevIN, channel independence, overlapping patches of length P with
        stride S and end padding (N = floor((L-P)/S) + 2 patches), learnable positional encoding,
        BatchNorm Transformer encoder and a flatten-linear head. The forecast of channel 0 is returned."""

        def __init__(self):
            super().__init__()
            P, S, d = arch["patch_len"], arch["stride"], arch["d_model"]
            self.P, self.S = P, S
            self.n = (lookback - P) // S + 2
            self.w = nn.Parameter(torch.ones(in_ch))
            self.b = nn.Parameter(torch.zeros(in_ch))
            self.proj = nn.Linear(P, d)
            self.pos = nn.Parameter(torch.empty(1, self.n, d).uniform_(-0.02, 0.02))
            self.drop = nn.Dropout(arch["dropout"])
            self.layers = nn.ModuleList([EncLayer(d, arch["nhead"], arch["dim_ff"], arch["dropout"])
                                         for _ in range(arch["layers"])])
            self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(arch["head_dropout"]),
                                      nn.Linear(self.n * d, out_len))

        def forward(self, x):
            B, L, C = x.shape
            m = x.mean(1, keepdim=True)
            s = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            z = (x - m) / s * self.w + self.b                       # RevIN
            z = z.permute(0, 2, 1).reshape(B * C, L)                # channel independence
            z = torch.cat([z, z[:, -1:].repeat(1, self.S)], 1)      # end padding
            p = z.unfold(1, self.P, self.S)                         # (B*C, n, P)
            h = self.drop(self.proj(p) + self.pos)
            for layer in self.layers:
                h = layer(h)
            out = self.head(h).reshape(B, C, -1)[:, 0, :]
            out = (out - self.b[0]) / (self.w[0] + 1e-8)
            return out * s[:, 0, :1] + m[:, 0, :1]

    if name == "nbeats_c":
        if arch["basis"] != "generic" and in_ch != 1:
            raise ValueError("interpretable N-BEATS is implemented for univariate input")
        return NBeats()
    if name == "patchtst_c":
        return PatchTSTc()
    if name == "lstm":
        return RNN(nn.LSTM)
    if name == "gru":
        return RNN(nn.GRU)
    if name == "tcn":
        return TCN()
    if name == "nbeats":
        return MLP()
    if name == "patchtst":
        return Patch()
    raise ValueError(name)
