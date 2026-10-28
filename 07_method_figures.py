"""
07_method_figures.py — method diagrams for the manuscript (compact layout, large text).
Output: manuscript_assets/fig_method_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import ROOT

OUT = ROOT / "manuscript_assets"
OUT.mkdir(exist_ok=True)
FS = 15
C = {"grey": "#f0f0f0", "green": "#e6f2e6", "yellow": "#fff2cc", "orange": "#fde3d0", "purple": "#ebe6f5", "blue": "#dde9f7"}


def text_size(ax, s):
    t = ax.text(0, 0, s, fontsize=FS, ha="center", va="center")
    r = ax.figure.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    t.remove()
    return bb.width, bb.height


def box(ax, x, y, s, color, pad=0.18):
    w, h = text_size(ax, s)
    W, Hh = w + 2 * pad, h + 2 * pad
    ax.add_patch(FancyBboxPatch((x - W / 2, y - Hh / 2), W, Hh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#333333", lw=1.4))
    ax.text(x, y, s, fontsize=FS, ha="center", va="center")
    return W, Hh


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.6, mutation_scale=16))


def flow(rows, fname, width=14, gap=0.55, row_h=1.45, label_w=0.0):
    """rows: list of (row_label or None, [(text, color), ...], note or None)."""
    n = len(rows)
    height = sum(row_h + (0.6 if r[2] else 0) + (0.55 if r[0] else 0) for r in rows)
    fig = plt.figure(figsize=(width, height * 0.72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    y = height
    for label, items, note in rows:
        if label:
            y -= 0.3
            ax.text(0.15, y, label, fontsize=FS + 1, fontweight="bold", va="center")
            y -= 0.25
        y -= row_h / 2
        sizes = [text_size(ax, t)[0] + 0.36 for t, _ in items]
        total = sum(sizes) + gap * (len(items) - 1)
        x = (width - total) / 2
        prev = None
        for (t, c), w in zip(items, sizes):
            cx = x + w / 2
            box(ax, cx, y, t, C[c])
            if prev is not None:
                arrow(ax, prev, y, x - 0.02, y)
            prev = x + w + 0.02
            x += w + gap
        y -= row_h / 2
        if note:
            ax.text(width / 2, y - 0.25, note, fontsize=FS - 1, ha="center", va="center", style="italic", color="#333333")
            y -= 0.6
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


flow([(None, [("Monthly POS sales\n1995–2025", "grey"), ("Train-only\nlog1p + z-score", "green"),
              ("Windows\nL = 24", "green"), ("LSTM · GRU · TCN\nMLP · simplified patch Transformer", "yellow"),
              ("Direct (12 outputs)\nor recursive (1 output)", "orange"), ("Evaluation on\nidentical origins", "purple")],
       "Each configuration trained with 20 seeds; classical baselines evaluated on the same origins")],
     "fig_method_framework.png", width=15.5, gap=0.45)

flow([(None, [("Training\nJan 1995–Dec 2022\nn = 336", "green"), ("Validation\nJan–Dec 2023\nn = 12", "purple"),
              ("Test\nJan 2024–Jul 2025\nn = 19", "orange")], None),
      (None, [("Estimate μ and σ of log(1+y)\non training months only", "green"),
              ("Apply the same μ, σ to\nvalidation and test months", "grey"),
              ("No future information\nused in scaling", "orange")], None)],
     "fig_method_preprocessing.png", width=14, row_h=1.6)

flow([("LSTM", [("Input\n[B, 24, 1]", "grey"), ("LSTM\nhidden 64", "yellow"), ("Last hidden\nstate", "green"),
                ("Linear\n64 → 12 or 1", "orange"), ("Forecast", "purple")], None),
      ("GRU", [("Input\n[B, 24, 1]", "grey"), ("GRU\nhidden 64", "yellow"), ("Last hidden\nstate", "green"),
               ("Linear\n64 → 12 or 1", "orange"), ("Forecast", "purple")], None)],
     "fig_method_recurrent.png", width=13, row_h=1.3)

flow([(None, [("Input\n[B, 1, 24]", "grey"), ("1×1 conv\n1 → 32", "yellow"), ("Causal conv\nk = 3, d = 1", "yellow"),
              ("Causal conv\nk = 3, d = 2", "yellow"), ("Last position\nLinear 32 → 12 or 1", "orange")],
       "Receptive field: 7 months (TCN-24 variant: dilations 1, 2, 4, 8; 31 months)")],
     "fig_method_tcn.png", width=13.5)

flow([("MLP", [("24-month\nvector", "grey"), ("Dense 256\n+ ReLU", "yellow"), ("Dense 256\n+ ReLU", "yellow"),
                        ("Dense\n12 or 1", "orange")], None),
      ("Simplified patch Transformer", [("24-month\nvector", "grey"), ("4 patches\nlength 6", "green"), ("Patch embedding\n6 → 64", "yellow"),
                          ("+ positional\nembedding", "purple"), ("Encoder: 2 layers, 4 heads\nFF 2048, dropout 0.1", "yellow"),
                          ("Linear\n64 → 12 or 1", "orange")], None)],
     "fig_method_nbeats_patchtst.png", width=15.5, row_h=1.3, gap=0.4)

flow([("N-BEATS (canonical)", [("Input x\n24 values", "grey"), ("Block 1: FC stack\n→ backcast b₁, forecast f₁", "yellow"),
                               ("Residual\nx − b₁", "green"), ("Block 2 … K\nsame structure", "yellow"),
                               ("Forecast = sum of\nblock forecasts", "orange")],
       "Generic or interpretable (polynomial trend + monthly Fourier) bases; K blocks of 2–4 FC layers"),
      ("PatchTST (canonical)", [("Input\n24 values", "grey"), ("RevIN\n(per window)", "purple"),
                                ("Overlapping patches\nP = 4, S = 2, 12 tokens", "green"),
                                ("Embedding +\nlearned positions", "yellow"),
                                ("Encoder with\nBatchNorm, 3 layers", "yellow"), ("Flatten head\n→ 12 or 1", "orange"),
                                ("Reverse\nRevIN", "purple")], None)],
     "fig_method_canonical.png", width=17, row_h=1.5, gap=0.4)

flow([(None, [("301 windows (direct)\n312 windows (recursive)", "grey"), ("Shuffled\nmini-batches of 32", "green"),
              ("Forward pass", "yellow"), ("MSE loss", "purple"), ("Backpropagation\nclip norm 1.0", "green"),
              ("Adam, lr 10⁻³\n250 epochs", "orange")],
       "Identical settings for all architectures; repeated with 20 random seeds")],
     "fig_method_training.png", width=15.5, gap=0.4)

flow([("Hampel / MAD", [("Observed monthly\nPOS sales", "grey"), ("Centred 7-month window\n|y − median| > 3 × 1.4826 MAD", "green"),
                        ("Hampel flag", "green")], None),
      ("Robust STL", [("Observed monthly\nPOS sales", "grey"), ("STL on log(1+y), period 12\n|robust z of residual| > 3", "purple"),
                      ("STL flag", "purple")], None),
      (None, [("Hampel OR STL", "yellow"), ("Flags used for interpretation and\nsensitivity analysis only (no replacement)", "orange")], None)],
     "fig_method_anomaly.png", width=13, row_h=1.3)

flow([("Rolling first-horizon sample", [("Every validation and\ntest origin t", "grey"), ("12-month forecast\nfrom each model", "yellow"),
                                         ("Keep h = 1", "green"), ("Validation n = 12\nTest n = 19", "orange")], None),
      ("Complete-horizon sample", [("Test origins with 12\nobserved future months", "grey"), ("Same stored\nforecasts", "yellow"),
                                    ("Keep h = 1, …, 12", "green"), ("8 origins\nper horizon", "orange")],
       "All neural configurations (both strategies, 20 seeds) and all classical baselines")],
     "fig_method_evaluation.png", width=13.5, row_h=1.3)

print("method figures written to", OUT)
