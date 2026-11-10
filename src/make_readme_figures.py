"""Render the README figures from committed result files only.

Reads small CSV summaries under results/ (no data/, no predictions, no models),
writes PNGs to figures/readme/ and a manifest with input/output SHA256 hashes.
It performs no training, inference, resampling or calibration fitting.

Usage (from the repository root, with requirements-plot.txt installed):
    python src/make_readme_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "readme"
DPI = 200

# Colorblind-safe categorical slots (blue/orange pass the CVD check as a pair).
BLUE, ORANGE, GREEN, RED, TEAL = "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"
INK, INK2, MUTED, GRID, ZERO = "#1F2328", "#57606A", "#8C959F", "#E6E8EB", "#6E7781"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK2,
    "axes.labelsize": 11,
    "axes.edgecolor": GRID,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
})

INPUTS: dict[str, str] = {}


def read_csv(rel: str) -> list[dict]:
    path = ROOT / rel
    INPUTS[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pick(rows: list[dict], **match) -> dict:
    found = [r for r in rows if all(r[k] == v for k, v in match.items())]
    if len(found) != 1:
        raise ValueError(f"expected one row for {match}, found {len(found)}")
    return found[0]


def style(ax, ygrid=True):
    """White plot, horizontal hairline gridlines only, no box."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C9CED4")
    ax.tick_params(axis="both", length=0, pad=6)
    ax.xaxis.grid(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def headline(fig, title, subtitle, y=0.985):
    fig.text(0.012, y, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.012, y - 0.065, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)


def save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return str(path.relative_to(ROOT))


def test_metric(run: str, metric: str) -> float:
    rows = read_csv(f"results/{run}/evaluation/overall_metrics.csv")
    return float(pick(rows, split="test", segment="overall")[metric])


# ---------------------------------------------------------------- figure 1
def fig_information_vs_model() -> str:
    ablation = [("F1", "lr_f1"), ("F1+F2", "lr_f1_f2"), ("F1+F2+F3", "lr_f1_f2_f3")]
    models = [("LR", "lr_f1_f2_f3"), ("LightGBM", "lightgbm_f1_f2_f3"),
              ("Wide & Deep", "wide_deep_f1_f2_f3"), ("DeepFM", "deepfm_f1_f2_f3")]
    a_vals = [test_metric(run, "auc") for _, run in ablation]
    m_vals = [test_metric(run, "auc") for _, run in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True,
                                   gridspec_kw={"width_ratios": [3, 4], "wspace": 0.08})
    base = 0.5
    for ax, labels, vals in ((ax1, [a for a, _ in ablation], a_vals),
                             (ax2, [m for m, _ in models], m_vals)):
        style(ax)
        x = range(len(vals))
        ax.bar(x, [v - base for v in vals], bottom=base, width=0.56, color=BLUE, zorder=2)
        for i, v in zip(x, vals):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=10.5, color=INK)
        ax.set_xticks(list(x), labels)
        ax.set_xlim(-0.6, len(vals) - 0.4)
    ax1.set_ylim(base, 0.685)
    ax1.set_ylabel("Test AUC (axis starts at 0.5 = random)")
    ax1.set_title("(a) Logistic regression, adding feature groups", loc="left", fontsize=11.5)
    ax2.set_title("(b) Four model families, same F1+F2+F3 features", loc="left", fontsize=11.5)
    ax1.set_xlabel("Feature set")
    ax2.set_xlabel("Model")

    d2, d3 = a_vals[1] - a_vals[0], a_vals[2] - a_vals[1]
    # Step sizes sit inside the bar that adds the feature group (white on the bar fill).
    ax1.text(1, 0.545, f"+{d2:.4f}\nfrom F2", ha="center", va="center", fontsize=10,
             color="white", fontweight="bold")
    ax1.text(2, 0.545, f"+{d3:.4f}\nfrom F3", ha="center", va="center", fontsize=10,
             color="white", fontweight="bold")
    spread = max(m_vals) - min(m_vals)
    ax2.text(1.5, 0.668, f"Spread across the four models: {spread:.4f} AUC",
             ha="center", va="bottom", fontsize=10, color=INK)

    headline(fig, "Historical click feedback moves test AUC far more than model choice",
             "Test day 2017-05-13 (3,308,665 impressions). F3 = completed-day click feedback; "
             "one seed per model.")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.12)
    return save(fig, "fig1_information_vs_model.png")


# ---------------------------------------------------------------- figure 2
def fig_model_differences() -> str:
    rows = read_csv("results/frozen_model_bootstrap_v1/bootstrap_intervals.csv")
    names = {"deepfm_f123": "DeepFM\n(primary)", "wide_deep_f123": "Wide & Deep",
             "lightgbm_f123": "LightGBM"}
    order = ["deepfm_f123", "wide_deep_f123", "lightgbm_f123"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.9), gridspec_kw={"wspace": 0.28})
    for ax, metric, title in ((ax1, "auc", "(a) Δ test AUC, LR minus model"),
                              (ax2, "gauc", "(b) Δ test GAUC, LR minus model")):
        style(ax)
        ax.axhline(0, color=ZERO, linewidth=1.0, zorder=1)
        for i, right in enumerate(order):
            match = [r for r in rows if r["right"] == right and r["metric"] == metric]
            if not match:
                ax.text(i, 0.00012, "not in the\npreregistered\nbootstrap", ha="center",
                        va="bottom", fontsize=9, color=MUTED)
                continue
            r = match[0]
            eff, lo, hi = float(r["effect"]), float(r["ci_low"]), float(r["ci_high"])
            ax.errorbar(i, eff, yerr=[[eff - lo], [hi - eff]], fmt="o", color=BLUE,
                        ecolor=BLUE, elinewidth=2, capsize=5, capthick=2, markersize=8,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            ax.text(i + 0.1, eff, f"{eff:+.5f}", ha="left", va="center", fontsize=9.5, color=INK)
        ax.set_xticks(range(len(order)), [names[k] for k in order])
        ax.set_xlim(-0.5, len(order) - 0.3)
        ax.set_xlabel("Compared model (LR minus this model)")
        ax.set_title(title, loc="left", fontsize=11.5)
    ax1.set_ylabel("Difference with 95% CI")
    ax1.set_ylim(-0.0005, 0.0080)
    ax2.set_ylim(-0.0016, 0.0019)
    headline(fig, "LR's global-ranking lead survives user resampling; within-user differences do not",
             "Paired user-cluster bootstrap of saved test predictions, 1,000 replicates over "
             "391,741 users. LR − DeepFM AUC: Holm p = 0.002.")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.18)
    return save(fig, "fig2_model_differences_ci.png")


# ---------------------------------------------------------------- figure 3
def fig_sequence_controls() -> str:
    rows = read_csv("results/f5_sequence_test_v1/bootstrap_intervals.csv")
    comps = [("1", "Matched\n− zero", True), ("2", "Matched\n− mismatched", True),
             ("5", "Attention\n− mean", False), ("6", "F5 − F4\nsource", False)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.1), gridspec_kw={"wspace": 0.24})
    for ax, metric, title in ((ax1, "auc", "(a) Δ test AUC"), (ax2, "gauc", "(b) Δ test GAUC")):
        style(ax)
        ax.axhline(0, color=ZERO, linewidth=1.0, zorder=1)
        for i, (cid, _, primary) in enumerate(comps):
            r = pick(rows, id=cid, metric=metric)
            eff, lo, hi = float(r["effect"]), float(r["ci_low"]), float(r["ci_high"])
            color = BLUE if primary else ORANGE
            ax.errorbar(i, eff, yerr=[[eff - lo], [hi - eff]], fmt="o" if primary else "s",
                        color=color, ecolor=color, elinewidth=2, capsize=5, capthick=2,
                        markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            ax.text(i + 0.12, eff, f"{eff:+.5f}".replace("-", "−"), ha="left", va="center",
                    fontsize=9, color=INK)
        ax.set_xticks(range(len(comps)), [c[1] for c in comps])
        ax.set_xlim(-0.5, len(comps) - 0.3)
        ax.set_title(title, loc="left", fontsize=11.5)
    ax1.set_ylabel("Difference with 95% CI")
    ax1.set_ylim(-0.0002, 0.0018)
    ax2.set_ylim(-0.0008, 0.0013)
    ax1.errorbar([], [], fmt="o", color=BLUE, markersize=8, label="Preregistered primary (Holm on AUC/LogLoss)")
    ax1.errorbar([], [], fmt="s", color=ORANGE, markersize=8, label="Secondary (no multiplicity correction)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.012, 0.87), ncol=2, fontsize=10, handletextpad=0.4)
    headline(fig, "Matched behavior history gives a small AUC gain; primary GAUC intervals include zero",
             "F5: mean-pooled residual over the last 200 behavior_log events on a frozen DeepFM; "
             "mean of 3 seeds; user-cluster bootstrap.")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.74, bottom=0.15)
    return save(fig, "fig3_f5_sequence_controls.png")


# ---------------------------------------------------------------- figure 4
def fig_calibration() -> str:
    raw = read_csv("results/deepfm_f1_f2_f3/evaluation/reliability_bins.csv")
    cal = read_csv("results/deepfm_f1_f2_f3/calibration/evaluation/reliability_bins.csv")
    raw = [r for r in raw if r["split"] == "test"]
    cal = [r for r in cal if r["split"] == "test"]
    seg_raw = read_csv("results/deepfm_f1_f2_f3/evaluation/cold_start_metrics.csv")
    seg_cal = read_csv("results/deepfm_f1_f2_f3/calibration/evaluation/cold_start_metrics.csv")
    ece_raw = test_metric("deepfm_f1_f2_f3", "ece")
    ece_cal = float(pick(read_csv("results/deepfm_f1_f2_f3/calibration/evaluation/overall_metrics.csv"),
                         split="test", segment="overall")["ece"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.3), gridspec_kw={"wspace": 0.3,
                                                                           "width_ratios": [1, 1.15]})
    style(ax1)
    lim = 12.5
    ax1.plot([0, lim], [0, lim], color=MUTED, linewidth=1, zorder=1)
    ax1.text(7.4, 9.0, "perfect calibration (y = x)", fontsize=9, color=MUTED, ha="right", va="bottom")
    for rows, color, marker, label in ((raw, BLUE, "o", f"Raw (ECE {ece_raw:.4f})"),
                                       (cal, ORANGE, "s", f"Isotonic (ECE {ece_cal:.4f})")):
        xs = [100 * float(r["mean_prediction"]) for r in rows]
        ys = [100 * float(r["actual_ctr"]) for r in rows]
        ax1.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                 markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Mean predicted CTR in bin (%)")
    ax1.set_ylabel("Observed CTR in bin (%)")
    ax1.set_title("(a) Reliability, 10 equal-count bins", loc="left", fontsize=11.5)
    ax1.legend(loc="upper left")

    style(ax2)
    segs = [("known_user__known_ad", "Known user\nknown ad"), ("known_user__new_ad", "Known user\nnew ad"),
            ("new_user__known_ad", "New user\nknown ad"), ("new_user__new_ad", "New user\nnew ad")]
    ax2.axhline(1.0, color=MUTED, linewidth=1, zorder=1)
    ax2.text(3.45, 1.006, "COPC = 1 (calibrated)", fontsize=9, color=MUTED, ha="right", va="bottom")
    for i, (key, _) in enumerate(segs):
        r0 = pick(seg_raw, split="test", segment=key)
        r1 = pick(seg_cal, split="test", segment=key)
        c0, c1 = float(r0["copc"]), float(r1["copc"])
        ax2.plot([i - 0.12, i + 0.12], [c0, c1], color=GRID, linewidth=2, zorder=2)
        ax2.plot(i - 0.12, c0, "o", color=BLUE, markersize=8, markeredgecolor="white",
                 markeredgewidth=1.2, zorder=3, label="Raw" if i == 0 else None)
        ax2.plot(i + 0.12, c1, "s", color=ORANGE, markersize=8, markeredgecolor="white",
                 markeredgewidth=1.2, zorder=3, label="Isotonic" if i == 0 else None)
        rows = int(r0["rows"])
        ax2.text(i, 0.475, f"{rows:,} rows", ha="center", va="bottom", fontsize=8.5, color=MUTED)
        if key == "new_user__new_ad":
            over0, over1 = float(r0["prediction_bias_pct"]), float(r1["prediction_bias_pct"])
            ax2.text(i - 0.12, c1 + 0.045, f"over-predicts\n{over0:.0f}% → {over1:.0f}%",
                     ha="center", va="bottom", fontsize=9, color=INK)
    ax2.set_xticks(range(len(segs)), [s[1] for s in segs])
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(0.45, 1.12)
    ax2.set_ylabel("COPC = observed / predicted clicks\n(below 1 = over-prediction)")
    ax2.set_title("(b) COPC by cold-start segment", loc="left", fontsize=11.5)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08))
    headline(fig, "Isotonic calibration fixes DeepFM's overall probability level, not cold-start bias",
             "DeepFM F1+F2+F3, frozen test day. Calibrator fit on validation only "
             "(selected by validation LogLoss).")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.14)
    return save(fig, "fig4_deepfm_calibration.png")


# ---------------------------------------------------------------- figure 5
def fig_serving_cost() -> str:
    rows = read_csv("results/serving_benchmark_v1/serving_benchmark.csv")
    deep = pick(rows, model="deepfm_f123")
    f5 = pick(rows, model="f5_mean_matched")
    labels = ["DeepFM\nbackbone", "F5 sequence\n(DeepFM + residual)"]
    tput = [float(deep["end_to_end_rps_batch_8192"]), float(f5["end_to_end_rps_batch_8192"])]
    mem = [float(deep["peak_rss_mib"]), float(f5["peak_rss_mib"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.3})
    for ax, vals, fmt, ylabel, title in (
            (ax1, [v / 1e6 for v in tput], "{:.2f}M", "Rows per second (millions)",
             "(a) Throughput, batch 8192 (higher is better)"),
            (ax2, mem, "{:,.1f} MiB", "Peak resident memory (MiB)",
             "(b) Peak memory (lower is better)")):
        style(ax)
        ax.bar([0, 1], vals, width=0.5, color=[BLUE, ORANGE], zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=10.5,
                    color=INK)
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11.5)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ratio_t = tput[0] / tput[1]
    ratio_m = mem[1] / mem[0]
    ax1.text(1, tput[1] / 1e6 + 0.22, f"1/{ratio_t:.2f} of DeepFM", ha="center", fontsize=10,
             color=INK, fontweight="bold")
    ax2.text(1, mem[1] * 1.10, f"{ratio_m:.2f}× DeepFM", ha="center", fontsize=10, color=INK,
             fontweight="bold")
    headline(fig, "The F5 sequence gain costs about 3.3× compute per row and 7.5× memory here",
             "CPU only, 8 threads; in-memory prepare + inference (no disk I/O timed). "
             "Memory is dominated by the resident 723M-event behavior store.")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.15)
    return save(fig, "fig5_serving_cost.png")


def main() -> None:
    outputs = [fig_information_vs_model(), fig_model_differences(), fig_sequence_controls(),
               fig_calibration(), fig_serving_cost()]
    manifest = {
        "script": "src/make_readme_figures.py",
        "scope": "reads committed result CSVs only; no data, predictions, models, fitting or resampling",
        "matplotlib": matplotlib.__version__,
        "dpi": DPI,
        "inputs_sha256": dict(sorted(INPUTS.items())),
        "figures_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in outputs},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["figures_sha256"], indent=2))


if __name__ == "__main__":
    main()
