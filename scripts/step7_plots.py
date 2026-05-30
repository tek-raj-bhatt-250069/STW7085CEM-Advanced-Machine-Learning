"""Step 7 figures. Consume step7_metrics.json + the cross-method OOF parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import CLASS_NAMES  # noqa: E402
from gp_classification import reliability_curve  # noqa: E402
from calibration_deepdive import METHODS, METHOD_LABELS  # noqa: E402

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# per-method palette
METHOD_COLOURS = {
    "rf_direct":    "#3B6BB0",
    "direct_svgp":  "#7F4A89",
    "rf_threshold": "#88B0D6",
    "gp_threshold": "#2E8B8B",
}
CLASS_COLOURS = ["#78A678", "#E6A157", "#B73A2B"]


def load_data() -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((PROJECT_ROOT / "data" / "processed" / "step7_metrics.json").read_text())
    oof = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "step7_cross_method_oof.parquet")
    return metrics, oof


def plot_unified_reliability(
    metrics: dict, oof: pd.DataFrame,
    out_path: Path = FIG_DIR / "step7_unified_reliability.png",
) -> Path:
    """4×3 grid: rows = methods, columns = classes. The single most important figure."""
    y_true = oof["true_class"].to_numpy()
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), sharex=True, sharey=True)

    for i, method in enumerate(METHODS):
        proba = oof[[f"{method}_proba_normal",
                     f"{method}_proba_delays",
                     f"{method}_proba_diversions"]].to_numpy()
        for j, class_idx in enumerate([0, 1, 2]):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
            mp, ef, counts = reliability_curve(y_true, proba, class_idx)
            mp_arr, ef_arr = np.array(mp, dtype=float), np.array(ef, dtype=float)
            valid = ~(np.isnan(mp_arr) | np.isnan(ef_arr))
            ax.plot(mp_arr[valid], ef_arr[valid], "o-",
                    color=METHOD_COLOURS[method], linewidth=2, markersize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                n_pos = int((y_true == class_idx).sum())
                ax.set_title(f"{CLASS_NAMES[class_idx]}  (n={n_pos})", fontsize=11)
            if j == 0:
                ax.set_ylabel(METHOD_LABELS[method] + "\nEmpirical freq", fontsize=10)
            if i == 3:
                ax.set_xlabel("Mean predicted probability", fontsize=10)
            # annotate REL/RES per panel
            bs = metrics[method]["brier_decomposition"]["per_class"][CLASS_NAMES[class_idx]]
            ax.text(0.97, 0.03,
                    f"REL={bs['REL']:.3f}\nRES={bs['RES']:.3f}",
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="grey", alpha=0.85))

    fig.suptitle(
        "Unified reliability diagrams: 4 methods × 3 classes on 1206 OOF nights\n"
        "Each panel: empirical freq vs mean predicted probability per bin. Closer to diagonal = better calibrated.\n"
        "Annotated per-panel: REL (calibration loss) and RES (resolution).",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_brier_decomposition(
    metrics: dict, out_path: Path = FIG_DIR / "step7_brier_decomposition.png",
) -> Path:
    """Per-method, per-class REL (left) and RES (right). RF-threshold wins both."""
    x = np.arange(4)
    width = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, comp, label, lower_better in [
        (axes[0], "REL", "Reliability (calibration loss; lower = better calibrated)", True),
        (axes[1], "RES", "Resolution (sharpness; higher = more informative)", False),
    ]:
        for ci, cname in enumerate(CLASS_NAMES.values()):
            vals = [metrics[m]["brier_decomposition"]["per_class"][cname][comp] for m in METHODS]
            ax.bar(x + (ci - 1) * width, vals, width=width,
                   color=CLASS_COLOURS[ci], edgecolor="black", linewidth=0.4,
                   label=cname)
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS],
                                              rotation=15, ha="right")
        ax.set_ylabel(comp)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9, title="Class")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Murphy decomposition: Brier = REL − RES + UNC  (UNC = 0.369 fixed across methods)\n"
        "RF-threshold has both the lowest REL and highest RES → best Brier",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_net_benefit(
    metrics: dict, out_path: Path = FIG_DIR / "step7_net_benefit.png",
) -> Path:
    """Vickers (2006) net-benefit curve for the Diversions alert, restricted to operationally useful τ."""
    TAU_MAX = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in METHODS:
        dc = metrics[m]["net_benefit_curve"]
        tau = np.asarray(dc["tau"]); nb = np.asarray(dc["nb_model"])
        mask = tau <= TAU_MAX
        ax.plot(tau[mask], nb[mask], "-",
                color=METHOD_COLOURS[m], linewidth=2.2, label=METHOD_LABELS[m])
    dc0 = metrics["rf_direct"]["net_benefit_curve"]
    tau0 = np.asarray(dc0["tau"]); nb_all = np.asarray(dc0["nb_treat_all"])
    mask0 = tau0 <= TAU_MAX
    ax.plot(tau0[mask0], nb_all[mask0], "--",
            color="black", linewidth=1.4, alpha=0.7,
            label="Treat all (alert every night)")
    ax.axhline(0, color="grey", linewidth=1.2, linestyle=":", alpha=0.7,
               label="Treat none (no alerts)")
    ax.axvline(0.15, color="grey", linewidth=0.7, linestyle=":", alpha=0.5)

    pi = dc0["base_rate_diversions"]
    ax.set_xlabel("Alert threshold τ on P(Diversions)\n"
                  "(implicit cost ratio: FP cost = τ/(1−τ) × FN cost)", fontsize=11)
    ax.set_ylabel("Net benefit per night")
    ax.set_title(f"Operational decision curve for the Diversions alert  "
                 f"(base rate π = {pi:.3f})\n"
                 "Curves above the zero line AND above the treat-all line are operationally useful at that τ",
                 fontsize=11)
    ax.set_xlim(0, TAU_MAX)
    ax.set_ylim(-0.10, 0.025)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_heatmap(
    metrics: dict, out_path: Path = FIG_DIR / "step7_summary_heatmap.png",
) -> Path:
    """Methods × metrics heatmap, coloured by within-column rank."""
    rows = []
    for m in METHODS:
        mm = metrics[m]["metrics"]
        dec = metrics[m]["brier_decomposition"]["summed"]
        adiv = next(a for a in mm["diversions_alert_curve"] if a["threshold"] == 0.15)
        rows.append({
            "Brier ↓": mm["multiclass_brier"],
            "REL ↓": dec["REL"],
            "RES ↑": dec["RES"],
            "sumECE ↓": mm["ece_sum"],
            "Acc ↑": mm["accuracy"],
            "BalAcc ↑": mm["balanced_accuracy"],
            "Macro F1 ↑": mm["macro_f1"],
            "Div F1 ↑": mm["per_class_f1"]["Diversions"],
            "Div recall@.15 ↑": adiv["recall"],
            "Div prec@.15 ↑": adiv["precision"],
        })
    df = pd.DataFrame(rows, index=[METHOD_LABELS[m] for m in METHODS])
    values = df.values
    # rank within column; flip sign for lower-is-better metrics
    lower_better = ["Brier ↓", "REL ↓", "sumECE ↓"]
    rank_values = np.empty_like(values)
    for j, col in enumerate(df.columns):
        v = values[:, j]
        if col in lower_better:
            rank_values[:, j] = pd.Series(v).rank(method="min", ascending=True).values
        else:
            rank_values[:, j] = pd.Series(v).rank(method="min", ascending=False).values

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = LinearSegmentedColormap.from_list("rank", ["#2E8B8B", "#FFFFFF"])
    im = ax.imshow(rank_values, cmap=cmap, aspect="auto", vmin=1, vmax=4)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = f"{values[i, j]:.3f}\n(#{int(rank_values[i, j])})"
            colour = "black" if rank_values[i, j] > 2 else "white"
            ax.text(j, i, txt, ha="center", va="center", color=colour, fontsize=9)
    ax.set_title("Step 7 summary: methods × metrics  (cell = value; (#) = rank within column, 1=best)",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("Rank (1=best)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    metrics, oof = load_data()
    for fn, args in [
        (plot_unified_reliability, (metrics, oof)),
        (plot_brier_decomposition, (metrics,)),
        (plot_net_benefit, (metrics,)),
        (plot_summary_heatmap, (metrics,)),
    ]:
        p = fn(*args)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
