"""Step 6 plotting routines. Compares RF, Direct SVGP, and Threshold-derived GP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import CLASS_NAMES  # noqa: E402
from gp_classification import (  # noqa: E402
    classwise_ece, multiclass_brier, reliability_curve,
    diversions_alert_curve, ALERT_THRESHOLDS,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# RF blue, direct GP purple, threshold-derived GP teal
COL_RF = "#3B6BB0"
COL_SVGP = "#7F4A89"
COL_THRESH = "#2E8B8B"
COL_NORMAL = "#78A678"
COL_DELAYS = "#E6A157"
COL_DIVERSIONS = "#B73A2B"
CLASS_COLOURS = [COL_NORMAL, COL_DELAYS, COL_DIVERSIONS]

METHOD_ORDER = [
    ("RF (Step 4)", "rf", COL_RF, "s"),
    ("Direct SVGP", "svgp", COL_SVGP, "o"),
    ("Threshold-derived GP", "thresh", COL_THRESH, "^"),
]


def load_all() -> dict:
    """Load all OOF parquets and metrics JSONs needed for the figures."""
    proc = PROJECT_ROOT / "data" / "processed"
    rf_oof = pd.read_parquet(proc / "rf_oof_predictions.parquet")
    gp_clf_oof = pd.read_parquet(proc / "gp_clf_oof_predictions.parquet")
    step4 = json.loads((proc / "step4_metrics.json").read_text())
    step6 = json.loads((proc / "step6_metrics.json").read_text())

    rf_proba = rf_oof[["rf_proba_normal", "rf_proba_delays", "rf_proba_diversions"]].to_numpy()
    rf_yt = rf_oof["true_class"].to_numpy()
    svgp_proba = gp_clf_oof[["svgp_proba_normal", "svgp_proba_delays", "svgp_proba_diversions"]].to_numpy()
    thresh_proba = gp_clf_oof[["thresh_proba_normal", "thresh_proba_delays", "thresh_proba_diversions"]].to_numpy()
    yt = gp_clf_oof["true_class"].to_numpy()
    return {
        "rf_yt": rf_yt, "rf_proba": rf_proba,
        "yt": yt, "svgp_proba": svgp_proba, "thresh_proba": thresh_proba,
        "rf_oof": rf_oof, "gp_clf_oof": gp_clf_oof,
        "step4": step4, "step6": step6,
    }


def plot_reliability_diagrams(
    data: dict, out_path: Path = FIG_DIR / "step6_reliability_diagrams.png"
) -> Path:
    """Per-class reliability curves overlaid for all 3 methods (THE headline)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                              gridspec_kw={"height_ratios": [3, 1]})

    methods = [
        ("RF", data["rf_yt"], data["rf_proba"], COL_RF, "s"),
        ("Direct SVGP", data["yt"], data["svgp_proba"], COL_SVGP, "o"),
        ("Threshold-derived GP", data["yt"], data["thresh_proba"], COL_THRESH, "^"),
    ]
    for k, cls_name in enumerate(["Normal", "Delays", "Diversions"]):
        ax = axes[0, k]
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        for label, yt, pr, colour, marker in methods:
            mp, ef, _ = reliability_curve(yt, pr, k)
            ax.plot(mp, ef, marker=marker, linestyle="-",
                    color=colour, linewidth=1.8, markersize=7, label=label, alpha=0.9)
        n_pos = int((data["yt"] == k).sum())
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"{cls_name}  (n true positives: {n_pos})")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        # bottom: bin counts for the threshold-derived method
        axh = axes[1, k]
        _, _, counts = reliability_curve(data["yt"], data["thresh_proba"], k)
        bin_centres = 0.5 * (np.linspace(0, 1, 11)[:-1] + np.linspace(0, 1, 11)[1:])
        axh.bar(bin_centres, counts, width=0.085,
                color=CLASS_COLOURS[k], edgecolor="black", linewidth=0.3, alpha=0.7)
        axh.set_xlim(0, 1); axh.set_xlabel("Predicted prob bin (Thresh-GP)")
        axh.set_ylabel("# samples"); axh.grid(axis="y", alpha=0.3, linestyle="--")
        axh.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Per-class reliability diagrams: RF vs Direct SVGP vs Threshold-derived GP\n"
                 "(closer to diagonal = better calibrated; bottom histograms show "
                 "predicted-probability distribution for the threshold-derived method)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_diversions_alert_curve(
    data: dict, out_path: Path = FIG_DIR / "step6_diversions_alert.png"
) -> Path:
    """Recall vs alert threshold + precision-recall trade-off for the Diversions alert."""
    methods = [
        ("RF", data["rf_yt"], data["rf_proba"], COL_RF, "s"),
        ("Direct SVGP", data["yt"], data["svgp_proba"], COL_SVGP, "o"),
        ("Threshold-derived GP", data["yt"], data["thresh_proba"], COL_THRESH, "^"),
    ]
    dense = np.linspace(0.005, 0.95, 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for label, yt, pr, colour, marker in methods:
        rec = []
        for tau in dense:
            flag = pr[:, 2] > tau
            rec.append((flag & (yt == 2)).sum() / max((yt == 2).sum(), 1))
        ax.plot(dense, rec, "-", color=colour, linewidth=2, label=label)
    ax.set_xlabel("Alert threshold τ on P(Diversions)")
    ax.set_ylabel("Diversions recall")
    ax.set_title("Recall vs alert threshold\n(at τ=0.15 the threshold-derived GP catches 72% of Diversions)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axvline(0.15, color="grey", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for label, yt, pr, colour, marker in methods:
        rec, prec = [], []
        for tau in dense:
            flag = pr[:, 2] > tau
            tp = int((flag & (yt == 2)).sum())
            n_flag = int(flag.sum())
            n_pos = int((yt == 2).sum())
            if n_flag > 0:
                rec.append(tp / max(n_pos, 1))
                prec.append(tp / n_flag)
        ax.plot(rec, prec, "-", color=colour, linewidth=2, label=label, alpha=0.9)
    ax.set_xlabel("Diversions recall")
    ax.set_ylabel("Diversions precision")
    ax.set_title("Precision–recall trade-off for the Diversions alert\n(further to the upper-right is better)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    data: dict, out_path: Path = FIG_DIR / "step6_confusion_matrices.png"
) -> Path:
    """Three row-normalised confusion matrices side by side."""
    rf_cm = np.array(data["step4"]["aggregate"]["rf_classification"]["confusion"])
    svgp_cm = np.array(data["step6"]["aggregate"]["svgp"]["confusion"])
    thr_cm = np.array(data["step6"]["aggregate"]["threshold_derived"]["confusion"])
    panels = [("RF (Step 4)", rf_cm), ("Direct SVGP", svgp_cm),
              ("Threshold-derived GP", thr_cm)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, cm) in zip(axes, panels):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Normal", "Delays", "Diversions"])
        ax.set_yticklabels(["Normal", "Delays", "Diversions"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
        for i in range(3):
            for j in range(3):
                colour = "white" if pct[i, j] > 50 else "black"
                ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]})",
                        ha="center", va="center", color=colour, fontsize=10)
    fig.suptitle("Row-normalised confusion matrices on 1206 OOF nights",
                 fontsize=12, y=1.03)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="row-%")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_fold_metrics(
    data: dict, out_path: Path = FIG_DIR / "step6_per_fold_metrics.png"
) -> Path:
    """Per-fold Brier / balanced acc / macro F1 / Diversions F1 across the 3 methods."""
    rf_pf = data["step4"]["per_fold"]
    s6_pf = data["step6"]["per_fold"]
    folds = [r["fold_id"] for r in s6_pf]
    x = np.arange(len(folds))
    width = 0.27

    def get(m_list, key_path):
        out = []
        for r in m_list:
            v = r
            for k in key_path:
                v = v[k]
            out.append(v)
        return np.array(out)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("Multi-class Brier (lower is better)",
         get(rf_pf, ["rf_classification", "multiclass_brier"]),
         get(s6_pf, ["svgp", "multiclass_brier"]),
         get(s6_pf, ["threshold_derived", "multiclass_brier"])),
        ("Balanced accuracy (higher is better)",
         get(rf_pf, ["rf_classification", "balanced_accuracy"]),
         get(s6_pf, ["svgp", "balanced_accuracy"]),
         get(s6_pf, ["threshold_derived", "balanced_accuracy"])),
        ("Macro F1 (higher is better)",
         get(rf_pf, ["rf_classification", "macro_f1"]),
         get(s6_pf, ["svgp", "macro_f1"]),
         get(s6_pf, ["threshold_derived", "macro_f1"])),
        ("Diversions F1 (higher is better)",
         np.array([r["rf_classification"]["per_class_f1"]["Diversions"] for r in rf_pf]),
         np.array([r["svgp"]["per_class_f1"]["Diversions"] for r in s6_pf]),
         np.array([r["threshold_derived"]["per_class_f1"]["Diversions"] for r in s6_pf])),
    ]
    for ax, (title, rf, svgp, thr) in zip(axes.ravel(), panels):
        ax.bar(x - width, rf, width=width, color=COL_RF, edgecolor="black", linewidth=0.4, label="RF")
        ax.bar(x, svgp, width=width, color=COL_SVGP, edgecolor="black", linewidth=0.4, label="Direct SVGP")
        ax.bar(x + width, thr, width=width, color=COL_THRESH, edgecolor="black", linewidth=0.4, label="Thresh-GP")
        ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds])
        ax.set_xlabel("Fold"); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Per-fold classification metrics: RF vs Direct SVGP vs Threshold-derived GP",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_calibration_decomposition(
    data: dict, out_path: Path = FIG_DIR / "step6_calibration_decomposition.png"
) -> Path:
    """Aggregate Brier + per-class ECE side by side. Brier mixes calibration+sharpness; ECE isolates calibration."""
    rf_yt = data["rf_yt"]; rf_pr = data["rf_proba"]
    yt = data["yt"]; svgp = data["svgp_proba"]; thr = data["thresh_proba"]

    method_data = [
        ("RF", rf_yt, rf_pr, COL_RF),
        ("Direct SVGP", yt, svgp, COL_SVGP),
        ("Thresh-GP", yt, thr, COL_THRESH),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: Brier per method
    ax = axes[0]
    brier_vals = [multiclass_brier(y, p) for _, y, p, _ in method_data]
    ax.bar(range(3), brier_vals,
           color=[c for *_, c in method_data],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(3)); ax.set_xticklabels([m[0] for m in method_data])
    ax.set_ylabel("Multi-class Brier score")
    ax.set_title("Aggregate Brier (lower is better)")
    for i, v in enumerate(brier_vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # right: per-class ECE
    ax = axes[1]
    x = np.arange(3)
    width = 0.27
    for i, (name, y, p, c) in enumerate(method_data):
        ece = [classwise_ece(y, p, k) for k in range(3)]
        ax.bar(x + (i - 1) * width, ece, width=width, color=c, label=name,
               edgecolor="black", linewidth=0.4)
        for k, v in enumerate(ece):
            ax.text(k + (i - 1) * width, v + 0.003, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Normal", "Delays", "Diversions"])
    ax.set_ylabel("One-vs-rest ECE")
    ax.set_title("Per-class calibration error (lower is better)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Calibration decomposition: Brier vs per-class ECE\n"
                 "(threshold-derived GP wins summed ECE; RF wins Brier through sharpness on Normal)",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    data = load_all()
    for fn in [
        plot_reliability_diagrams,
        plot_diversions_alert_curve,
        plot_confusion_matrices,
        plot_per_fold_metrics,
        plot_calibration_decomposition,
    ]:
        p = fn(data)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
