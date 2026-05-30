"""Step 4 plotting routines. Each function takes saved OOF/metrics and writes one figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_models import PROJECT_ROOT
from cv_splits import FEATURE_COLUMNS, CLASS_NAMES

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# shared palette
COL_RF = "#3B6BB0"
COL_CLIM = "#7AAE63"
COL_PERSIST = "#C95C5C"
COL_NORMAL = "#78A678"
COL_DELAYS = "#E6A157"
COL_DIVERSIONS = "#B73A2B"
CLASS_COLOURS = [COL_NORMAL, COL_DELAYS, COL_DIVERSIONS]


def plot_regression_metrics(metrics: dict, out_path: Path = FIG_DIR / "step4_regression_metrics.png") -> Path:
    """Per-fold MAE/RMSE/R^2 bars: RF vs climatology vs persistence."""
    per_fold = metrics["per_fold"]
    folds = [f["fold_id"] for f in per_fold]
    models = [("RF", "rf_regression", COL_RF),
              ("Climatology", "climatology_regression", COL_CLIM),
              ("Persistence", "persistence_regression", COL_PERSIST)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.27
    x = np.arange(len(folds))

    metric_specs = [
        ("mae_m", "MAE (metres, lower is better)"),
        ("rmse_m", "RMSE (metres, lower is better)"),
        ("r2", "R² (higher is better)"),
    ]
    for ax, (key, title) in zip(axes, metric_specs):
        for i, (label, mkey, colour) in enumerate(models):
            vals = [f[mkey][key] for f in per_fold]
            ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=colour,
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in folds])
        ax.set_xlabel("Fold")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if key == "r2":
            ax.axhline(0, color="grey", linewidth=0.6)

    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.suptitle("Per-fold regression metrics across 8 forward-chaining folds", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_predicted_vs_actual(oof: pd.DataFrame, out_path: Path = FIG_DIR / "step4_predicted_vs_actual.png") -> Path:
    """RF pred vs truth scatter, coloured by true class, with 90% PI bands."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for cls in [0, 1, 2]:
        m = oof["true_class"] == cls
        ax.errorbar(
            oof.loc[m, "rf_pred_vis_m"], oof.loc[m, "true_vis_m"],
            xerr=1.6449 * oof.loc[m, "rf_pred_std_m"],
            fmt="o", markersize=4, alpha=0.5,
            color=CLASS_COLOURS[cls], label=CLASS_NAMES[cls],
            ecolor=CLASS_COLOURS[cls], elinewidth=0.6, capsize=0,
        )
    lo, hi = 0, max(oof["true_vis_m"].max(), oof["rf_pred_vis_m"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1.0, label="y = x")
    ax.set_xlabel("RF predicted morning min visibility (metres)  ± 90% PI", fontsize=11)
    ax.set_ylabel("True morning min visibility (metres)", fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title("RF predictions on out-of-fold test data (8 folds concatenated)", fontsize=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_classification_metrics(metrics: dict, out_path: Path = FIG_DIR / "step4_classification_metrics.png") -> Path:
    """Per-fold accuracy/balanced-acc/macro-F1 + per-class F1."""
    per_fold = metrics["per_fold"]
    folds = [f["fold_id"] for f in per_fold]
    clf = [f["rf_classification"] for f in per_fold]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(folds))

    # left: overall scores
    ax = axes[0]
    metrics_to_plot = [
        ("accuracy", "Accuracy", "#5B7FB9"),
        ("balanced_accuracy", "Balanced accuracy", "#E69F4C"),
        ("macro_f1", "Macro F1", "#A03C2D"),
    ]
    width = 0.27
    for i, (key, label, colour) in enumerate(metrics_to_plot):
        ax.bar(x + (i - 1) * width, [c[key] for c in clf],
               width=width, label=label, color=colour, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_xlabel("Fold"); ax.set_ylabel("Score")
    ax.set_title("Overall classification scores per fold")
    ax.set_ylim(0, 1); ax.axhline(1/3, color="grey", linewidth=0.6, linestyle=":")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # right: per-class F1
    ax = axes[1]
    for i, cls_name in enumerate(["Normal", "Delays", "Diversions"]):
        vals = [c["per_class_f1"][cls_name] for c in clf]
        ax.bar(x + (i - 1) * width, vals,
               width=width, label=cls_name, color=CLASS_COLOURS[i],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_xlabel("Fold"); ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 per fold (Diversions especially fragile)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(metrics: dict, out_path: Path = FIG_DIR / "step4_confusion_matrix.png") -> Path:
    """Row-normalised confusion matrix on the concatenated OOF predictions."""
    cm = np.array(metrics["aggregate"]["rf_classification"]["confusion"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Normal", "Delays", "Diversions"])
    ax.set_yticklabels(["Normal", "Delays", "Diversions"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("RF aggregate confusion matrix (8 folds, 1206 OOF nights)\n"
                 "rows normalised to 100% — each cell shows row-% and (count)",
                 fontsize=11)
    for i in range(3):
        for j in range(3):
            colour = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})",
                    ha="center", va="center", color=colour, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04, label="Row-normalised %")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(metrics: dict, out_path: Path = FIG_DIR / "step4_feature_importance.png") -> Path:
    """Gini + permutation importance side by side, mean ± 1-sigma across 8 folds."""
    fi = metrics["feature_importance"]
    feats = fi["features"]
    gini_mean = np.array(fi["gini_mean"])
    gini_std = np.array(fi["gini_std"])
    perm_mean = np.array(fi["perm_r2_mean"])
    perm_std = np.array(fi["perm_r2_std"])
    order = np.argsort(gini_mean)
    feats_sorted = [feats[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    y = np.arange(len(feats))

    axes[0].barh(y, gini_mean[order], xerr=gini_std[order],
                 color=COL_RF, edgecolor="black", linewidth=0.4, ecolor="grey")
    axes[0].set_yticks(y); axes[0].set_yticklabels(feats_sorted, fontsize=9)
    axes[0].set_xlabel("Gini importance (impurity decrease)")
    axes[0].set_title("Gini importance (RF internal, training-side)")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")

    axes[1].barh(y, perm_mean[order], xerr=perm_std[order],
                 color=COL_CLIM, edgecolor="black", linewidth=0.4, ecolor="grey")
    axes[1].set_yticks(y); axes[1].set_yticklabels(feats_sorted, fontsize=9)
    axes[1].set_xlabel("Permutation importance (R² drop on held-out fold)")
    axes[1].set_title("Permutation importance (held-out, model-agnostic)")
    axes[1].axvline(0, color="grey", linewidth=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle("RF regressor feature importance, averaged across 8 forward-chaining folds (± 1σ)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reliability_diagram(oof: pd.DataFrame, out_path: Path = FIG_DIR / "step4_reliability_diagram.png") -> Path:
    """Per-class reliability diagrams + per-bin sample counts."""
    proba_cols = ["rf_proba_normal", "rf_proba_delays", "rf_proba_diversions"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [3, 1]})

    bins = np.linspace(0, 1, 11)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    for k, (col, cls_name, colour) in enumerate(zip(proba_cols, CLASS_NAMES.values(), CLASS_COLOURS)):
        proba = oof[col].to_numpy()
        true = (oof["true_class"] == k).to_numpy().astype(float)
        ax = axes[0, k]
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect calibration")
        bin_idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
        mean_pred, mean_true, counts = [], [], []
        for b in range(10):
            mask = bin_idx == b
            counts.append(mask.sum())
            if mask.sum() > 0:
                mean_pred.append(proba[mask].mean())
                mean_true.append(true[mask].mean())
            else:
                mean_pred.append(np.nan)
                mean_true.append(np.nan)
        ax.plot(mean_pred, mean_true, "o-", color=colour, linewidth=1.6,
                markersize=7, label=f"RF — {cls_name}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"{cls_name}  (n positives: {int(true.sum())})")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        # bottom: bin counts
        axh = axes[1, k]
        axh.bar(bin_centres, counts, width=0.09, color=colour, edgecolor="black",
                linewidth=0.4, alpha=0.7)
        axh.set_xlim(0, 1)
        axh.set_xlabel("Predicted probability bin")
        axh.set_ylabel("# samples")
        axh.spines[["top", "right"]].set_visible(False)
        axh.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("RF reliability diagrams — per-class calibration on out-of-fold data\n"
                 "(headline figure: the gap between curve and diagonal is the calibration error)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Regenerate all Step 4 figures from saved OOF + metrics files."""
    oof = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "rf_oof_predictions.parquet")
    metrics = json.loads((PROJECT_ROOT / "data" / "processed" / "step4_metrics.json").read_text())
    for fn in [
        plot_regression_metrics,
        plot_predicted_vs_actual,
        plot_classification_metrics,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_reliability_diagram,
    ]:
        if fn in (plot_regression_metrics, plot_classification_metrics,
                  plot_confusion_matrix, plot_feature_importance):
            p = fn(metrics)
        else:
            p = fn(oof)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
