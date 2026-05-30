"""Step 5 plotting routines. Consumes saved Step 4 + Step 5 outputs; no re-training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import CLASS_NAMES  # noqa: E402

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# palette matches Step 4
COL_RF = "#3B6BB0"
COL_GP = "#7F4A89"
COL_CLIM = "#7AAE63"
COL_PERSIST = "#C95C5C"
COL_NORMAL = "#78A678"
COL_DELAYS = "#E6A157"
COL_DIVERSIONS = "#B73A2B"
CLASS_COLOURS = [COL_NORMAL, COL_DELAYS, COL_DIVERSIONS]

CALIB_LEVELS = np.linspace(0.1, 0.9, 9)


def rf_calibration_from_oof(
    rf_oof: pd.DataFrame, levels: np.ndarray = CALIB_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RF calibration curve from the Step 4 OOF (Gaussian-PI assumption)."""
    y = rf_oof["true_vis_m"].to_numpy()
    mu = rf_oof["rf_pred_vis_m"].to_numpy()
    sd = rf_oof["rf_pred_std_m"].to_numpy()
    empirical = []
    for alpha in levels:
        z = norm.ppf(0.5 + alpha / 2)
        empirical.append(float(((y >= mu - z * sd) & (y <= mu + z * sd)).mean()))
    return np.asarray(levels, dtype=float), np.asarray(empirical)


def plot_regression_metrics(
    gp_metrics: dict, rf_metrics: dict,
    out_path: Path = FIG_DIR / "step5_regression_metrics.png",
) -> Path:
    """Per-fold MAE/RMSE/R^2 with GP + RF + climatology + persistence."""
    gp_pf = gp_metrics["per_fold"]
    rf_pf = rf_metrics["per_fold"]
    folds = [r["fold_id"] for r in gp_pf]
    x = np.arange(len(folds))
    width = 0.20

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (key, title) in zip(axes, [
        ("mae_m", "MAE (metres, lower is better)"),
        ("rmse_m", "RMSE (metres, lower is better)"),
        ("r2", "R² (higher is better)"),
    ]):
        ax.bar(x - 1.5 * width, [r["regression"][key] for r in gp_pf],
               width=width, color=COL_GP, edgecolor="black", linewidth=0.4, label="GP")
        ax.bar(x - 0.5 * width, [r["rf_regression"][key] for r in rf_pf],
               width=width, color=COL_RF, edgecolor="black", linewidth=0.4, label="RF")
        ax.bar(x + 0.5 * width, [r["climatology_regression"][key] for r in rf_pf],
               width=width, color=COL_CLIM, edgecolor="black", linewidth=0.4, label="Climatology")
        ax.bar(x + 1.5 * width, [r["persistence_regression"][key] for r in rf_pf],
               width=width, color=COL_PERSIST, edgecolor="black", linewidth=0.4, label="Persistence")
        ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds])
        ax.set_xlabel("Fold"); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if key == "r2":
            ax.axhline(0, color="grey", linewidth=0.6)

    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.suptitle("Per-fold regression metrics: GP vs RF vs baselines across 8 forward-chaining folds",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_predicted_vs_actual(
    gp_oof: pd.DataFrame,
    out_path: Path = FIG_DIR / "step5_predicted_vs_actual.png",
) -> Path:
    """GP median vs truth; asymmetric x-error bars from log-Normal back-transform."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for cls in [0, 1, 2]:
        m = gp_oof["true_class"] == cls
        med = gp_oof.loc[m, "gp_median_vis_m"].to_numpy()
        lo = gp_oof.loc[m, "gp_pi90_lo_m"].to_numpy()
        hi = gp_oof.loc[m, "gp_pi90_hi_m"].to_numpy()
        ax.errorbar(
            med, gp_oof.loc[m, "true_vis_m"],
            xerr=np.stack([med - lo, hi - med]),  # asymmetric
            fmt="o", markersize=4, alpha=0.55,
            color=CLASS_COLOURS[cls], label=CLASS_NAMES[cls],
            ecolor=CLASS_COLOURS[cls], elinewidth=0.6, capsize=0,
        )
    hi_lim = max(gp_oof["true_vis_m"].max(), gp_oof["gp_pi90_hi_m"].max()) * 1.05
    ax.plot([0, hi_lim], [0, hi_lim], "--", color="grey", linewidth=1.0, label="y = x")
    ax.set_xlabel("GP predicted morning min visibility (median, metres)  ± 90% PI", fontsize=11)
    ax.set_ylabel("True morning min visibility (metres)", fontsize=11)
    ax.set_xlim(0, hi_lim); ax.set_ylim(0, hi_lim)
    ax.set_title("GP predictions on out-of-fold test data (8 folds concatenated)\n"
                 "Asymmetric PI: log-Normal back-transform produces longer upper tails",
                 fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_calibration_curve(
    gp_metrics: dict, rf_oof: pd.DataFrame,
    out_path: Path = FIG_DIR / "step5_calibration_curve.png",
) -> Path:
    """Reliability diagram for regression intervals (THE paper headline)."""
    calib = gp_metrics["calibration"]
    nominal = np.array(calib["nominal_levels"])
    gp_weighted = np.array(calib["weighted_empirical"])
    gp_per_fold = np.array(calib["per_fold_empirical"])
    _, rf_emp = rf_calibration_from_oof(rf_oof, levels=nominal)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw={"width_ratios": [1, 1]})

    # left: aggregate GP vs RF
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect calibration")
    ax.plot(nominal, gp_weighted, "o-", color=COL_GP, linewidth=2, markersize=8, label="GP")
    ax.plot(nominal, rf_emp, "s-", color=COL_RF, linewidth=2, markersize=7, label="RF (Step 4)")
    ax.set_xlabel("Nominal coverage level α", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Predictive-interval calibration on OOF data\n(GP near diagonal, RF below = over-confident)",
                 fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # right: per-fold GP curves to show it's not one lucky fold
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    for k in range(gp_per_fold.shape[0]):
        ax.plot(nominal, gp_per_fold[k], "-", color=COL_GP, alpha=0.35, linewidth=1)
    ax.plot(nominal, gp_weighted, "o-", color=COL_GP, linewidth=2.5,
            markersize=8, label="Weighted aggregate")
    ax.set_xlabel("Nominal coverage level α", fontsize=11)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("GP per-fold calibration curves (8 thin lines)\n+ weighted aggregate",
                 fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ard_lengthscales(
    gp_metrics: dict, rf_metrics: dict,
    out_path: Path = FIG_DIR / "step5_ard_lengthscales.png",
) -> Path:
    """ARD lengthscale per feature (median + IQR), log scale. Smaller = more relevant."""
    ard = gp_metrics["ard"]
    features = ard["features"]
    ls_per_fold = np.array(ard["lengthscales_per_fold"])

    # floor so log-scale handles near-zero lengthscales from degenerate folds
    floor = 1e-2
    ls_per_fold_c = np.clip(ls_per_fold, floor, None)
    ls_median = np.median(ls_per_fold_c, axis=0)
    ls_q25 = np.percentile(ls_per_fold_c, 25, axis=0)
    ls_q75 = np.percentile(ls_per_fold_c, 75, axis=0)

    rf_gini = np.array(rf_metrics["feature_importance"]["gini_mean"])
    rf_gini_std = np.array(rf_metrics["feature_importance"]["gini_std"])

    order = np.argsort(ls_median)  # smallest (most relevant) first
    feats_sorted = [features[i] for i in order]
    y = np.arange(len(features))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # left: GP median lengthscale (log x)
    ax = axes[0]
    xerr = np.stack([
        ls_median[order] - ls_q25[order],
        ls_q75[order] - ls_median[order],
    ])
    xerr = np.clip(xerr, 0, None)
    ax.barh(y, ls_median[order], color=COL_GP, edgecolor="black", linewidth=0.4)
    ax.errorbar(ls_median[order], y, xerr=xerr, fmt="none",
                ecolor="grey", elinewidth=1.0, capsize=2)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(feats_sorted, fontsize=9)
    ax.set_xlabel("GP ARD lengthscale, log scale  (SMALLER = more relevant)")
    ax.set_title("GP-ARD lengthscale per feature\n(median ± IQR across 8 folds)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--", which="both")
    ax.invert_yaxis()

    # right: RF Gini on the same ordering
    ax = axes[1]
    ax.barh(y, rf_gini[order], xerr=rf_gini_std[order],
            color=COL_RF, edgecolor="black", linewidth=0.4, ecolor="grey")
    ax.set_yticks(y); ax.set_yticklabels(feats_sorted, fontsize=9)
    ax.set_xlabel("RF Gini importance (Step 4)  (LARGER = more relevant)")
    ax.set_title("RF Gini importance on the same feature ordering\n"
                 "(agreement = GP-relevant features also have high RF Gini)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()

    fig.suptitle("Feature relevance: GP-ARD lengthscale (sorted) vs RF Gini (same ordering)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pi_sharpness(
    gp_oof: pd.DataFrame, rf_oof: pd.DataFrame,
    out_path: Path = FIG_DIR / "step5_pi_sharpness.png",
) -> Path:
    """PI-width distributions + per-night GP/RF width ratio."""
    merged = gp_oof.merge(
        rf_oof[["date_npt", "rf_pi90_lo_m", "rf_pi90_hi_m"]],
        on="date_npt", how="inner",
    )
    gp_width = (merged["gp_pi90_hi_m"] - merged["gp_pi90_lo_m"]).to_numpy()
    rf_width = (merged["rf_pi90_hi_m"] - merged["rf_pi90_lo_m"]).to_numpy()
    ratio = gp_width / np.maximum(rf_width, 1.0)  # avoid div-by-zero

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    bins = np.linspace(0, max(gp_width.max(), rf_width.max()), 60)
    ax.hist(rf_width, bins=bins, color=COL_RF, alpha=0.55, label=f"RF (mean={rf_width.mean():.0f}m)",
            edgecolor="black", linewidth=0.3)
    ax.hist(gp_width, bins=bins, color=COL_GP, alpha=0.55, label=f"GP (mean={gp_width.mean():.0f}m)",
            edgecolor="black", linewidth=0.3)
    ax.set_xlabel("90% PI width (metres)")
    ax.set_ylabel("Number of OOF nights")
    ax.set_title("PI width distribution: GP vs RF on identical OOF nights")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.hist(ratio, bins=60, color="#555555", edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", label="ratio = 1")
    ax.axvline(float(np.median(ratio)), color=COL_GP, linewidth=1.4,
               linestyle=":", label=f"median = {np.median(ratio):.2f}")
    ax.set_xlabel("Per-night PI width ratio: GP / RF")
    ax.set_ylabel("Number of OOF nights")
    ax.set_title("Per-night sharpness trade-off")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Regenerate all Step 5 figures from saved outputs."""
    proc = PROJECT_ROOT / "data" / "processed"
    gp_oof = pd.read_parquet(proc / "gp_oof_predictions.parquet")
    rf_oof = pd.read_parquet(proc / "rf_oof_predictions.parquet")
    gp_metrics = json.loads((proc / "step5_metrics.json").read_text())
    rf_metrics = json.loads((proc / "step4_metrics.json").read_text())

    for fn, args in [
        (plot_regression_metrics, (gp_metrics, rf_metrics)),
        (plot_predicted_vs_actual, (gp_oof,)),
        (plot_calibration_curve, (gp_metrics, rf_oof)),
        (plot_ard_lengthscales, (gp_metrics, rf_metrics)),
        (plot_pi_sharpness, (gp_oof, rf_oof)),
    ]:
        p = fn(*args)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
