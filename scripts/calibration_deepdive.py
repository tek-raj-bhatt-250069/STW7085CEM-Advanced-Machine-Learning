"""Step 7: cross-model calibration deep-dive. Compares 4 classifiers on the same OOF nights."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import CLASS_NAMES, setup_logging  # noqa: E402
from gp_classification import (  # noqa: E402
    THRESH_DIVERSIONS_M, THRESH_DELAYS_M, ALERT_THRESHOLDS,
    multiclass_brier, classwise_ece, reliability_curve,
    diversions_alert_curve, classification_metrics,
)

PROC = PROJECT_ROOT / "data" / "processed"
DEFAULT_METRICS_PATH = PROC / "step7_metrics.json"
DEFAULT_TABLES_DIR = PROC / "step7_tables"

# fixed display order for tables/figures
METHODS = ["rf_direct", "direct_svgp", "rf_threshold", "gp_threshold"]
METHOD_LABELS = {
    "rf_direct": "RF-direct",
    "direct_svgp": "Direct SVGP",
    "rf_threshold": "RF-threshold",
    "gp_threshold": "GP-threshold",
}

log = setup_logging(); log.name = "calibration_deepdive"


def gaussian_threshold_proba(mean_m: np.ndarray, std_m: np.ndarray,
                              t_div: float = THRESH_DIVERSIONS_M,
                              t_del: float = THRESH_DELAYS_M) -> np.ndarray:
    """3-class probs from Gaussian-in-metres regression posterior."""
    std_m = np.maximum(std_m, 1.0)  # avoid zero
    z_div = (t_div - mean_m) / std_m
    z_del = (t_del - mean_m) / std_m
    p_div = norm.cdf(z_div)
    p_del = norm.cdf(z_del) - norm.cdf(z_div)
    p_nrm = 1.0 - norm.cdf(z_del)
    proba = np.clip(np.stack([p_nrm, p_del, p_div], axis=1), 1e-9, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def build_method_proba_table() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Join Steps 4-6 OOFs on date_npt; one (N,3) prob array per method."""
    rf_oof = pd.read_parquet(PROC / "rf_oof_predictions.parquet")
    gp_clf_oof = pd.read_parquet(PROC / "gp_clf_oof_predictions.parquet")
    rf_oof["date_npt"] = pd.to_datetime(rf_oof["date_npt"])
    gp_clf_oof["date_npt"] = pd.to_datetime(gp_clf_oof["date_npt"])

    # RF-threshold derived from RF regression posterior
    rf_thresh = gaussian_threshold_proba(
        rf_oof["rf_pred_vis_m"].to_numpy(), rf_oof["rf_pred_std_m"].to_numpy(),
    )
    rf_oof["rf_threshold_proba_normal"] = rf_thresh[:, 0]
    rf_oof["rf_threshold_proba_delays"] = rf_thresh[:, 1]
    rf_oof["rf_threshold_proba_diversions"] = rf_thresh[:, 2]

    merged = rf_oof.merge(gp_clf_oof, on="date_npt", suffixes=("", "_clf"))
    assert (merged["true_class"] == merged["true_class_clf"]).all(), \
        "true_class mismatch between RF and GP classification OOFs"
    merged = merged.drop(columns=["true_class_clf"])

    y_true = merged["true_class"].to_numpy()
    proba = {
        "rf_direct":    merged[["rf_proba_normal", "rf_proba_delays", "rf_proba_diversions"]].to_numpy(),
        "direct_svgp":  merged[["svgp_proba_normal", "svgp_proba_delays", "svgp_proba_diversions"]].to_numpy(),
        "rf_threshold": merged[["rf_threshold_proba_normal", "rf_threshold_proba_delays", "rf_threshold_proba_diversions"]].to_numpy(),
        "gp_threshold": merged[["thresh_proba_normal", "thresh_proba_delays", "thresh_proba_diversions"]].to_numpy(),
    }
    return merged, proba, y_true


def brier_decomposition_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy decomposition: BS = REL - RES + UNC + residual."""
    N = len(y_true)
    o_bar = float(y_true.mean())
    UNC = o_bar * (1.0 - o_bar)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    REL = RES = 0.0
    for k in range(n_bins):
        m = idx == k
        n_k = int(m.sum())
        if n_k == 0:
            continue
        w = n_k / N
        p_bar_k = float(p[m].mean())
        o_bar_k = float(y_true[m].mean())
        REL += w * (p_bar_k - o_bar_k) ** 2
        RES += w * (o_bar_k - o_bar) ** 2
    BS = float(((p - y_true) ** 2).mean())
    residual = BS - (REL - RES + UNC)
    return {"BS": BS, "REL": REL, "RES": RES, "UNC": UNC, "residual": float(residual)}


def murphy_decomposition_multiclass(y_true: np.ndarray, proba: np.ndarray,
                                     n_bins: int = 10) -> dict:
    """Per-class + summed Murphy decomposition; multi-class Brier is additive across classes."""
    per_class = {}
    for c in range(3):
        binary = (y_true == c).astype(float)
        per_class[CLASS_NAMES[c]] = brier_decomposition_binary(binary, proba[:, c], n_bins=n_bins)
    total = {k: sum(per_class[CLASS_NAMES[c]][k] for c in range(3))
             for k in ["BS", "REL", "RES", "UNC", "residual"]}
    return {"per_class": per_class, "summed": total}


def net_benefit(y_true: np.ndarray, p_div: np.ndarray, taus: np.ndarray) -> dict:
    """Vickers (2006) net benefit at each alert threshold tau on P(Diversions)."""
    N = len(y_true)
    pi = float((y_true == 2).mean())
    nb_model = np.empty_like(taus, dtype=float)
    nb_all = np.empty_like(taus, dtype=float)
    for i, tau in enumerate(taus):
        flag = p_div > tau
        tp = int((flag & (y_true == 2)).sum())
        fp = int((flag & (y_true != 2)).sum())
        w = tau / (1 - tau) if tau < 1 else np.inf
        nb_model[i] = tp / N - fp / N * w
        nb_all[i] = pi - (1 - pi) * w
    return {"tau": taus.tolist(),
            "nb_model": nb_model.tolist(),
            "nb_treat_all": nb_all.tolist(),
            "base_rate_diversions": pi}


@dataclass
class MethodResult:
    name: str
    proba: np.ndarray
    metrics: dict
    decomposition: dict
    decision_curve: dict


def compute_all(proba_by_method: dict[str, np.ndarray], y_true: np.ndarray) -> dict:
    """Run the full Step 7 suite across all methods."""
    taus = np.linspace(0.005, 0.95, 60)
    results = {}
    for name in METHODS:
        proba = proba_by_method[name]
        m = classification_metrics(y_true, proba)
        dec = murphy_decomposition_multiclass(y_true, proba)
        dc = net_benefit(y_true, proba[:, 2], taus)
        results[name] = {
            "label": METHOD_LABELS[name],
            "metrics": m,
            "brier_decomposition": dec,
            "net_benefit_curve": dc,
        }
    return results


def build_summary_table(all_results: dict) -> pd.DataFrame:
    """One row per method, every headline metric the paper uses."""
    rows = []
    for name in METHODS:
        r = all_results[name]
        m = r["metrics"]
        dec = r["brier_decomposition"]["summed"]
        adiv = next(a for a in m["diversions_alert_curve"] if a["threshold"] == 0.15)
        rows.append({
            "Method": r["label"],
            "Brier": m["multiclass_brier"],
            "REL": dec["REL"],
            "RES": dec["RES"],
            "BS-residual": dec["residual"],
            "sumECE": m["ece_sum"],
            "Accuracy": m["accuracy"],
            "BalAcc": m["balanced_accuracy"],
            "MacroF1": m["macro_f1"],
            "DivF1": m["per_class_f1"]["Diversions"],
            "Div recall@0.15": adiv["recall"],
            "Div precision@0.15": adiv["precision"],
        })
    df = pd.DataFrame(rows).set_index("Method")
    return df


def build_decomposition_table(all_results: dict) -> pd.DataFrame:
    """Per-class REL/RES/UNC table for the Discussion section."""
    rows = []
    for name in METHODS:
        for cls in CLASS_NAMES.values():
            d = all_results[name]["brier_decomposition"]["per_class"][cls]
            rows.append({"Method": METHOD_LABELS[name], "Class": cls,
                          "BS": d["BS"], "REL": d["REL"], "RES": d["RES"], "UNC": d["UNC"]})
    return pd.DataFrame(rows)


def write_tables(all_results: dict, out_dir: Path) -> dict[str, Path]:
    """Dump summary tables as Markdown + LaTeX + CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_table(all_results)
    decomp = build_decomposition_table(all_results)
    paths = {}
    for name, df in [("summary", summary), ("decomposition", decomp)]:
        paths[f"{name}_csv"] = out_dir / f"{name}.csv"
        df.to_csv(paths[f"{name}_csv"])
        paths[f"{name}_md"] = out_dir / f"{name}.md"
        paths[f"{name}_md"].write_text(df.round(3).to_markdown())
        paths[f"{name}_tex"] = out_dir / f"{name}.tex"
        paths[f"{name}_tex"].write_text(df.round(3).to_latex(float_format="%.3f"))
    return paths


def main() -> None:
    log.info("Step 7: cross-model calibration deep-dive")
    merged, proba_by_method, y_true = build_method_proba_table()
    log.info(f"Joined OOF table: {len(merged)} nights × 4 methods")

    # diagnostic: how much RF-threshold mass leaks into vis<0
    mu = merged["rf_pred_vis_m"].to_numpy()
    sd = np.maximum(merged["rf_pred_std_m"].to_numpy(), 1.0)
    p_neg = norm.cdf(-mu / sd)
    log.info(f"RF-threshold P(vis<0) caveat: mean={p_neg.mean():.4f}, "
             f"max={p_neg.max():.3f}, fraction>0.05={np.mean(p_neg > 0.05):.3f}")

    all_results = compute_all(proba_by_method, y_true)

    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS_PATH.write_text(json.dumps(all_results, indent=2, default=float))
    log.info(f"Wrote {DEFAULT_METRICS_PATH}")

    cross_oof = merged[["date_npt", "fold_id", "true_class"]].copy()
    for name in METHODS:
        for c, cls in enumerate(["normal", "delays", "diversions"]):
            cross_oof[f"{name}_proba_{cls}"] = proba_by_method[name][:, c]
    cross_oof_path = PROC / "step7_cross_method_oof.parquet"
    cross_oof.to_parquet(cross_oof_path, index=False)
    log.info(f"Wrote {cross_oof_path}")

    paths = write_tables(all_results, DEFAULT_TABLES_DIR)
    log.info(f"Wrote summary tables to {DEFAULT_TABLES_DIR}/")

    summary = build_summary_table(all_results)
    print()
    print("=" * 88)
    print("STEP 7 HEADLINE SUMMARY TABLE")
    print("=" * 88)
    print(summary.round(3).to_string())
    print()
    print("REL = reliability (lower = better calibrated).  RES = resolution (higher = more informative).")
    print(f"UNC (per-class summed) = {sum(all_results['rf_direct']['brier_decomposition']['per_class'][c]['UNC'] for c in CLASS_NAMES.values()):.4f}  (identical across methods)")


if __name__ == "__main__":
    main()
