"""Step 6: multi-class GP classification - direct SVGP vs threshold-derived from Step 5 regression."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import gpflow  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
    CLASS_NAMES,
    load_modelling_table,
    split_holdout,
    iter_prepared_folds,
    setup_logging,
)


DEFAULT_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "gp_clf_oof_predictions.parquet"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "step6_metrics.json"
GP_REG_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "gp_oof_predictions.parquet"
RF_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "rf_oof_predictions.parquet"

# class-defining vis thresholds (m)
THRESH_DIVERSIONS_M = 800.0
THRESH_DELAYS_M = 1600.0

# SVGP config - 120 inducing points + 400 iters is a sweet spot on CPU
SVGP_M = 120
SVGP_MAXITER = 400
SVGP_SEED = 42

PI_Z = 1.6448536269514722

# P(Diversions) thresholds where an operational tool would raise a fog alert
ALERT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

log = setup_logging()
log.name = "gp_classification"


def fit_svgp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    M: int = SVGP_M,
    maxiter: int = SVGP_MAXITER,
) -> np.ndarray:
    """Sparse variational multi-class GP. Returns (n_test, 3) probs in [Normal, Delays, Diversions] order."""
    rng = np.random.default_rng(SVGP_SEED)
    n = X_train.shape[0]
    Z = X_train[rng.choice(n, min(M, n), replace=False)].copy()

    kernel = gpflow.kernels.Matern52(lengthscales=np.ones(n_features, dtype=np.float64))
    likelihood = gpflow.likelihoods.MultiClass(3)  # RobustMax by default
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=3,
    )
    data = (
        tf.convert_to_tensor(X_train, dtype=tf.float64),
        tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float64),
    )
    gpflow.optimizers.Scipy().minimize(
        model.training_loss_closure(data),
        model.trainable_variables,
        options={"maxiter": maxiter, "disp": False},
    )
    proba, _ = model.predict_y(tf.convert_to_tensor(X_test, dtype=tf.float64))
    return proba.numpy()


def threshold_derived_proba(
    gp_median_vis_m: np.ndarray,
    gp_pi90_hi_m: np.ndarray,
    t_div: float = THRESH_DIVERSIONS_M,
    t_del: float = THRESH_DELAYS_M,
) -> np.ndarray:
    """Recover log-space (mean, std) from Step 5 OOF and integrate Gaussian CDF over thresholds."""
    mean_log = np.log1p(gp_median_vis_m)
    std_log = (np.log1p(gp_pi90_hi_m) - mean_log) / PI_Z
    std_log = np.maximum(std_log, 1e-6)  # avoid zero-width

    z_div = (np.log1p(t_div) - mean_log) / std_log
    z_del = (np.log1p(t_del) - mean_log) / std_log
    p_div = norm.cdf(z_div)
    p_del = norm.cdf(z_del) - norm.cdf(z_div)
    p_nrm = 1.0 - norm.cdf(z_del)

    proba = np.clip(np.stack([p_nrm, p_del, p_div], axis=1), 1e-9, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    oh = np.zeros_like(proba)
    oh[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def classwise_ece(y_true: np.ndarray, proba: np.ndarray, c: int, n_bins: int = 10) -> float:
    """One-vs-rest ECE for class c (10 equal-width bins)."""
    p = proba[:, c]
    y = (y_true == c).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y_true: np.ndarray, proba: np.ndarray, c: int, n_bins: int = 10):
    """Per-bin (mean_pred, empirical_freq, count) for class c."""
    p = proba[:, c]
    y = (y_true == c).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred, emp_freq, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(int(m.sum()))
        mean_pred.append(float(p[m].mean()) if m.sum() else float("nan"))
        emp_freq.append(float(y[m].mean()) if m.sum() else float("nan"))
    return mean_pred, emp_freq, counts


def diversions_alert_curve(y_true: np.ndarray, proba: np.ndarray,
                           thresholds=ALERT_THRESHOLDS) -> list[dict]:
    """Recall/precision sweep over P(Diversions) alert thresholds."""
    pdiv = proba[:, 2]
    is_div = (y_true == 2)
    out = []
    for tau in thresholds:
        flag = pdiv > tau
        n_flag = int(flag.sum())
        recall = float((flag & is_div).sum() / max(is_div.sum(), 1))
        precision = float((flag & is_div).sum() / max(n_flag, 1)) if n_flag else 0.0
        out.append({"threshold": tau, "n_flagged": n_flag,
                    "recall": recall, "precision": precision})
    return out


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Full metric bundle for one probabilistic classifier."""
    y_pred = proba.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": {CLASS_NAMES[c]: float(p[c]) for c in range(3)},
        "per_class_recall": {CLASS_NAMES[c]: float(r[c]) for c in range(3)},
        "per_class_f1": {CLASS_NAMES[c]: float(f1[c]) for c in range(3)},
        "multiclass_brier": multiclass_brier(y_true, proba),
        "ece_per_class": {CLASS_NAMES[c]: classwise_ece(y_true, proba, c) for c in range(3)},
        "ece_sum": float(sum(classwise_ece(y_true, proba, c) for c in range(3))),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "diversions_alert_curve": diversions_alert_curve(y_true, proba),
        "n": int(len(y_true)),
    }


@dataclass
class FoldClf:
    fold_id: int
    test_season: str
    svgp_proba: np.ndarray
    thresh_proba: np.ndarray
    y_true: np.ndarray
    dates: np.ndarray


def train_and_evaluate(
    train_pool: pd.DataFrame, gp_reg_oof: pd.DataFrame,
) -> tuple[list[FoldClf], pd.DataFrame]:
    """Fit SVGP per fold, derive threshold probs from Step 5 OOF joined on date."""
    results: list[FoldClf] = []
    oof_chunks: list[pd.DataFrame] = []
    total_t0 = time.time()

    reg_by_date = gp_reg_oof.set_index("date_npt")

    for prepared in iter_prepared_folds(train_pool):
        spec = prepared.spec
        log.info(f"Fold {spec.fold_id} ({spec.test_season}): "
                 f"train n={prepared.X_train.shape[0]}, test n={prepared.X_test.shape[0]} -- fitting SVGP...")
        t0 = time.time()

        # method 1: direct SVGP
        svgp_proba = fit_svgp_classifier(
            prepared.X_train, prepared.y_clf_train, prepared.X_test,
            n_features=len(FEATURE_COLUMNS),
        )

        # method 2: threshold-derived from Step 5 regression posterior
        test_dates = train_pool.loc[spec.test_idx, "date_npt"].values
        reg_slice = reg_by_date.loc[test_dates]
        thresh_proba = threshold_derived_proba(
            reg_slice["gp_median_vis_m"].to_numpy(),
            reg_slice["gp_pi90_hi_m"].to_numpy(),
        )

        y_true = prepared.y_clf_test
        results.append(FoldClf(
            fold_id=spec.fold_id, test_season=spec.test_season,
            svgp_proba=svgp_proba, thresh_proba=thresh_proba,
            y_true=y_true, dates=test_dates,
        ))

        oof_chunks.append(pd.DataFrame({
            "date_npt": test_dates,
            "fold_id": spec.fold_id,
            "test_season": spec.test_season,
            "true_class": y_true,
            "svgp_proba_normal": svgp_proba[:, 0],
            "svgp_proba_delays": svgp_proba[:, 1],
            "svgp_proba_diversions": svgp_proba[:, 2],
            "thresh_proba_normal": thresh_proba[:, 0],
            "thresh_proba_delays": thresh_proba[:, 1],
            "thresh_proba_diversions": thresh_proba[:, 2],
        }))
        log.info(f"  Fold {spec.fold_id}: {time.time()-t0:.1f}s")

    oof = pd.concat(oof_chunks, ignore_index=True)
    log.info(f"All folds complete: total {time.time()-total_t0:.1f}s")
    return results, oof


def aggregate_metrics(results: list[FoldClf], oof: pd.DataFrame) -> dict:
    """Per-fold + aggregate-OOF metrics for both methods."""
    out: dict = {"per_fold": [], "aggregate": {}}

    for r in results:
        out["per_fold"].append({
            "fold_id": r.fold_id,
            "test_season": r.test_season,
            "svgp": classification_metrics(r.y_true, r.svgp_proba),
            "threshold_derived": classification_metrics(r.y_true, r.thresh_proba),
        })

    yt = oof["true_class"].to_numpy()
    svgp = oof[["svgp_proba_normal", "svgp_proba_delays", "svgp_proba_diversions"]].to_numpy()
    thr = oof[["thresh_proba_normal", "thresh_proba_delays", "thresh_proba_diversions"]].to_numpy()
    out["aggregate"]["svgp"] = classification_metrics(yt, svgp)
    out["aggregate"]["threshold_derived"] = classification_metrics(yt, thr)

    # reliability curves on aggregate OOF
    out["reliability"] = {"svgp": {}, "threshold_derived": {}}
    for method, pr in [("svgp", svgp), ("threshold_derived", thr)]:
        for c in range(3):
            mp, ef, ct = reliability_curve(yt, pr, c)
            out["reliability"][method][CLASS_NAMES[c]] = {
                "mean_pred": mp, "empirical_freq": ef, "counts": ct,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GP classification: direct SVGP + threshold-derived from regression posterior."
    )
    parser.add_argument("--oof-out", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--gp-reg-oof", type=Path, default=GP_REG_OOF_PATH)
    args = parser.parse_args()

    if not args.gp_reg_oof.exists():
        raise FileNotFoundError(
            f"GP regression OOF not found at {args.gp_reg_oof}. Run Step 5 first."
        )

    df = load_modelling_table()
    train_pool, _holdout = split_holdout(df)
    gp_reg_oof = pd.read_parquet(args.gp_reg_oof)
    gp_reg_oof["date_npt"] = pd.to_datetime(gp_reg_oof["date_npt"])

    log.info("Training GP classifiers across all 8 folds + deriving threshold probabilities...")
    results, oof = train_and_evaluate(train_pool, gp_reg_oof)
    metrics = aggregate_metrics(results, oof)

    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.oof_out, index=False)
    log.info(f"Wrote OOF predictions ({len(oof)} rows): {args.oof_out}")
    args.metrics_out.write_text(json.dumps(metrics, indent=2))
    log.info(f"Wrote metrics JSON: {args.metrics_out}")

    agg = metrics["aggregate"]
    print()
    print("=" * 72)
    print("STEP 6 HEADLINE (out-of-fold aggregate, 1206 nights)")
    print("=" * 72)
    for name, key in [("Direct SVGP classifier", "svgp"),
                      ("Threshold-derived GP", "threshold_derived")]:
        m = agg[key]
        adiv = next(a for a in m["diversions_alert_curve"] if a["threshold"] == 0.15)
        print(f"\n{name}:")
        print(f"  Brier={m['multiclass_brier']:.3f}  sumECE={m['ece_sum']:.3f}  "
              f"acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")
        print(f"  Diversions: F1={m['per_class_f1']['Diversions']:.3f}  "
              f"alert@0.15 recall={adiv['recall']:.2f} precision={adiv['precision']:.2f}")
    print("\n(Compare against RF Step 4: Brier=0.280, sumECE=0.289, bal_acc=0.570, Div alert@0.15 recall=0.60)")


if __name__ == "__main__":
    main()
