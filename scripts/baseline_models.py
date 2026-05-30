"""Step 4: trains RF reg+clf plus climatology and persistence baselines on all CV folds."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

# make this importable from notebooks/ too
import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    CLASS_NAMES,
    load_modelling_table,
    split_holdout,
    iter_prepared_folds,
    setup_logging,
)


DEFAULT_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "rf_oof_predictions.parquet"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "step4_metrics.json"

RF_REG_PARAMS = dict(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
RF_CLF_PARAMS = dict(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

PI_Z = 1.6448536269514722  # z for 90% Gaussian PI

log = setup_logging()
log.name = "baseline_models"


def predict_rf_regressor_with_uncertainty(
    rf: RandomForestRegressor,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean + across-tree std as RF's natural uncertainty estimate."""
    tree_preds = np.stack([t.predict(X) for t in rf.estimators_], axis=0)
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def climatology_predict(
    train_dates: pd.Series, train_y: np.ndarray, test_dates: pd.Series,
) -> np.ndarray:
    """Per-day-of-year mean from training data, with overall-mean fallback."""
    train_doy = pd.to_datetime(train_dates).dt.dayofyear
    test_doy = pd.to_datetime(test_dates).dt.dayofyear
    doy_mean = pd.Series(train_y, index=train_doy.values).groupby(level=0).mean()
    overall_mean = float(train_y.mean())
    return np.array([doy_mean.get(d, overall_mean) for d in test_doy.values])


def persistence_predict(
    train_pool: pd.DataFrame, test_idx: np.ndarray,
) -> np.ndarray:
    """Predict tomorrow morning = tonight's sunset visibility, with NaN -> train median."""
    sv = train_pool.loc[test_idx, "sunset_visibility_m"].to_numpy(dtype=float)
    if np.isnan(sv).any():
        train_sv = train_pool["sunset_visibility_m"].to_numpy(dtype=float)
        sv = np.where(np.isnan(sv), np.nanmedian(train_sv), sv)
    return sv


@dataclass
class RegressionMetrics:
    mae_m: float
    rmse_m: float
    r2: float
    pi90_coverage: float | None = None
    pi90_mean_width_m: float | None = None
    n: int = 0


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    multiclass_brier: float
    confusion: list[list[int]]
    n: int = 0


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
) -> RegressionMetrics:
    """Regression metrics in metres; adds 90% PI stats if std given."""
    metrics = RegressionMetrics(
        mae_m=float(mean_absolute_error(y_true, y_pred)),
        rmse_m=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        n=int(len(y_true)),
    )
    if y_std is not None:
        lo = y_pred - PI_Z * y_std
        hi = y_pred + PI_Z * y_std
        inside = (y_true >= lo) & (y_true <= hi)
        metrics.pi90_coverage = float(inside.mean())
        metrics.pi90_mean_width_m = float((hi - lo).mean())
    return metrics


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> float:
    """Multi-class Brier; lower is better."""
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(((y_proba - one_hot) ** 2).sum(axis=1).mean())


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
) -> ClassificationMetrics:
    """3-class metrics + Brier; manual macro averages over all 3 labels."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    balanced_acc = float(np.mean(r))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=balanced_acc,
        macro_f1=macro_f1,
        per_class_precision={CLASS_NAMES[c]: float(p[c]) for c in range(3)},
        per_class_recall={CLASS_NAMES[c]: float(r[c]) for c in range(3)},
        per_class_f1={CLASS_NAMES[c]: float(f1[c]) for c in range(3)},
        multiclass_brier=multiclass_brier(y_true, y_proba),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        n=int(len(y_true)),
    )


@dataclass
class FoldResult:
    fold_id: int
    test_season: str
    oof_frame: pd.DataFrame
    rf_reg_metrics: RegressionMetrics
    persist_metrics: RegressionMetrics
    clim_metrics: RegressionMetrics
    rf_clf_metrics: ClassificationMetrics
    gini_importance: np.ndarray
    perm_importance_mean: np.ndarray
    perm_importance_std: np.ndarray


def train_and_evaluate(
    train_pool: pd.DataFrame,
    n_perm_repeats: int = 10,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Train RF + baselines per fold; return per-fold results and concatenated OOF."""
    results: list[FoldResult] = []

    for prepared in iter_prepared_folds(train_pool):
        spec = prepared.spec
        log.info(
            f"Fold {spec.fold_id} ({spec.test_season}): "
            f"train n={prepared.X_train.shape[0]}, test n={prepared.X_test.shape[0]}"
        )

        # RF regressor on raw metres (RF is scale-invariant)
        rf_reg = RandomForestRegressor(**RF_REG_PARAMS)
        rf_reg.fit(prepared.X_train, prepared.y_reg_train)
        rf_pred, rf_std = predict_rf_regressor_with_uncertainty(rf_reg, prepared.X_test)
        rf_reg_metrics = compute_regression_metrics(
            prepared.y_reg_test, rf_pred, rf_std,
        )

        # RF classifier
        rf_clf = RandomForestClassifier(**RF_CLF_PARAMS)
        rf_clf.fit(prepared.X_train, prepared.y_clf_train)
        rf_clf_pred = rf_clf.predict(prepared.X_test)
        # reindex predict_proba columns to fixed [0, 1, 2] order
        proba_raw = rf_clf.predict_proba(prepared.X_test)
        rf_clf_proba = np.zeros((proba_raw.shape[0], 3))
        for i, c in enumerate(rf_clf.classes_):
            rf_clf_proba[:, int(c)] = proba_raw[:, i]
        rf_clf_metrics = compute_classification_metrics(
            prepared.y_clf_test, rf_clf_pred, rf_clf_proba,
        )

        # Climatology
        train_dates = train_pool.loc[spec.train_idx, "date_npt"]
        test_dates = train_pool.loc[spec.test_idx, "date_npt"]
        clim_pred = climatology_predict(
            train_dates, prepared.y_reg_train, test_dates,
        )
        clim_metrics = compute_regression_metrics(prepared.y_reg_test, clim_pred)

        # Persistence
        persist_pred = persistence_predict(train_pool, spec.test_idx)
        persist_metrics = compute_regression_metrics(
            prepared.y_reg_test, persist_pred,
        )

        # Feature importance: Gini + permutation
        gini = rf_reg.feature_importances_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(
                rf_reg, prepared.X_test, prepared.y_reg_test,
                n_repeats=n_perm_repeats, random_state=42, n_jobs=-1,
                scoring="r2",
            )

        oof_slice = pd.DataFrame({
            "date_npt": test_dates.values,
            "fold_id": spec.fold_id,
            "test_season": spec.test_season,
            "true_vis_m": prepared.y_reg_test,
            "true_class": prepared.y_clf_test,
            "rf_pred_vis_m": rf_pred,
            "rf_pred_std_m": rf_std,
            "rf_pi90_lo_m": rf_pred - PI_Z * rf_std,
            "rf_pi90_hi_m": rf_pred + PI_Z * rf_std,
            "rf_pred_class": rf_clf_pred,
            "rf_proba_normal": rf_clf_proba[:, 0],
            "rf_proba_delays": rf_clf_proba[:, 1],
            "rf_proba_diversions": rf_clf_proba[:, 2],
            "climatology_pred_vis_m": clim_pred,
            "persistence_pred_vis_m": persist_pred,
        })

        results.append(FoldResult(
            fold_id=spec.fold_id,
            test_season=spec.test_season,
            oof_frame=oof_slice,
            rf_reg_metrics=rf_reg_metrics,
            persist_metrics=persist_metrics,
            clim_metrics=clim_metrics,
            rf_clf_metrics=rf_clf_metrics,
            gini_importance=gini,
            perm_importance_mean=perm.importances_mean,
            perm_importance_std=perm.importances_std,
        ))

    oof_full = pd.concat([r.oof_frame for r in results], ignore_index=True)
    return results, oof_full


def aggregate_metrics(
    results: list[FoldResult],
    oof: pd.DataFrame,
) -> dict:
    """Build per-fold + aggregate-OOF + feature-importance metrics dict for JSON."""
    out = {"per_fold": [], "aggregate": {}, "feature_importance": {}}

    for r in results:
        out["per_fold"].append({
            "fold_id": r.fold_id,
            "test_season": r.test_season,
            "rf_regression": r.rf_reg_metrics.__dict__,
            "climatology_regression": r.clim_metrics.__dict__,
            "persistence_regression": r.persist_metrics.__dict__,
            "rf_classification": r.rf_clf_metrics.__dict__,
        })

    out["aggregate"]["rf_regression"] = compute_regression_metrics(
        oof["true_vis_m"].to_numpy(),
        oof["rf_pred_vis_m"].to_numpy(),
        oof["rf_pred_std_m"].to_numpy(),
    ).__dict__
    out["aggregate"]["climatology_regression"] = compute_regression_metrics(
        oof["true_vis_m"].to_numpy(),
        oof["climatology_pred_vis_m"].to_numpy(),
    ).__dict__
    out["aggregate"]["persistence_regression"] = compute_regression_metrics(
        oof["true_vis_m"].to_numpy(),
        oof["persistence_pred_vis_m"].to_numpy(),
    ).__dict__
    out["aggregate"]["rf_classification"] = compute_classification_metrics(
        oof["true_class"].to_numpy(),
        oof["rf_pred_class"].to_numpy(),
        oof[["rf_proba_normal", "rf_proba_delays", "rf_proba_diversions"]].to_numpy(),
    ).__dict__

    gini = np.stack([r.gini_importance for r in results])
    perm = np.stack([r.perm_importance_mean for r in results])
    out["feature_importance"]["features"] = list(FEATURE_COLUMNS)
    out["feature_importance"]["gini_mean"] = gini.mean(axis=0).tolist()
    out["feature_importance"]["gini_std"] = gini.std(axis=0).tolist()
    out["feature_importance"]["perm_r2_mean"] = perm.mean(axis=0).tolist()
    out["feature_importance"]["perm_r2_std"] = perm.std(axis=0).tolist()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RF regressor + RF classifier + climatology + persistence baselines on every CV fold."
    )
    parser.add_argument("--oof-out", type=Path, default=DEFAULT_OOF_PATH,
                        help="Where to write the OOF predictions parquet.")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH,
                        help="Where to write the metrics JSON.")
    parser.add_argument("--n-perm-repeats", type=int, default=10,
                        help="Permutation importance repeats per fold (default 10).")
    args = parser.parse_args()

    df = load_modelling_table()
    train_pool, _holdout = split_holdout(df)

    log.info("Training models across all 8 forward-chaining CV folds...")
    results, oof = train_and_evaluate(train_pool, n_perm_repeats=args.n_perm_repeats)

    metrics = aggregate_metrics(results, oof)

    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.oof_out, index=False)
    log.info(f"Wrote OOF predictions ({len(oof)} rows): {args.oof_out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))
    log.info(f"Wrote metrics JSON: {args.metrics_out}")

    print()
    print("=" * 72)
    print("HEADLINE METRICS (out-of-fold, concatenated across 8 folds)")
    print("=" * 72)
    agg = metrics["aggregate"]
    print(f"\nRegression (target: morning min visibility, metres)")
    print(f"{'Model':<14} {'MAE':>10} {'RMSE':>10} {'R^2':>10} {'PI90 cov':>10} {'PI90 width':>12}")
    for name, key in [("RF", "rf_regression"), ("Climatology", "climatology_regression"),
                      ("Persistence", "persistence_regression")]:
        m = agg[key]
        cov = f"{m['pi90_coverage']:.3f}" if m.get("pi90_coverage") is not None else "  --  "
        wid = f"{m['pi90_mean_width_m']:.0f}" if m.get("pi90_mean_width_m") is not None else "  --  "
        print(f"{name:<14} {m['mae_m']:>10.0f} {m['rmse_m']:>10.0f} {m['r2']:>10.3f} {cov:>10} {wid:>12}")
    print(f"\nClassification (3-way: Normal / Delays / Diversions)")
    m = agg["rf_classification"]
    print(f"  Accuracy           : {m['accuracy']:.3f}")
    print(f"  Balanced accuracy  : {m['balanced_accuracy']:.3f}")
    print(f"  Macro F1           : {m['macro_f1']:.3f}")
    print(f"  Multi-class Brier  : {m['multiclass_brier']:.3f}")
    print(f"  Per-class F1       : Normal={m['per_class_f1']['Normal']:.3f}  "
          f"Delays={m['per_class_f1']['Delays']:.3f}  "
          f"Diversions={m['per_class_f1']['Diversions']:.3f}")


if __name__ == "__main__":
    main()
