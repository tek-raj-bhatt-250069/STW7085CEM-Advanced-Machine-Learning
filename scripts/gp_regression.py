"""Step 5: Matern-5/2 + ARD GP regressor per fold; writes OOF for calibration comparison vs RF."""

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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

# hush TF's GPU-not-found notices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import gpflow  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    load_modelling_table,
    split_holdout,
    iter_prepared_folds,
    setup_logging,
)


DEFAULT_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "gp_oof_predictions.parquet"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "step5_metrics.json"

PI_Z = 1.6448536269514722

# L-BFGS-B config; maxiter=500 converges on the smallest fold with headroom
SCIPY_MAXITER = 500
SCIPY_OPTIONS = {"maxiter": SCIPY_MAXITER, "disp": False}

# hyperparam init (in standardised X / log-target space)
INIT_LENGTHSCALE = 1.0
INIT_SIGNAL_VARIANCE = 1.0
INIT_NOISE_VARIANCE = 0.1

log = setup_logging()
log.name = "gp_regression"


@dataclass
class LogTargetTransform:
    """y_metres -> log1p(y) -> StandardScaler. Use inverse_quantile for PI endpoints."""
    scaler: StandardScaler

    @classmethod
    def fit(cls, y_metres: np.ndarray) -> "LogTargetTransform":
        scaler = StandardScaler().fit(np.log1p(y_metres).reshape(-1, 1))
        return cls(scaler=scaler)

    def transform(self, y_metres: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.log1p(y_metres).reshape(-1, 1)).ravel()

    def inverse_quantile(self, z: np.ndarray) -> np.ndarray:
        """Inverse-transform any z-space quantile back to metres (clipped at 0)."""
        log_q = self.scaler.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
        return np.maximum(np.expm1(log_q), 0.0)


@dataclass
class FoldGP:
    fold_id: int
    test_season: str
    median_m: np.ndarray
    pi90_lo_m: np.ndarray
    pi90_hi_m: np.ndarray
    pred_std_logspace: np.ndarray
    lengthscales: np.ndarray
    signal_variance: float
    noise_variance: float
    elbo: float  # actually log marginal likelihood
    n_train: int
    calib_nominal: np.ndarray
    calib_empirical: np.ndarray


def fit_gp_one_fold(
    X_train: np.ndarray,
    y_train_z: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, gpflow.models.GPR]:
    """Fit Matern52-ARD GPR; returns (mean, var_INCLUDING_NOISE, model)."""
    kernel = gpflow.kernels.Matern52(
        variance=INIT_SIGNAL_VARIANCE,
        lengthscales=np.full(n_features, INIT_LENGTHSCALE, dtype=np.float64),
    )
    model = gpflow.models.GPR(
        data=(
            tf.convert_to_tensor(X_train, dtype=tf.float64),
            tf.convert_to_tensor(y_train_z.reshape(-1, 1), dtype=tf.float64),
        ),
        kernel=kernel,
        mean_function=None,
        noise_variance=INIT_NOISE_VARIANCE,
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        model.trainable_variables,
        options=SCIPY_OPTIONS,
    )
    # predict_y includes Gaussian noise (right thing for an observation-level PI)
    mean_z, pred_var_z = model.predict_y(
        tf.convert_to_tensor(X_test, dtype=tf.float64)
    )
    return mean_z.numpy().ravel(), pred_var_z.numpy().ravel(), model


def train_and_evaluate(train_pool: pd.DataFrame) -> tuple[list[FoldGP], pd.DataFrame]:
    """Run GP on every CV fold."""
    results: list[FoldGP] = []
    oof_chunks: list[pd.DataFrame] = []
    total_t0 = time.time()

    for prepared in iter_prepared_folds(train_pool):
        spec = prepared.spec
        log.info(
            f"Fold {spec.fold_id} ({spec.test_season}): "
            f"train n={prepared.X_train.shape[0]}, test n={prepared.X_test.shape[0]} "
            f"-- fitting GP..."
        )
        fold_t0 = time.time()

        # rebuild target transform from raw metres (Step 3's scaler is linear-on-metres)
        transform = LogTargetTransform.fit(prepared.y_reg_train)
        y_train_z = transform.transform(prepared.y_reg_train)

        mean_z, pred_var_z, model = fit_gp_one_fold(
            prepared.X_train,
            y_train_z,
            prepared.X_test,
            n_features=len(FEATURE_COLUMNS),
        )
        pred_std_z = np.sqrt(pred_var_z)

        # 90% PI endpoints in z-space -> metres
        median_m = transform.inverse_quantile(mean_z)
        pi90_lo_m = transform.inverse_quantile(mean_z - PI_Z * pred_std_z)
        pi90_hi_m = transform.inverse_quantile(mean_z + PI_Z * pred_std_z)

        ls = model.kernel.lengthscales.numpy().astype(float)
        sv = float(model.kernel.variance.numpy())
        nv = float(model.likelihood.variance.numpy())
        lml = float(-model.training_loss().numpy())

        nominal, empirical = fold_coverage_curve(
            prepared.y_reg_test, mean_z, pred_std_z, transform,
        )
        fold_elapsed = time.time() - fold_t0

        results.append(FoldGP(
            fold_id=spec.fold_id,
            test_season=spec.test_season,
            median_m=median_m,
            pi90_lo_m=pi90_lo_m,
            pi90_hi_m=pi90_hi_m,
            pred_std_logspace=pred_std_z,
            lengthscales=ls,
            signal_variance=sv,
            noise_variance=nv,
            elbo=lml,
            n_train=int(prepared.X_train.shape[0]),
            calib_nominal=nominal,
            calib_empirical=empirical,
        ))

        test_dates = train_pool.loc[spec.test_idx, "date_npt"].values
        oof_chunks.append(pd.DataFrame({
            "date_npt": test_dates,
            "fold_id": spec.fold_id,
            "test_season": spec.test_season,
            "true_vis_m": prepared.y_reg_test,
            "true_class": prepared.y_clf_test,
            "gp_median_vis_m": median_m,
            "gp_pi90_lo_m": pi90_lo_m,
            "gp_pi90_hi_m": pi90_hi_m,
            "gp_pred_std_logspace": pred_std_z,
        }))

        log.info(
            f"  Fold {spec.fold_id}: {fold_elapsed:.1f}s, lml={lml:.1f}, "
            f"sigma_n^2={nv:.3f}, sigma_f^2={sv:.3f}, "
            f"max lengthscale={ls.max():.2f}, min lengthscale={ls.min():.2f}"
        )

    oof = pd.concat(oof_chunks, ignore_index=True)
    log.info(f"All folds complete: total GP training {time.time() - total_t0:.1f}s")
    return results, oof


@dataclass
class RegressionMetrics:
    mae_m: float
    rmse_m: float
    r2: float
    pi90_coverage: float
    pi90_mean_width_m: float
    pi90_median_width_m: float
    n: int


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, pi_lo: np.ndarray, pi_hi: np.ndarray,
) -> RegressionMetrics:
    inside = (y_true >= pi_lo) & (y_true <= pi_hi)
    widths = pi_hi - pi_lo
    return RegressionMetrics(
        mae_m=float(mean_absolute_error(y_true, y_pred)),
        rmse_m=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        pi90_coverage=float(inside.mean()),
        pi90_mean_width_m=float(widths.mean()),
        pi90_median_width_m=float(np.median(widths)),
        n=int(len(y_true)),
    )


CALIB_LEVELS = np.linspace(0.1, 0.9, 9)  # nominal levels for calibration curves


def fold_coverage_curve(
    y_true_m: np.ndarray,
    mean_z: np.ndarray,
    std_z: np.ndarray,
    transform: LogTargetTransform,
    levels: np.ndarray = CALIB_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal vs empirical coverage at multiple levels for one fold."""
    from scipy.stats import norm
    nominal = np.asarray(levels, dtype=float)
    empirical = np.empty_like(nominal)
    for i, alpha in enumerate(nominal):
        z_a = norm.ppf(0.5 + alpha / 2)
        lo = transform.inverse_quantile(mean_z - z_a * std_z)
        hi = transform.inverse_quantile(mean_z + z_a * std_z)
        empirical[i] = ((y_true_m >= lo) & (y_true_m <= hi)).mean()
    return nominal, empirical


def aggregate_metrics(
    results: list[FoldGP],
    oof: pd.DataFrame,
    train_pool: pd.DataFrame,
) -> dict:
    """Per-fold + aggregate-OOF + ARD lengthscale summary for JSON."""
    out: dict = {"per_fold": [], "aggregate": {}, "ard": {}}

    for r in results:
        mask = oof["fold_id"] == r.fold_id
        y_true = oof.loc[mask, "true_vis_m"].to_numpy()
        m = compute_regression_metrics(
            y_true, r.median_m, r.pi90_lo_m, r.pi90_hi_m,
        )
        out["per_fold"].append({
            "fold_id": r.fold_id,
            "test_season": r.test_season,
            "n_train": r.n_train,
            "log_marginal_likelihood": r.elbo,
            "signal_variance": r.signal_variance,
            "noise_variance": r.noise_variance,
            "regression": m.__dict__,
        })

    agg = compute_regression_metrics(
        oof["true_vis_m"].to_numpy(),
        oof["gp_median_vis_m"].to_numpy(),
        oof["gp_pi90_lo_m"].to_numpy(),
        oof["gp_pi90_hi_m"].to_numpy(),
    )
    out["aggregate"]["gp_regression"] = agg.__dict__

    # ARD lengthscales per fold + summary
    ls_stack = np.stack([r.lengthscales for r in results])
    out["ard"]["features"] = list(FEATURE_COLUMNS)
    out["ard"]["lengthscales_per_fold"] = ls_stack.tolist()
    out["ard"]["lengthscales_mean"] = ls_stack.mean(axis=0).tolist()
    out["ard"]["lengthscales_std"] = ls_stack.std(axis=0).tolist()
    # relevance = 1/lengthscale (small ls -> high relevance)
    relevance = 1.0 / ls_stack
    out["ard"]["relevance_mean"] = relevance.mean(axis=0).tolist()
    out["ard"]["relevance_std"] = relevance.std(axis=0).tolist()

    # calibration: per-fold + n_test-weighted aggregate
    n_per_fold = np.array([
        int((oof["fold_id"] == r.fold_id).sum()) for r in results
    ])
    calib_stack = np.stack([r.calib_empirical for r in results])
    weighted_emp = (calib_stack * n_per_fold[:, None]).sum(axis=0) / n_per_fold.sum()
    out["calibration"] = {
        "nominal_levels": results[0].calib_nominal.tolist(),
        "per_fold_empirical": calib_stack.tolist(),
        "weighted_empirical": weighted_emp.tolist(),
        "per_fold_n_test": n_per_fold.tolist(),
    }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Matern-5/2 + ARD GP regressor on every CV fold."
    )
    parser.add_argument("--oof-out", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    df = load_modelling_table()
    train_pool, _holdout = split_holdout(df)

    log.info("Training GP regressors across all 8 forward-chaining CV folds...")
    results, oof = train_and_evaluate(train_pool)
    metrics = aggregate_metrics(results, oof, train_pool)

    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.oof_out, index=False)
    log.info(f"Wrote OOF predictions ({len(oof)} rows): {args.oof_out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))
    log.info(f"Wrote metrics JSON: {args.metrics_out}")

    agg = metrics["aggregate"]["gp_regression"]
    print()
    print("=" * 72)
    print("GP HEADLINE METRICS (out-of-fold, concatenated across 8 folds)")
    print("=" * 72)
    print(f"  MAE             : {agg['mae_m']:.0f} m")
    print(f"  RMSE            : {agg['rmse_m']:.0f} m")
    print(f"  R^2             : {agg['r2']:.3f}")
    print(f"  90% PI coverage : {agg['pi90_coverage']:.3f}   (target: 0.900)")
    print(f"  90% PI width    : mean={agg['pi90_mean_width_m']:.0f} m, "
          f"median={agg['pi90_median_width_m']:.0f} m")
    print()
    print("Per-fold log marginal likelihoods (higher = better fit):")
    for r in results:
        ls = r.lengthscales
        print(f"  Fold {r.fold_id} ({r.test_season}): lml={r.elbo:>8.1f}, "
              f"sigma_n^2={r.noise_variance:.3f}, "
              f"ARD lengthscales min/median/max = "
              f"{ls.min():.2f} / {np.median(ls):.2f} / {ls.max():.2f}")


if __name__ == "__main__":
    main()
