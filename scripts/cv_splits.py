"""Step 3: forward-chaining time-series CV folds + per-fold impute/scale pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "processed" / "vnkt_modelling_table.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "cv_manifest.json"

HOLDOUT_SEASON: str = "2025-26"
INITIAL_TRAINING_SEASONS: list[str] = ["2015-16", "2016-17"]

# 19 features (night_obs_count deliberately excluded - it tracks IEM ingest schedule, not weather)
FEATURE_COLUMNS: list[str] = [
    "sunset_tempc",
    "sunset_dewpoint_depr_c",
    "sunset_pressure_hpa",
    "sunset_wind_speed_ms",
    "sunset_visibility_m",
    "predawn_tempc",
    "predawn_dewpoint_depr_c",
    "predawn_pressure_hpa",
    "overnight_temp_drop_c",
    "overnight_dewpoint_depr_drop_c",
    "overnight_pressure_change_hpa",
    "night_mean_wind_speed_ms",
    "night_calm_fraction",
    "night_mean_sky_cover",
    "night_clear_fraction",
    "night_mist_observed",
    "night_fog_observed",
    "doy_sin",
    "doy_cos",
]

TARGET_REGRESSION: str = "target_min_vis_m"
TARGET_CLASSIFICATION: str = "target_class"

CLASS_NAMES: dict[int, str] = {0: "Normal", 1: "Delays", 2: "Diversions"}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Module logger; idempotent so notebook re-imports don't duplicate output."""
    logger = logging.getLogger("cv_splits")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


log = setup_logging()


def assign_winter_season(date: pd.Timestamp) -> str:
    """Map a date to its winter-season label. Oct-Dec Y -> 'Y-(Y+1)'; Jan-Feb Y -> '(Y-1)-Y'."""
    year = date.year
    month = date.month
    if month >= 10:
        start_year = year
    elif month <= 2:
        start_year = year - 1
    else:
        raise ValueError(
            f"Date {date.date()} lies outside the Oct-Feb winter window; "
            "the modelling table appears not to have been season-filtered."
        )
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def load_modelling_table(parquet_path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Load Step 2 parquet + attach season; fail loudly if expected columns are missing."""
    log.info(f"Loading modelling table: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df["date_npt"] = pd.to_datetime(df["date_npt"])
    df = df.sort_values("date_npt").reset_index(drop=True)

    required = set(FEATURE_COLUMNS) | {TARGET_REGRESSION, TARGET_CLASSIFICATION, "date_npt"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Modelling table missing required columns: {sorted(missing)}"
        )

    df["season"] = df["date_npt"].apply(assign_winter_season)
    log.info(f"  rows={len(df)}  seasons={df['season'].nunique()}  "
             f"date_range={df['date_npt'].min().date()} -> {df['date_npt'].max().date()}")
    return df


def split_holdout(
    df: pd.DataFrame, holdout_season: str = HOLDOUT_SEASON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train_pool, holdout). Holdout is reserved until the final results section."""
    if holdout_season not in df["season"].unique():
        raise ValueError(
            f"Holdout season '{holdout_season}' not present in data; "
            f"available seasons: {sorted(df['season'].unique())}"
        )
    holdout = df[df["season"] == holdout_season].copy().reset_index(drop=True)
    train_pool = df[df["season"] != holdout_season].copy().reset_index(drop=True)
    log.info(f"Holdout season {holdout_season}: {len(holdout)} rows  |  "
             f"train pool: {len(train_pool)} rows across {train_pool['season'].nunique()} seasons")
    return train_pool, holdout


@dataclass
class FoldSpec:
    """One fold's metadata. train_idx/test_idx index into train_pool (must be reset_index)."""
    fold_id: int
    train_seasons: list[str]
    test_season: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def forward_chaining_folds(train_pool: pd.DataFrame) -> list[FoldSpec]:
    """Generate growing-window forward-chaining folds (8 total)."""
    seasons_sorted = sorted(train_pool["season"].unique())

    # sanity: the seasons we bundle as initial must actually be the earliest
    for s in INITIAL_TRAINING_SEASONS:
        if s not in seasons_sorted:
            raise ValueError(
                f"Initial training season '{s}' not found in train pool. "
                f"Available seasons: {seasons_sorted}"
            )
    if seasons_sorted[: len(INITIAL_TRAINING_SEASONS)] != INITIAL_TRAINING_SEASONS:
        raise ValueError(
            f"INITIAL_TRAINING_SEASONS {INITIAL_TRAINING_SEASONS} are not "
            f"the earliest in the data {seasons_sorted[:len(INITIAL_TRAINING_SEASONS)]}. "
            "This would break temporal ordering."
        )

    test_seasons = seasons_sorted[len(INITIAL_TRAINING_SEASONS):]
    folds: list[FoldSpec] = []
    cumulative_train = list(INITIAL_TRAINING_SEASONS)

    for fold_id, test_season in enumerate(test_seasons, start=1):
        train_mask = train_pool["season"].isin(cumulative_train)
        test_mask = train_pool["season"] == test_season
        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_id} has empty train ({len(train_idx)}) "
                f"or test ({len(test_idx)}) set."
            )
        folds.append(FoldSpec(
            fold_id=fold_id,
            train_seasons=list(cumulative_train),
            test_season=test_season,
            train_idx=train_idx,
            test_idx=test_idx,
            train_start=train_pool.loc[train_idx, "date_npt"].min(),
            train_end=train_pool.loc[train_idx, "date_npt"].max(),
            test_start=train_pool.loc[test_idx, "date_npt"].min(),
            test_end=train_pool.loc[test_idx, "date_npt"].max(),
        ))
        cumulative_train.append(test_season)

    log.info(f"Built {len(folds)} forward-chaining folds")
    return folds


def build_feature_pipeline() -> Pipeline:
    """Unfitted: median impute -> standardise. Fitted on train only per fold."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def build_target_scaler() -> StandardScaler:
    """Unfitted target scaler; GPs converge much better with a standardised target."""
    return StandardScaler()


@dataclass
class PreparedFold:
    """Everything Step 4-6 need out of one fold."""
    spec: FoldSpec
    X_train: np.ndarray
    X_test: np.ndarray
    y_reg_train: np.ndarray
    y_reg_test: np.ndarray
    y_reg_train_scaled: np.ndarray
    y_reg_test_scaled: np.ndarray
    y_clf_train: np.ndarray
    y_clf_test: np.ndarray
    feature_pipeline: Pipeline
    target_scaler: StandardScaler
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


def iter_prepared_folds(
    train_pool: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> Iterator[PreparedFold]:
    """Yield PreparedFold for each CV fold (fit pipeline + target scaler on train only)."""
    folds = forward_chaining_folds(train_pool)
    for spec in folds:
        X_train_raw = train_pool.loc[spec.train_idx, feature_columns].to_numpy(dtype=float)
        X_test_raw = train_pool.loc[spec.test_idx, feature_columns].to_numpy(dtype=float)
        y_reg_train = train_pool.loc[spec.train_idx, TARGET_REGRESSION].to_numpy(dtype=float)
        y_reg_test = train_pool.loc[spec.test_idx, TARGET_REGRESSION].to_numpy(dtype=float)
        y_clf_train = train_pool.loc[spec.train_idx, TARGET_CLASSIFICATION].to_numpy(dtype=int)
        y_clf_test = train_pool.loc[spec.test_idx, TARGET_CLASSIFICATION].to_numpy(dtype=int)

        pipe = build_feature_pipeline().fit(X_train_raw)
        X_train = pipe.transform(X_train_raw)
        X_test = pipe.transform(X_test_raw)

        target_scaler = build_target_scaler().fit(y_reg_train.reshape(-1, 1))
        y_reg_train_scaled = target_scaler.transform(y_reg_train.reshape(-1, 1)).ravel()
        y_reg_test_scaled = target_scaler.transform(y_reg_test.reshape(-1, 1)).ravel()

        yield PreparedFold(
            spec=spec,
            X_train=X_train,
            X_test=X_test,
            y_reg_train=y_reg_train,
            y_reg_test=y_reg_test,
            y_reg_train_scaled=y_reg_train_scaled,
            y_reg_test_scaled=y_reg_test_scaled,
            y_clf_train=y_clf_train,
            y_clf_test=y_clf_test,
            feature_pipeline=pipe,
            target_scaler=target_scaler,
            feature_names=list(feature_columns),
        )


def find_constant_features(
    X_train: np.ndarray,
    feature_names: list[str] = FEATURE_COLUMNS,
    tol: float = 1e-6,
) -> list[str]:
    """Names of features whose training-fold std is below tol (no variance in this fold)."""
    std = X_train.std(axis=0)
    return [feature_names[i] for i in np.where(std < tol)[0]]


def summarise_folds(
    train_pool: pd.DataFrame,
    folds: list[FoldSpec],
) -> pd.DataFrame:
    """Tidy per-fold summary table (sizes, dates, class counts)."""
    rows = []
    for fold in folds:
        train_classes = train_pool.loc[fold.train_idx, TARGET_CLASSIFICATION].value_counts()
        test_classes = train_pool.loc[fold.test_idx, TARGET_CLASSIFICATION].value_counts()
        rows.append({
            "fold": fold.fold_id,
            "train_seasons": " + ".join(fold.train_seasons) if len(fold.train_seasons) <= 2
                              else f"{fold.train_seasons[0]} .. {fold.train_seasons[-1]}",
            "test_season": fold.test_season,
            "train_n": len(fold.train_idx),
            "test_n": len(fold.test_idx),
            "train_Normal": int(train_classes.get(0, 0)),
            "train_Delays": int(train_classes.get(1, 0)),
            "train_Div": int(train_classes.get(2, 0)),
            "test_Normal": int(test_classes.get(0, 0)),
            "test_Delays": int(test_classes.get(1, 0)),
            "test_Div": int(test_classes.get(2, 0)),
            "train_end": fold.train_end.date().isoformat(),
            "test_start": fold.test_start.date().isoformat(),
        })
    return pd.DataFrame(rows)


def save_manifest(
    folds: list[FoldSpec],
    train_pool: pd.DataFrame,
    holdout: pd.DataFrame,
    out_path: Path = DEFAULT_MANIFEST,
) -> None:
    """JSON record of the splits for the paper appendix (Step 4-6 don't depend on this)."""
    holdout_classes = holdout[TARGET_CLASSIFICATION].value_counts()
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "holdout_season": HOLDOUT_SEASON,
        "initial_training_seasons": INITIAL_TRAINING_SEASONS,
        "feature_columns": FEATURE_COLUMNS,
        "target_regression": TARGET_REGRESSION,
        "target_classification": TARGET_CLASSIFICATION,
        "class_names": CLASS_NAMES,
        "n_folds": len(folds),
        "train_pool_size": int(len(train_pool)),
        "holdout": {
            "season": HOLDOUT_SEASON,
            "size": int(len(holdout)),
            "start": holdout["date_npt"].min().date().isoformat(),
            "end": holdout["date_npt"].max().date().isoformat(),
            "class_counts": {
                CLASS_NAMES[c]: int(holdout_classes.get(c, 0)) for c in [0, 1, 2]
            },
        },
        "folds": [],
    }
    for f in folds:
        train_classes = train_pool.loc[f.train_idx, TARGET_CLASSIFICATION].value_counts()
        test_classes = train_pool.loc[f.test_idx, TARGET_CLASSIFICATION].value_counts()
        manifest["folds"].append({
            "fold_id": f.fold_id,
            "train_seasons": f.train_seasons,
            "test_season": f.test_season,
            "train_size": int(len(f.train_idx)),
            "test_size": int(len(f.test_idx)),
            "train_start": f.train_start.date().isoformat(),
            "train_end": f.train_end.date().isoformat(),
            "test_start": f.test_start.date().isoformat(),
            "test_end": f.test_end.date().isoformat(),
            "train_class_counts": {
                CLASS_NAMES[c]: int(train_classes.get(c, 0)) for c in [0, 1, 2]
            },
            "test_class_counts": {
                CLASS_NAMES[c]: int(test_classes.get(c, 0)) for c in [0, 1, 2]
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote CV manifest: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct and inspect time-series CV splits for the VNKT fog forecasting task."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                        help="Path to Step 2 modelling table (default: data/processed/vnkt_modelling_table.parquet)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Output path for JSON manifest (default: data/processed/cv_manifest.json)")
    args = parser.parse_args()

    df = load_modelling_table(args.parquet)
    train_pool, holdout = split_holdout(df)
    folds = forward_chaining_folds(train_pool)

    summary = summarise_folds(train_pool, folds)
    log.info("Per-fold summary:")
    print()
    print(summary.to_string(index=False))
    print()

    log.info(f"Holdout {HOLDOUT_SEASON}: {len(holdout)} rows  "
             f"(Normal={int((holdout[TARGET_CLASSIFICATION]==0).sum())}, "
             f"Delays={int((holdout[TARGET_CLASSIFICATION]==1).sum())}, "
             f"Diversions={int((holdout[TARGET_CLASSIFICATION]==2).sum())})")

    save_manifest(folds, train_pool, holdout, args.manifest)

    # sanity check: prepared folds run end-to-end with no NaN after impute
    log.info("Running end-to-end iter_prepared_folds() sanity check...")
    for prepared in iter_prepared_folds(train_pool):
        fid = prepared.spec.fold_id
        assert not np.isnan(prepared.X_train).any(), \
            f"Fold {fid}: NaN in X_train after imputation"
        assert not np.isnan(prepared.X_test).any(), \
            f"Fold {fid}: NaN in X_test after imputation"
        assert prepared.X_train.shape[1] == len(FEATURE_COLUMNS), \
            f"Fold {fid}: wrong feature count"
        assert abs(prepared.X_train.mean(axis=0)).max() < 1e-6, \
            f"Fold {fid}: X_train not zero-mean after scaling"

        # each feature column is either unit-var (normal) or zero-var (constant, sklearn safe-divides)
        train_std = prepared.X_train.std(axis=0)
        is_zero_var = train_std < 1e-6
        is_unit_var = np.abs(train_std - 1.0) < 1e-6
        bad = ~(is_zero_var | is_unit_var)
        assert not bad.any(), (
            f"Fold {fid}: features with unexpected std (neither 0 nor 1): "
            f"{[(FEATURE_COLUMNS[i], float(train_std[i])) for i in np.where(bad)[0]]}"
        )
        if is_zero_var.any():
            const_names = [FEATURE_COLUMNS[i] for i in np.where(is_zero_var)[0]]
            log.warning(
                f"Fold {fid}: zero-variance training features {const_names} "
                f"-- harmless (sklearn safe-divides; RF/GP-ARD will deweight them) "
                f"but these features carry no signal in this fold's training window."
            )
    log.info("  all folds passed sanity checks")


if __name__ == "__main__":
    main()
