#!/usr/bin/env python3
"""Step 2: turn the cleaned hourly METAR into a daily modelling table (one row per target morning)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# target window = morning arrival rush at TIA (05:45-09:45 NPT)
TARGET_WINDOW_START_HOUR = 5
TARGET_WINDOW_END_HOUR = 9

# feature window = prior night only (18:00 D-1 -> 05:00 D), no peeking at the morning
FEATURE_WINDOW_START_HOUR = 18
FEATURE_WINDOW_END_HOUR = 5

# starting condition of the night
SUNSET_HOUR_RANGE = (17, 18)

# the setup right before the target window
PREDAWN_HOUR_RANGE = (2, 4)

# vis thresholds in metres (from TIA landing/departure minima reporting)
CLASS_THRESHOLDS = {
    "diversions_likely": 800.0,
    "delays_likely":     1600.0,
}

# Oct-Feb fog season (Nov-Jan core + Oct/Feb buffer for more rows)
FOG_SEASON_MONTHS = [10, 11, 12, 1, 2]


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Console logger."""
    logger = logging.getLogger("daily_features")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def compute_targets(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Per-date target: min visibility in the morning window + 3-class label + fog flag."""
    logger.info("Computing target variable per date...")

    morning = df[
        df["valid_npt"].dt.hour.between(
            TARGET_WINDOW_START_HOUR, TARGET_WINDOW_END_HOUR
        )
    ].copy()
    morning["date_npt"] = morning["valid_npt"].dt.date

    targets = (
        morning.groupby("date_npt")
        .agg(
            target_min_vis_m=("visibility_m", "min"),
            target_morning_obs=("visibility_m", "count"),
            target_fog_observed=("wx_fog", "max"),
        )
        .reset_index()
    )
    targets["date_npt"] = pd.to_datetime(targets["date_npt"])

    def classify(vis_m: float) -> int:
        if pd.isna(vis_m):
            return -1
        if vis_m < CLASS_THRESHOLDS["diversions_likely"]:
            return 2
        if vis_m < CLASS_THRESHOLDS["delays_likely"]:
            return 1
        return 0

    targets["target_class"] = targets["target_min_vis_m"].apply(classify)

    n_div = int((targets["target_class"] == 2).sum())
    n_del = int((targets["target_class"] == 1).sum())
    n_nor = int((targets["target_class"] == 0).sum())
    logger.info(f"  daily class distribution (all-year, pre-season-filter):")
    logger.info(f"    Diversions-Likely (<800m):   {n_div:>5,}")
    logger.info(f"    Delays-Likely (800-1600m):   {n_del:>5,}")
    logger.info(f"    Normal (>=1600m):            {n_nor:>5,}")
    return targets


def attach_target_date(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each obs with the target date whose prior night it belongs to."""
    df = df.copy()
    df["hour_npt"] = df["valid_npt"].dt.hour
    df["obs_date_npt"] = df["valid_npt"].dt.date

    evening_mask = df["hour_npt"] >= FEATURE_WINDOW_START_HOUR
    early_morning_mask = df["hour_npt"] < FEATURE_WINDOW_END_HOUR + 1

    df["target_date_npt"] = pd.NaT
    df.loc[evening_mask, "target_date_npt"] = (
        pd.to_datetime(df.loc[evening_mask, "obs_date_npt"]) + pd.Timedelta(days=1)
    )
    df.loc[early_morning_mask, "target_date_npt"] = pd.to_datetime(
        df.loc[early_morning_mask, "obs_date_npt"]
    )
    return df


def _snapshot(df_night: pd.DataFrame, hour_lo: int, hour_hi: int,
              cols: list[str]) -> pd.Series:
    """Median of cols across rows in [hour_lo, hour_hi]; NaN if no rows in window."""
    window = df_night[df_night["hour_npt"].between(hour_lo, hour_hi)]
    if window.empty:
        return pd.Series({c: np.nan for c in cols})
    return window[cols].median()


def build_features_for_one_night(df_night: pd.DataFrame) -> pd.Series:
    """Engineer ~18 features from one prior night's obs."""
    feats: dict[str, float] = {}

    # sunset snapshot
    sunset = _snapshot(
        df_night, *SUNSET_HOUR_RANGE,
        cols=["tempc", "dewpointc", "dewpoint_depression_c", "relh",
              "pressure_hpa", "wind_speed_ms", "visibility_m"],
    )
    feats["sunset_tempc"]              = sunset["tempc"]
    feats["sunset_dewpoint_depr_c"]    = sunset["dewpoint_depression_c"]
    feats["sunset_pressure_hpa"]       = sunset["pressure_hpa"]
    feats["sunset_wind_speed_ms"]      = sunset["wind_speed_ms"]
    feats["sunset_visibility_m"]       = sunset["visibility_m"]

    # pre-dawn snapshot
    predawn = _snapshot(
        df_night, *PREDAWN_HOUR_RANGE,
        cols=["tempc", "dewpointc", "dewpoint_depression_c", "relh",
              "pressure_hpa"],
    )
    feats["predawn_tempc"]             = predawn["tempc"]
    feats["predawn_dewpoint_depr_c"]   = predawn["dewpoint_depression_c"]
    feats["predawn_pressure_hpa"]      = predawn["pressure_hpa"]

    # overnight drift = classic radiation-fog signal
    feats["overnight_temp_drop_c"]     = sunset["tempc"] - predawn["tempc"]
    feats["overnight_dewpoint_depr_drop_c"] = (
        sunset["dewpoint_depression_c"] - predawn["dewpoint_depression_c"]
    )
    feats["overnight_pressure_change_hpa"] = (
        predawn["pressure_hpa"] - sunset["pressure_hpa"]
    )

    # wind over the whole night
    if df_night.empty:
        feats["night_mean_wind_speed_ms"] = np.nan
        feats["night_calm_fraction"]      = np.nan
    else:
        feats["night_mean_wind_speed_ms"] = df_night["wind_speed_ms"].mean()
        feats["night_calm_fraction"]      = df_night["wind_calm"].mean()

    # sky cover: clear nights cool faster -> fog
    if df_night.empty:
        feats["night_mean_sky_cover"]     = np.nan
        feats["night_clear_fraction"]     = np.nan
    else:
        feats["night_mean_sky_cover"]     = df_night["sky_cover_ord"].mean()
        feats["night_clear_fraction"]     = (
            df_night["sky_cover_ord"] <= 1
        ).mean()

    # pre-existing fog/mist during the night
    if df_night.empty:
        feats["night_mist_observed"] = 0
        feats["night_fog_observed"]  = 0
    else:
        feats["night_mist_observed"] = int(df_night["wx_mist"].max())
        feats["night_fog_observed"]  = int(df_night["wx_fog"].max())

    # QC: how many obs we had
    feats["night_obs_count"] = len(df_night)

    return pd.Series(feats)


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical sin/cos of day-of-year."""
    doy = df["date_npt"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


@dataclass
class BuildStats:
    """Counts for the validation report."""
    hourly_rows: int = 0
    candidate_dates: int = 0
    dates_after_qc: int = 0
    dates_after_season_filter: int = 0
    final_normal: int = 0
    final_delays: int = 0
    final_diversions: int = 0
    feature_missingness_pct: dict[str, float] = None

    def __post_init__(self) -> None:
        if self.feature_missingness_pct is None:
            self.feature_missingness_pct = {}


def build_daily_table(
    hourly_path: Path,
    output_path: Path,
    report_path: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Full Step 2 pipeline."""
    if logger is None:
        logger = setup_logging()

    stats = BuildStats()

    logger.info(f"Loading cleaned hourly table: {hourly_path}")
    df = pd.read_parquet(hourly_path)
    stats.hourly_rows = len(df)
    logger.info(f"  loaded {len(df):,} hourly rows")

    targets = compute_targets(df, logger)
    stats.candidate_dates = len(targets)

    logger.info("Engineering prior-night features for each target date...")
    df_with_td = attach_target_date(df)

    in_window = df_with_td.dropna(subset=["target_date_npt"]).copy()
    in_window["target_date_npt"] = pd.to_datetime(in_window["target_date_npt"])

    feature_rows = (
        in_window.groupby("target_date_npt")
        .apply(build_features_for_one_night, include_groups=False)
        .reset_index()
        .rename(columns={"target_date_npt": "date_npt"})
    )
    logger.info(f"  built features for {len(feature_rows):,} dates")

    merged = targets.merge(feature_rows, on="date_npt", how="inner")
    logger.info(f"  merged target+features: {len(merged):,} rows")

    merged = add_seasonal_features(merged)

    # need at least 2 morning obs so a single SPECI doesn't bias the min
    before = len(merged)
    merged = merged[merged["target_morning_obs"] >= 2].copy()
    logger.info(
        f"  dropped {before - len(merged):,} dates with <2 target-window obs"
    )

    # need at least 6 prior-night obs
    before = len(merged)
    merged = merged[merged["night_obs_count"] >= 6].copy()
    logger.info(
        f"  dropped {before - len(merged):,} dates with <6 prior-night obs"
    )
    stats.dates_after_qc = len(merged)

    # keep only fog season
    merged["month"] = merged["date_npt"].dt.month
    in_season = merged["month"].isin(FOG_SEASON_MONTHS)
    season_filtered = merged[in_season].copy()
    season_filtered = season_filtered.drop(columns=["month"])
    logger.info(
        f"  fog-season filter (Oct-Feb): "
        f"{len(season_filtered):,} / {len(merged):,} dates retained"
    )
    stats.dates_after_season_filter = len(season_filtered)

    stats.final_normal     = int((season_filtered["target_class"] == 0).sum())
    stats.final_delays     = int((season_filtered["target_class"] == 1).sum())
    stats.final_diversions = int((season_filtered["target_class"] == 2).sum())

    feature_cols = [c for c in season_filtered.columns
                    if c not in ("date_npt", "target_min_vis_m", "target_class",
                                 "target_morning_obs", "target_fog_observed")]
    for c in feature_cols:
        stats.feature_missingness_pct[c] = float(
            100 * season_filtered[c].isna().mean()
        )

    column_order = (
        ["date_npt"]
        + ["target_min_vis_m", "target_class",
           "target_morning_obs", "target_fog_observed"]
        + feature_cols
    )
    final = season_filtered[column_order].sort_values("date_npt").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False)
    logger.info(f"Wrote modelling table: {output_path}  ({len(final):,} rows)")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_step2_report(final, stats, report_path, logger)

    return final


def write_step2_report(df: pd.DataFrame, stats: BuildStats,
                       out_path: Path, logger: logging.Logger) -> None:
    logger.info(f"Writing Step 2 validation report: {out_path}")

    lines = []
    lines.append("# Step 2 Validation Report - daily modelling table")
    lines.append("")
    lines.append("## Pipeline counts")
    lines.append("")
    lines.append(f"- Hourly rows loaded from Step 1 :  {stats.hourly_rows:>8,}")
    lines.append(f"- Candidate target dates         :  {stats.candidate_dates:>8,}")
    lines.append(f"- Dates after observation QC     :  {stats.dates_after_qc:>8,}")
    lines.append(f"- Dates after fog-season filter  :  {stats.dates_after_season_filter:>8,}")
    lines.append("")
    lines.append(f"- Date range (NPT) : {df['date_npt'].min().date()} -> {df['date_npt'].max().date()}")
    lines.append("")
    lines.append("## Final daily class distribution (Oct-Feb only)")
    lines.append("")
    total = stats.final_normal + stats.final_delays + stats.final_diversions
    lines.append("| Class | Threshold | Count | Share |")
    lines.append("|---|---|---:|---:|")
    lines.append(
        f"| Diversions-Likely | vis < 800 m       | "
        f"{stats.final_diversions:,} | "
        f"{100*stats.final_diversions/total:.2f}% |"
    )
    lines.append(
        f"| Delays-Likely     | 800 <= vis < 1600m| "
        f"{stats.final_delays:,} | "
        f"{100*stats.final_delays/total:.2f}% |"
    )
    lines.append(
        f"| Normal            | vis >= 1600 m     | "
        f"{stats.final_normal:,} | "
        f"{100*stats.final_normal/total:.2f}% |"
    )
    lines.append("")
    lines.append("## Feature missingness (% NaN in final table)")
    lines.append("")
    lines.append("| Feature | % missing |")
    lines.append("|---|---:|")
    for feat, pct in sorted(stats.feature_missingness_pct.items(),
                             key=lambda kv: -kv[1]):
        lines.append(f"| `{feat}` | {pct:.2f}% |")
    lines.append("")
    lines.append("## Cross-check: target_class vs target_fog_observed")
    lines.append("")
    lines.append(
        "If the threshold-based class label and the independent METAR fog-code "
        "label agree, that is evidence the dataset is internally consistent."
    )
    lines.append("")
    ct = pd.crosstab(
        df["target_class"].map({0: "Normal", 1: "Delays", 2: "Diversions"}),
        df["target_fog_observed"].map({0: "no FG code", 1: "FG code seen"}),
        margins=True, margins_name="Total",
    )
    lines.append("```")
    lines.append(ct.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append(f"Total columns: {df.shape[1]}, total rows: {df.shape[0]:,}")
    lines.append("")
    lines.append("| Column | Dtype | Role |")
    lines.append("|---|---|---|")
    role_map = {
        "date_npt":              "identifier",
        "target_min_vis_m":      "target (regression)",
        "target_class":          "target (classification)",
        "target_morning_obs":    "QC flag",
        "target_fog_observed":   "independent label",
    }
    for c in df.columns:
        role = role_map.get(c, "feature")
        lines.append(f"| `{c}` | {df[c].dtype} | {role} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build daily modelling table from cleaned hourly METAR.",
    )
    p.add_argument(
        "--hourly", type=Path,
        default=Path("data/interim/vnkt_clean.parquet"),
        help="Path to the cleaned hourly parquet file from Step 1.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("data/processed/vnkt_modelling_table.parquet"),
        help="Path to write the final daily modelling table.",
    )
    p.add_argument(
        "--report", type=Path,
        default=Path("reports/step2_validation.md"),
        help="Path to write the Markdown validation report.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_daily_table(args.hourly, args.out, args.report)
