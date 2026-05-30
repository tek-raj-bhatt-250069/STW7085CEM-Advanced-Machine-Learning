# Step 2 Validation Report - daily modelling table

## Pipeline counts

- Hourly rows loaded from Step 1 :   165,040
- Candidate target dates         :     3,785
- Dates after observation QC     :     3,743
- Dates after fog-season filter  :     1,561

- Date range (NPT) : 2016-01-02 -> 2026-02-28

## Final daily class distribution (Oct-Feb only)

| Class | Threshold | Count | Share |
|---|---|---:|---:|
| Diversions-Likely | vis < 800 m       | 43 | 2.75% |
| Delays-Likely     | 800 <= vis < 1600m| 389 | 24.92% |
| Normal            | vis >= 1600 m     | 1,129 | 72.33% |

## Feature missingness (% NaN in final table)

| Feature | % missing |
|---|---:|
| `overnight_pressure_change_hpa` | 3.84% |
| `sunset_pressure_hpa` | 3.01% |
| `overnight_temp_drop_c` | 1.86% |
| `overnight_dewpoint_depr_drop_c` | 1.86% |
| `sunset_tempc` | 1.15% |
| `sunset_dewpoint_depr_c` | 1.15% |
| `sunset_wind_speed_ms` | 1.09% |
| `sunset_visibility_m` | 1.09% |
| `predawn_pressure_hpa` | 0.96% |
| `predawn_tempc` | 0.83% |
| `predawn_dewpoint_depr_c` | 0.83% |
| `night_mean_wind_speed_ms` | 0.00% |
| `night_calm_fraction` | 0.00% |
| `night_mean_sky_cover` | 0.00% |
| `night_clear_fraction` | 0.00% |
| `night_mist_observed` | 0.00% |
| `night_fog_observed` | 0.00% |
| `night_obs_count` | 0.00% |
| `doy_sin` | 0.00% |
| `doy_cos` | 0.00% |

## Cross-check: target_class vs target_fog_observed

If the threshold-based class label and the independent METAR fog-code label agree, that is evidence the dataset is internally consistent.

```
target_fog_observed  FG code seen  no FG code  Total
target_class                                        
Delays                         30         359    389
Diversions                     42           1     43
Normal                          0        1129   1129
Total                          72        1489   1561
```

## Schema

Total columns: 25, total rows: 1,561

| Column | Dtype | Role |
|---|---|---|
| `date_npt` | datetime64[ns] | identifier |
| `target_min_vis_m` | float64 | target (regression) |
| `target_class` | int64 | target (classification) |
| `target_morning_obs` | int64 | QC flag |
| `target_fog_observed` | int64 | independent label |
| `sunset_tempc` | float64 | feature |
| `sunset_dewpoint_depr_c` | float64 | feature |
| `sunset_pressure_hpa` | float64 | feature |
| `sunset_wind_speed_ms` | float64 | feature |
| `sunset_visibility_m` | float64 | feature |
| `predawn_tempc` | float64 | feature |
| `predawn_dewpoint_depr_c` | float64 | feature |
| `predawn_pressure_hpa` | float64 | feature |
| `overnight_temp_drop_c` | float64 | feature |
| `overnight_dewpoint_depr_drop_c` | float64 | feature |
| `overnight_pressure_change_hpa` | float64 | feature |
| `night_mean_wind_speed_ms` | float64 | feature |
| `night_calm_fraction` | float64 | feature |
| `night_mean_sky_cover` | float64 | feature |
| `night_clear_fraction` | float64 | feature |
| `night_mist_observed` | float64 | feature |
| `night_fog_observed` | float64 | feature |
| `night_obs_count` | float64 | feature |
| `doy_sin` | float64 | feature |
| `doy_cos` | float64 | feature |