# Step 1 Validation Report — VNKT METAR cleaning

- **Raw rows loaded**:           165,040
- **After deduplication**:       165,040
- **After quality control**:     165,040
- **Date range (NPT)**:       2016-01-01 06:05:00 → 2026-05-23 05:15:00

## Column completeness (% non-missing)

| Column | % present |
|---|---:|
| `station` | 100.00% |
| `wind_u_ms` | 100.00% |
| `wx_haze` | 100.00% |
| `wx_mist` | 100.00% |
| `wx_freezing_fog` | 100.00% |
| `wx_fog` | 100.00% |
| `sky_cover_ord` | 100.00% |
| `valid_utc` | 100.00% |
| `wind_v_ms` | 100.00% |
| `wind_calm` | 100.00% |
| `valid_npt` | 100.00% |
| `metar` | 100.00% |
| `wind_speed_ms` | 99.92% |
| `visibility_m` | 99.89% |
| `tempc` | 99.86% |
| `dewpointc` | 99.85% |
| `dewpoint_depression_c` | 99.84% |
| `cloud_base_m` | 99.82% |
| `relh` | 99.78% |
| `pressure_hpa` | 93.93% |
| `wind_dir_deg` | 76.69% |
| `wxcodes` | 30.66% |

## QC violations (out-of-range values replaced with NaN)

| Field | Violations |
|---|---:|
| `tempc` | 1 |
| `dewpointc` | 1 |
| `wind_speed_ms` | 6 |
| `pressure_hpa` | 60 |
| `dewpoint_gt_temp` | 22 |

## Weather code event counts

- Fog (`FG`)  events: **295**
- Mist (`BR`) events: **23,572**

## Hourly-level class distribution (winter Nov/Dec/Jan only)

_Total winter observations with valid visibility: 42,225_

| Class | Threshold | Count | Share |
|---|---|---:|---:|
| Diversions-Likely | vis < 800 m       | 125 | 0.30% |
| Delays-Likely     | 800 ≤ vis < 1600m | 1,121 | 2.65% |
| Normal            | vis ≥ 1600 m      | 40,979 | 97.05% |

_Note: the modelling table built in Step 2 will be at **daily** resolution (next-morning 05:45–09:45 NPT minimum visibility per day), where the Diversions-Likely class rises to ~3.4% — an imbalance that is workable for ML._