# Report Naming and Structure Fixes

**Date:** January 29, 2026  
**Status:** ✅ Complete

## Issues Resolved

### 1. **Report Filename Naming**
**Problem:** Report files were named `report.md` while folder was `report_type_YYYYMMDD_vXXX`

**Fix:** Report files now match folder names:
- Folder: `report_tech_weekly_20260129_v001/`
- File: `report_tech_weekly_20260129_v001.md` (inside folder)

- Folder: `report_weekly_stl_20260129_v001/`
- File: `report_weekly_stl_20260129_v001.md` (inside folder)

- Folder: `report_financial_20260129_v001/`
- File: `report_financial_20260129_v001.md` (inside folder)

### 2. **STL Report Not Being Produced**
**Problem:** The `create_weekly_stl_report_for_all_parks` wrapper wasn't passing `workspace_root` parameter correctly, causing the report generation to fail.

**Fix:** 
- Updated wrapper to always pass `workspace_root` to `create_weekly_stl_report`
- Set `save_dir=None` to force using config's `REPORTS_DIR`
- Updated notebook cell 28 to pass `workspace_root=config.WORKSPACE_ROOT` when calling the function

**Root Cause:** When `workspace_root` is not passed, the function tries to get fresh config, but without proper workspace_root context, it may fail or not find the correct reports directory.

## Changes Made

### `src/report_generator.py`

**Line ~167 (Technical Report):**
```python
# Before
report_path = report_folder / "report.md"

# After
report_filename = Path(report_folder).name + ".md"
report_path = report_folder / report_filename
```

**Line ~373 (STL Report):**
```python
# Before
report_path = report_folder / "report.md"

# After
report_filename = Path(report_folder).name + ".md"
report_path = report_folder / report_filename
```

**Line ~916 (Financial Report):**
```python
# Before
report_path = report_folder / "report.md"

# After
report_filename = Path(report_folder).name + ".md"
report_path = report_folder / report_filename
```

**Line ~1043 (STL Wrapper):**
```python
# Before
return create_weekly_stl_report(
    ...
    save_dir=save_dir,
    workspace_root=workspace_root,
    ...
)

# After
return create_weekly_stl_report(
    ...
    save_dir=None,  # Uses config to get REPORTS_DIR
    workspace_root=workspace_root,  # Pass through workspace_root
    ...
)
```

### `notebooks/01_prototype_pvgis_pi_heatmap.ipynb` (Cell 28)

Added `workspace_root=config.WORKSPACE_ROOT` parameter when calling `create_weekly_stl_report_for_all_parks`:
```python
weekly_stl_report = create_weekly_stl_report_for_all_parks(
    daily_df=daily_historical,
    report_date=None,
    parks=[showcase_park],
    max_parks=1,
    period=365,
    robust=True,
    anomaly_threshold=-3.0,
    min_consecutive_days=2,
    apply_log=False,
    workspace_root=config.WORKSPACE_ROOT,  # <- ADDED
    dpi=180,
)
```

## New Report Structure

```
reports/
├── report_tech_weekly_20260129_v001/
│   ├── report_tech_weekly_20260129_v001.md   ← Matches folder name
│   ├── fig1_daily_energy_timeseries_20260129.png
│   ├── fig2_pi_heatmap_full_20260129.png
│   ├── fig3_pi_heatmap_mtd_20260129.png
│   ├── fig4_revenue_mtd_all_parks_20260129.png
│   └── fig5_revenue_mtd_grid_20260129.png
│
├── report_weekly_stl_20260129_v001/
│   ├── report_weekly_stl_20260129_v001.md   ← Matches folder name
│   ├── stl_park_alpha_20260129.png
│   ├── stl_park_beta_20260129.png
│   └── stl_park_gamma_20260129.png
│
└── report_financial_20260129_v001/
    ├── report_financial_20260129_v001.md   ← Matches folder name
    └── financial_analysis_park_alpha.png
```

## Testing

Run the notebook cells in order:
1. Cell 26 (Technical Report) - Should save to `/reports/report_tech_weekly_20260129_v001/report_tech_weekly_20260129_v001.md`
2. Cell 25 (Financial Report) - Should save to `/reports/report_financial_20260129_v001/report_financial_20260129_v001.md`
3. Cell 28 (STL Report) - Should now produce output and save to `/reports/report_weekly_stl_20260129_v001/report_weekly_stl_20260129_v001.md`

All reports will have consistent naming: **folder name = report filename (without .md)**
