# Reports Folder Structure Implementation

**Date:** January 29, 2026  
**Status:** ✅ Complete

## Overview

Consolidated all report generation functions to use a unified `/reports` folder structure with versioned subfolders for each report type.

## Changes Made

### 1. **Config Update** (`src/config.py`)
- Added `REPORTS_DIR = workspace_root / "reports"` path
- Included in `get_all_directories()` for automatic creation

### 2. **Technical Report** (`create_weekly_technical_report_for_all_parks`)
**New Location:** `/reports/report_tech_weekly_YYYYMMDD_vXXX/`
- Folder name: `report_tech_weekly_{date}_{auto_versioned}`
- Contains:
  - `report.md` - Main markdown file
  - `fig1_daily_energy_timeseries_YYYYMMDD.png`
  - `fig2_pi_heatmap_full_YYYYMMDD.png`
  - `fig3_pi_heatmap_mtd_YYYYMMDD.png`
  - `fig4_revenue_mtd_all_parks_YYYYMMDD.png`
  - `fig5_revenue_mtd_grid_YYYYMMDD.png`

### 3. **STL Degradation Report** (`create_weekly_stl_report` / `create_weekly_stl_report_for_all_parks`)
**New Location:** `/reports/report_weekly_stl_YYYYMMDD_vXXX/`
- Folder name: `report_weekly_stl_{date}_{auto_versioned}`
- Contains:
  - `report.md` - Main markdown file
  - `stl_{park_name}_{date}.png` - Individual STL plots (one per park)

### 4. **Financial Report** (`create_financial_report_for_all_parks`)
**New Location:** `/reports/report_financial_YYYYMMDD_vXXX/`
- Folder name: `report_financial_{date}_{auto_versioned}`
- Contains:
  - `report.md` - Main markdown file
  - `financial_analysis_{park_name}.png` - Economic dashboard plot

## Versioning

Each report type gets automatic versioning using `generate_versioned_filename()`:
- First run: `report_tech_weekly_20260127_v001`
- Second run (same day): `report_tech_weekly_20260127_v002`
- Different day: `report_tech_weekly_20260128_v001`

## File References

Markdown files use **relative paths** to reference plots within the same folder:
```markdown
![Plot Name](fig1_daily_energy_timeseries_YYYYMMDD.png)
```

No complex relative path traversal (`../../plots/...`) needed anymore.

## Benefits

✅ **Organized:** Each report type has its own dedicated folder  
✅ **Self-contained:** Markdown + all plots in one location  
✅ **Versioned:** Multiple reports from the same day are preserved  
✅ **Simple:** No complex path logic in markdown (all files in same folder)  
✅ **Consistent:** All three report types follow same pattern  

## Usage Example

```python
from src.report_generator import create_weekly_technical_report_for_all_parks

# Report saved to: /reports/report_tech_weekly_20260129_v001/
report_path = create_weekly_technical_report_for_all_parks(
    daily_df=daily_data,
    pi_df=performance_index,
    daily_historical_df=historical_data,
)
# report_path = Path("/workspace/reports/report_tech_weekly_20260129_v001/report.md")
```

## Folder Structure After Reports Generated

```
reports/
├── report_tech_weekly_20260127_v001/
│   ├── report.md
│   ├── fig1_daily_energy_timeseries_20260127.png
│   ├── fig2_pi_heatmap_full_20260127.png
│   ├── fig3_pi_heatmap_mtd_20260127.png
│   ├── fig4_revenue_mtd_all_parks_20260127.png
│   └── fig5_revenue_mtd_grid_20260127.png
│
├── report_weekly_stl_20260127_v001/
│   ├── report.md
│   ├── stl_park_alpha_20260127.png
│   ├── stl_park_beta_20260127.png
│   └── stl_park_gamma_20260127.png
│
└── report_financial_20260127_v001/
    ├── report.md
    └── financial_analysis_park_alpha.png
```
