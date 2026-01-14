# Daily Average PV KPI Heatmap

A concise guide to setting up the environment, cleaning the data, visualizing results, and computing KP performance indicators (KPIs) for PV parks using PVGIS expectations.

## Overview
- Load daily energy data from Excel.
- Clean data by detecting columns with mostly missing or zero values and outliers.
- Compute expected production via PVGIS and derive KPIs.
- Visualize distributions, per-park time series, boxplots, and heatmaps.

## Setup
1. Activate your environment:
   ```bash
   conda activate pv-kpi
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data
- Input file: `data/daily_energy.xlsx`
- Expected columns:
  - `Timestamp`: pandas-parseable datetime
  - One column per PV park (header contains name and capacity, e.g. `Park [1234 kWp]`)
- The notebook builds:
  - `daily`: a DataFrame indexed by normalized `date` with one column per park
  - `park_cols`: list of park column names

## Cleaning Workflow
This workflow is implemented in `notebooks/01_prototype_pvgis_pi_heatmap.ipynb`.

1. Inspect NaNs
   - Function: `barplot_nan_count(daily)`
   - Shows NaN counts and percentages per column; highlights 50% threshold.

2. Detect problematic columns
   - Function: `detect_problematic_columns(daily, nan_threshold=0.5, zero_threshold=0.8)`
   - Flags columns where:
     - NaN fraction ≥ 50%
     - Zero fraction among non-NaN ≥ 80%
   - Stores a summary table with `nan_pct` and `zero_pct`.

3. Remove problematic columns (but keep a copy)
   - Code creates `daily_problematic = daily[problematic_cols].copy()`
   - Updates `daily = daily.drop(columns=problematic_cols)` and `park_cols` accordingly.

4. Detect outliers via IQR
   - Function: `detect_outliers_iqr(daily, multiplier=1.5)`
   - Computes per-column bounds with the IQR rule and returns outlier masks and counts.
   - Visualize: `visualize_outliers(outliers)` shows counts and percentages.

### Equations: IQR Outliers
- Quartiles: $Q_1 = \text{quantile}(0.25)$, $Q_3 = \text{quantile}(0.75)$, $\text{IQR} = Q_3 - Q_1$.
- Bounds: $\text{lower} = Q_1 - k\,\text{IQR}$, $\text{upper} = Q_3 + k\,\text{IQR}$ (typical $k=1.5$; use $k=3.0$ for extreme outliers).
- Outlier condition: $x < \text{lower}$ or $x > \text{upper}$.

## KPI Computation
Functions are provided in `src/pvgis_pi_heatmap.py` and used by the notebook.

1. PVGIS Expected Energy
   - Hourly PVGIS data is fetched and cached (`pvgis_cache/`), then converted to expected daily energy.
   - PVGIS returns PV power `P` in watts. Expected daily energy in kWh is computed by summing hourly power (converted to kW) per day:
     $$\text{expected\_kWh}(d) = \sum_{h \in d} \frac{P_h}{1000}$$
   - Timezone conversion is applied to align with local time (e.g., `Europe/Athens`).

2. Performance Index (PI)
   - Definition:
     $$\text{PI}(d,\text{park}) = \frac{\text{measured\_kWh}(d,\text{park})}{\text{expected\_kWh}(d,\text{park})}$$
   - Interprets how production compares to PVGIS expectation.

3. Robust Anomaly Score (Rolling Median/MAD)
   - For each park, compute rolling median and MAD over a window (e.g., 31 days). The robust z-score is:
     $$z(d) = \frac{x(d) - \text{median}_{\text{window}}(x)}{1.4826 \times \text{MAD}_{\text{window}}(x)}$$
   - The constant $1.4826$ scales MAD to be comparable to standard deviation under normality.
   - This produces a `score` DataFrame; thresholds (e.g., $|z| \ge 3$) can indicate anomalies.

4. Flags
   - A simple flag can be derived from `score` (example policy):
     - $\text{flag} = -1$ if $z \le -3$ (significantly low)
     - $\text{flag} = +1$ if $z \ge +3$ (significantly high)
     - $\text{flag} = 0$ otherwise
   - The notebook renders a heatmap labeled "Flags (-1 low, 0 ok, +1 high)".

## Visualization
The notebook provides clear, per-park visualizations:

- Per-park time series grid:
  - `lineplot_timeseries_per_column(daily, ncols=3)` plots one line per subplot for each park.
- Per-park distributions:
  - `histplot_distribution_per_column(daily, ncols=3, bins=40, show_stats=True)` displays histograms with mean/median lines.
- Boxplots by park:
  - `plot_boxplot(daily, title="Daily Boxplot")` summarizes distributions and outliers.
- KPI Heatmaps:
  - `plot_heatmap(pi, ...)` uses the `viridis` colormap for PI.
  - `plot_heatmap(score, ...)` for robust anomaly scores.
  - `plot_heatmap(flag, ...)` for anomaly flags.

## Workflow Summary
1. Load data and construct `daily` and `park_cols`.
2. Explore with line and bar plots.
3. Analyze data quality: NaNs, zeros, and IQR outliers.
4. Remove problematic columns (retain `daily_problematic`).
5. Build `meta` for PVGIS (randomized lat/lon placeholders, kWp parsed from headers).
6. Compute `pi`, `score`, and `flag` via PVGIS expectations and robust statistics.
7. Visualize with boxplots and heatmaps.

## Notes
- Column headers should include capacity (e.g., `[1234 kWp]`) to parse `kwp`.
- PVGIS requests are cached in `pvgis_cache/` with a content hash to avoid redundant calls.
- Ensure the timezone (e.g., `Europe/Athens`) matches your data context.
- You can tune thresholds (`nan_threshold`, `zero_threshold`, `IQR multiplier`) and rolling window lengths for your operational needs.

## Running the Notebook
Open `notebooks/01_prototype_pvgis_pi_heatmap.ipynb` and execute cells in order:
1. Imports and workspace setup
2. Data load and `daily` construction
3. Quality analysis and cleaning
4. KPI computation (`compute_pi_anomaly`)
5. Visualizations (distributions, boxplots, heatmaps)

If you need help customizing thresholds or adding additional KPIs, consider extending the notebook with parameter cells for reproducible runs.
