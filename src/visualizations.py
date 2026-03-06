"""
Visualization Functions for PV Analysis

This module provides plotting functions for time series, heatmaps, 
distributions, and revenue analysis.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pvgis_pi_heatmap import parse_kwp_from_header, short_label
from src.utils import save_figure
from src.metrics_calculator import annual_mtd_energy


def _infer_heatmap_metric(mat: pd.DataFrame, title: str | None) -> str:
    title_lower = (title or "").lower()
    if "ratio" in title_lower or "pi" in title_lower or "performance" in title_lower:
        return "performance"
    if "score" in title_lower or "anomaly" in title_lower:
        return "anomaly_score"
    if "flag" in title_lower or "flags" in title_lower:
        return "flags"

    values = mat.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "generic"
    unique_vals = set(np.unique(values))
    if unique_vals.issubset({-1, 0, 1}):
        return "flags"
    if np.nanmin(values) >= 0 and np.nanmax(values) <= 2.5:
        return "performance"
    if np.nanmin(values) < 0 and np.nanmax(values) > 0 and np.nanstd(values) > 1.5:
        return "anomaly_score"
    return "generic"


def _park_label(col) -> str:
    try:
        return short_label(col)
    except Exception:
        return str(col)


def _generate_heatmap_findings(mat: pd.DataFrame, title: str | None = None) -> str:
    """Generate findings and interpretation for heatmap data."""
    findings = []
    
    # Time coverage
    if len(mat) > 0:
        date_range = f"{mat.index.min().date()} to {mat.index.max().date()}"
        findings.append(f"📅 Time Period: {date_range} ({len(mat)} days)")
    
    # Park coverage
    n_parks = len(mat.columns)
    findings.append(f"🏢 Parks: {n_parks} monitored")
    
    # Data quality
    total_cells = len(mat) * n_parks
    missing = mat.isna().sum().sum()
    completeness = 100 * (1 - missing / total_cells) if total_cells > 0 else 0
    findings.append(f"✓ Data Completeness: {completeness:.1f}%")
    
    # Value statistics
    mat_values = mat.values.flatten()
    mat_values = mat_values[~np.isnan(mat_values)]
    if len(mat_values) > 0:
        findings.append(f"📊 Mean: {np.mean(mat_values):.2f}, StdDev: {np.std(mat_values):.2f}")
        findings.append(f"📈 Range: {np.min(mat_values):.2f} - {np.max(mat_values):.2f}")
    
    # Anomalies
    zero_count = (mat == 0).sum().sum()
    if total_cells > 0:
        zero_pct = 100 * zero_count / total_cells
        if zero_pct > 5:
            findings.append(f"⚠️  Zero values: {zero_pct:.1f}% (potential outages or low production)")

    # PV-engineer focused findings
    metric = _infer_heatmap_metric(mat, title)
    park_stats = []

    if metric == "performance":
        for col in mat.columns:
            series = mat[col].dropna()
            if len(series) == 0:
                continue
            underperf_pct = 100 * np.mean(series < 0.90)
            severe_pct = 100 * np.mean(series < 0.80)
            median_val = float(np.median(series))
            park_stats.append((col, underperf_pct, severe_pct, median_val))
        park_stats.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)
        if park_stats:
            findings.append("🔧 PV Performance Hotspots (most days below PI/ratio 0.90):")
            for col, underperf_pct, severe_pct, median_val in park_stats[:3]:
                label = _park_label(col)
                findings.append(
                    f"  - {label}: <0.90 on {underperf_pct:.1f}% of days, <0.80 on {severe_pct:.1f}%, median={median_val:.2f}"
                )
            findings.append("💡 Interpretation: Persistent low PI/ratio suggests losses (soiling, shading, curtailment, inverter issues)")

    elif metric == "anomaly_score":
        for col in mat.columns:
            series = mat[col].dropna()
            if len(series) == 0:
                continue
            high_anom_pct = 100 * np.mean(np.abs(series) >= 3.0)
            park_stats.append((col, high_anom_pct))
        park_stats.sort(key=lambda x: x[1], reverse=True)
        if park_stats:
            findings.append("🔧 Anomaly Hotspots (|z| ≥ 3):")
            for col, high_anom_pct in park_stats[:3]:
                label = _park_label(col)
                findings.append(f"  - {label}: {high_anom_pct:.1f}% of days")
            findings.append("💡 Interpretation: Clusters of high |z| often indicate sensor drift, data gaps, or true underperformance")

    elif metric == "flags":
        for col in mat.columns:
            series = mat[col].dropna()
            if len(series) == 0:
                continue
            negative_pct = 100 * np.mean(series < 0)
            park_stats.append((col, negative_pct))
        park_stats.sort(key=lambda x: x[1], reverse=True)
        if park_stats:
            findings.append("🔧 Flag Hotspots (negative flags):")
            for col, negative_pct in park_stats[:3]:
                label = _park_label(col)
                findings.append(f"  - {label}: {negative_pct:.1f}% of days")
            findings.append("💡 Interpretation: Recurrent negative flags indicate repeated underperformance vs PVGIS")

    else:
        for col in mat.columns:
            series = mat[col].dropna()
            if len(series) == 0:
                continue
            median_val = float(np.median(series))
            park_stats.append((col, median_val))
        park_stats.sort(key=lambda x: x[1])
        if park_stats:
            findings.append("🔧 Lowest median parks (potential underperformance):")
            for col, median_val in park_stats[:3]:
                label = _park_label(col)
                findings.append(f"  - {label}: median={median_val:.2f}")
            findings.append("💡 Interpretation: Low medians are consistent with chronic losses or data issues")
    
    return "\n".join(findings)


def _generate_timeseries_findings(df: pd.DataFrame) -> str:
    """Generate findings and interpretation for time series data."""
    findings = []
    
    findings.append(f"📊 Time Series Analysis ({len(df.columns)} series)")
    findings.append("")
    
    # Overall trend
    if len(df) > 0:
        first_val = df.iloc[0].mean()
        last_val = df.iloc[-1].mean()
        if first_val > 0:
            trend = 100 * (last_val - first_val) / first_val
            direction = "📈 INCREASING" if trend > 5 else "📉 DECREASING" if trend < -5 else "➡️  STABLE"
            findings.append(f"{direction}: {trend:+.1f}% change")
    
    # Variability
    cv = df.std().mean() / df.mean().mean() if df.mean().mean() != 0 else 0
    if cv > 0.3:
        findings.append(f"⚠️  High variability (CV={cv:.2f}): Expect frequent fluctuations")
    elif cv > 0.15:
        findings.append(f"✓ Moderate variability (CV={cv:.2f}): Normal PV generation pattern")
    else:
        findings.append(f"✓ Low variability (CV={cv:.2f}): Consistent performance")
    
    # Outliers
    Q1 = df.quantile(0.25).mean()
    Q3 = df.quantile(0.75).mean()
    IQR = Q3 - Q1
    outlier_bound = Q3 + 1.5 * IQR
    outliers = (df > outlier_bound).sum().sum()
    if outliers > 0:
        findings.append(f"🔴 Outliers detected: {outliers} anomalous readings")
    
    # Seasonality hint
    findings.append("💡 Interpretation: Rolling average (red) reveals trends; IQR bands show expected variation")
    
    return "\n".join(findings)


def _generate_distribution_findings(df: pd.DataFrame) -> str:
    """Generate findings and interpretation for distribution data."""
    findings = []
    
    findings.append(f"📊 Distribution Analysis ({len(df.columns)} series)")
    findings.append("")
    
    # Central tendency
    overall_mean = df.mean().mean()
    overall_median = df.median().mean()
    findings.append(f"Central Tendency: Mean={overall_mean:.2f}, Median={overall_median:.2f}")
    
    # Skewness indicators
    skewness = df.skew().mean()
    if abs(skewness) > 0.5:
        direction = "right-skewed" if skewness > 0 else "left-skewed"
        findings.append(f"⚠️  {direction.capitalize()} distribution (skewness={skewness:.2f})")
    else:
        findings.append(f"✓ Near-symmetric distribution")
    
    # Spread
    overall_std = df.std().mean()
    cv = overall_std / overall_mean if overall_mean > 0 else 0
    findings.append(f"Spread: σ={overall_std:.2f} (CV={cv:.2f})")
    
    # Data quality
    missing_pct = 100 * df.isna().sum().sum() / (len(df) * len(df.columns)) if len(df) * len(df.columns) > 0 else 0
    findings.append(f"Data Quality: {100-missing_pct:.1f}% complete")
    
    findings.append("💡 Interpretation: Peak indicates most common values; tails show rare extremes")
    
    return "\n".join(findings)


def _generate_scatter_findings(measured_df: pd.DataFrame, reference_df: pd.DataFrame) -> str:
    """Generate findings and interpretation for scatter plot comparisons."""
    findings = []
    
    findings.append("📊 Measured vs Reference Comparison")
    findings.append("")
    
    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col

    def align_by_calendar_day(df_m, df_r):
        measured_parks = set(get_column_key(col) for col in df_m.columns)
        reference_parks = set(get_column_key(col) for col in df_r.columns)
        common_parks = sorted(measured_parks & reference_parks)
        if len(common_parks) == 0:
            return np.array([]), np.array([])

        measured_cols = {get_column_key(col): col for col in df_m.columns}
        reference_cols = {get_column_key(col): col for col in df_r.columns}

        measured_all = []
        reference_all = []

        for park_id in common_parks:
            measured_col = measured_cols[park_id]
            reference_col = reference_cols[park_id]
            measured = df_m[measured_col].dropna()
            reference = df_r[reference_col].dropna()
            if len(measured) == 0 or len(reference) == 0:
                continue

            measured_mds = pd.Series([(d.month, d.day) for d in measured.index], index=measured.index)
            reference_mds = pd.Series([(d.month, d.day) for d in reference.index], index=reference.index)

            for md in reference_mds.unique():
                meas_mask = measured_mds == md
                if meas_mask.any():
                    meas_vals = measured[meas_mask]
                    ref_val = reference[reference_mds == md].iloc[0]
                    measured_all.append(meas_vals.values)
                    reference_all.append(np.full_like(meas_vals.values, ref_val))

        if measured_all and reference_all:
            return np.concatenate(measured_all), np.concatenate(reference_all)
        return np.array([]), np.array([])

    def align_by_calendar_day_per_park(df_m, df_r):
        measured_parks = set(get_column_key(col) for col in df_m.columns)
        reference_parks = set(get_column_key(col) for col in df_r.columns)
        common_parks = sorted(measured_parks & reference_parks)
        if len(common_parks) == 0:
            return {}

        measured_cols = {get_column_key(col): col for col in df_m.columns}
        reference_cols = {get_column_key(col): col for col in df_r.columns}

        park_data = {}
        for park_id in common_parks:
            measured_col = measured_cols[park_id]
            reference_col = reference_cols[park_id]
            measured = df_m[measured_col].dropna()
            reference = df_r[reference_col].dropna()
            if len(measured) == 0 or len(reference) == 0:
                continue

            measured_mds = pd.Series([(d.month, d.day) for d in measured.index], index=measured.index)
            reference_mds = pd.Series([(d.month, d.day) for d in reference.index], index=reference.index)

            measured_list = []
            reference_list = []
            for md in reference_mds.unique():
                meas_mask = measured_mds == md
                if meas_mask.any():
                    meas_vals = measured[meas_mask]
                    ref_val = reference[reference_mds == md].iloc[0]
                    measured_list.append(meas_vals.values)
                    reference_list.append(np.full_like(meas_vals.values, ref_val))

            if measured_list and reference_list:
                park_data[str(park_id)] = (
                    np.concatenate(measured_list),
                    np.concatenate(reference_list),
                )

        return park_data

    def most_problematic_parks(df_m, df_r, tol_pct=10.0, top_n=3):
        park_data = align_by_calendar_day_per_park(df_m, df_r)
        worst = []
        for park_id, (m_vals, r_vals) in park_data.items():
            if len(m_vals) == 0:
                continue
            mean_vals = (m_vals + r_vals) / 2
            valid_mean = mean_vals > 0
            if not np.any(valid_mean):
                continue
            diff_vals = (m_vals[valid_mean] - r_vals[valid_mean]) / mean_vals[valid_mean] * 100
            agreement_pct = 100 * np.mean(np.abs(diff_vals) <= tol_pct)
            bias_pct = float(np.mean(diff_vals))
            try:
                label = short_label(park_id)
            except Exception:
                label = str(park_id)
            worst.append((label, float(agreement_pct), abs(bias_pct)))
        worst.sort(key=lambda x: (x[1], -x[2]))
        return worst[:top_n]

    m_vals, r_vals = align_by_calendar_day(measured_df, reference_df)
    
    if len(m_vals) > 0 and len(r_vals) > 0:
        # Performance ratio
        ratio = m_vals.mean() / r_vals.mean() if r_vals.mean() != 0 else 0
        ratio_pct = 100 * ratio
        if ratio_pct > 95:
            finding = "✓ GOOD performance"
        elif ratio_pct > 80:
            finding = "⚠️  MODERATE performance"
        else:
            finding = "🔴 POOR performance"
        findings.append(f"{finding}: {ratio_pct:.1f}% of expected output")
        
        # Correlation
        if len(m_vals) > 2:
            corr = np.corrcoef(m_vals, r_vals)[0, 1]
            if not np.isnan(corr):
                findings.append(f"Correlation: {corr:.3f} (tracking PVGIS variability)")
        
        # Residuals
        residuals = m_vals - r_vals
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        findings.append(f"Errors: RMSE={rmse:.2f} kWh/day, MAE={mae:.2f} kWh/day")
        
        # Bias
        bias = np.mean(residuals)
        bias_pct = (bias / r_vals.mean()) * 100 if r_vals.mean() != 0 else 0
        if abs(bias) > mae * 0.1:
            direction = "underperforming" if bias < 0 else "overperforming"
            findings.append(f"⚠️  Systematic {direction}: {bias:+.2f} kWh/day ({bias_pct:+.1f}%)")

        agreement_pct = 100 * np.mean(np.abs((m_vals - r_vals) / ((m_vals + r_vals) / 2)) <= 0.10)
        findings.append(f"Agreement within ±10%: {agreement_pct:.1f}%")

        worst = most_problematic_parks(measured_df, reference_df, tol_pct=10.0)
        if worst:
            findings.append("🔧 Most problematic parks (lowest agreement ±10%):")
            for label, agreement_pct, bias_abs in worst:
                findings.append(f"  - {label}: agreement={agreement_pct:.1f}%, |bias|={bias_abs:.1f}%")
    else:
        findings.append("⚠️  No paired observations after alignment")
        findings.append("   Check date coverage, timezone consistency, and park mapping")
    
    findings.append("💡 Interpretation: Points on diagonal = PVGIS agreement; persistent offsets indicate losses or systematic bias")
    findings.append("Look for seasonal curvature (temperature, clipping) and spread (soiling, curtailment, availability)")
    
    return "\n".join(findings)


def extract_park_name_before_pcc(col):
    """Extract readable park name from column name."""
    try:
        return short_label(col)
    except Exception:
        return str(col)


def plot_heatmap(
    mat: pd.DataFrame,
    title: str,
    vmin=None,
    vmax=None,
    start_date=None,
    end_date=None,
    config=None,
    plot_name: str | None = None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = True,
    add_date: bool = True,
):
    """
    Plot a heatmap of date x park data.
    
    Parameters
    ----------
    mat : pd.DataFrame
        Date-indexed DataFrame with parks as columns
    title : str
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    start_date, end_date : str or Timestamp, optional
        Date range filter
    config : WorkspaceConfig, optional
        Workspace configuration object. If provided, uses config.PLOTS_DIR as default save_dir
    plot_name : str, optional
        Short name for the plot (e.g., "power_ratio_heatmap"). Used for filename if provided.
    save : bool
        Whether to save the figure
    save_dir : Path or str
        Directory for saving. If None and config provided, uses config.PLOTS_DIR
    base_filename : str
        Base filename for saving (deprecated, use plot_name instead)
    dpi : int
        Resolution
    fmt : str
        Image format
    auto_version : bool
        If True, automatically adds date (YYYYMMDD) and version (v001, v002, etc.)
    add_date : bool
        If True with auto_version, adds YYYYMMDD to filename
        
    Returns
    -------
    Path or None
        Path to saved file if save=True
    """
    # Determine save directory
    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"
    
    # Determine filename
    if base_filename is None:
        if plot_name:
            base_filename = plot_name
        else:
            base_filename = "heatmap"
    
    # Filter by date range if specified
    if start_date is not None or end_date is not None:
        tz = mat.index.tz if isinstance(mat.index, pd.DatetimeIndex) else None
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            if tz and start_date.tzinfo is None:
                start_date = start_date.tz_localize(tz)
            if tz is None and start_date.tzinfo is not None:
                start_date = start_date.tz_convert(None)
            mat = mat[mat.index >= start_date]
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            if tz and end_date.tzinfo is None:
                end_date = end_date.tz_localize(tz)
            if tz is None and end_date.tzinfo is not None:
                end_date = end_date.tz_convert(None)
            mat = mat[mat.index <= end_date]
        if len(mat) == 0:
            print("Warning: No data found in the specified date range")
            return None

    m = mat.T.copy()
    y = []
    for col in m.index:
        if isinstance(col, tuple):
            park_id = col[0]
            if "__" in park_id:
                parts = park_id.split("__")
                park_name = parts[0].replace("p_", "").replace("_", " ").title()
                if len(parts) > 1 and "kwp" in parts[1].lower():
                    capacity = parts[1].replace("_kwp", "").replace("kwp", "")
                    y.append(f"{park_name} ({capacity} kWp)")
                else:
                    y.append(park_name)
            else:
                y.append(park_id.replace("_", " ").title())
        else:
            try:
                y.append(f"{str(col)} ({parse_kwp_from_header(col):.0f} kWp)")
            except Exception:
                y.append(str(col))
    m.index = y

    fig, ax = plt.subplots(figsize=(14, max(6, 0.28 * len(m.index))))
    im = ax.imshow(m.values, aspect="auto", interpolation="nearest", cmap="turbo", vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(np.arange(len(m.index)))
    ax.set_yticklabels(m.index, fontsize=10)

    dates = pd.to_datetime(m.columns)
    step = max(1, len(dates) // 12)
    xticks = np.arange(0, len(dates), step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates[::step]], rotation=45, ha="right")

    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("KPI", rotation=90, fontsize=10)
    plt.tight_layout()

    saved_path = save_figure(
        fig=fig,
        title_prefix=title,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt=fmt,
        auto_version=auto_version,
        add_date=add_date,
    )

    # Generate findings
    findings = _generate_heatmap_findings(mat, title=title)
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)

    plt.show()
    return saved_path


def lineplot_timeseries_per_column(
    df: pd.DataFrame,
    title_prefix: str = "Time Series",
    ylabel: str = "Value",
    ncols: int = 3,
    sharex: bool = True,
    sharey: bool = False,
    rolling_window: int = 7,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = True,
    add_date: bool = True,
):
    """
    Plot one line chart per column in a grid of subplots with rolling average and IQR bands.

    Parameters
    -----------
    df: pd.DataFrame
        Date-indexed DataFrame; each column is a park/series
    title_prefix: str
        Prefix used in subplot titles and default filename
    ylabel: str
        Y-axis label for all subplots (default: "Value")
    ncols: int
        Number of columns in the subplot grid
    sharex/sharey: bool
        Share axes across subplots
    rolling_window: int
        Window size for rolling average (default: 7 days)
    save: bool
        If True, saves the figure
    save_dir: str | Path | None
        Directory where the figure will be saved
    base_filename: str | None
        Base filename without extension; if None, derived from title_prefix
    dpi: int
        Resolution for the saved image
    fmt: str
        File format for saving (e.g., "png", "pdf", "svg")
        
    Returns
    -------
    Path or None
        Path to saved file if save=True
    """
    cols = list(df.columns)
    if len(cols) == 0:
        print("No columns to plot.")
        return None

    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(6 * ncols, 24), max(2.8 * nrows, 4))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    axes_flat = axes.ravel()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        series = df[col]
        
        # Plot raw data with 40% opacity
        ax.plot(df.index, series, linewidth=1.0, alpha=0.6, color='steelblue', label='Raw data')
        
        # Compute rolling average
        rolling_mean = series.rolling(window=rolling_window, center=True, min_periods=1).mean()
        ax.plot(df.index, rolling_mean, linewidth=2.0, color='red', label=f'{rolling_window}-day avg')
        
        # Compute IQR bounds (Q1 - 1.5*IQR and Q3 + 1.5*IQR)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Plot IQR bounds as horizontal lines
        ax.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='IQR low')
        ax.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='IQR high')
        
        # Handle tuple columns (park_id, signal, unit)
        if isinstance(col, tuple) and len(col) >= 3:
            park_id, signal, unit = col[0], col[1], col[2]
            # Try to extract capacity and park name
            try:
                import re
                m = re.search(r'(\d+)\s*kWp?\s*[_\-–—]?\s*(.+)', park_id)
                if m:
                    capacity_kwp = m.group(1)
                    park_name = m.group(2).strip()
                    label = f"{park_name} ({capacity_kwp} kWp)"
                else:
                    label = park_id
            except Exception:
                label = park_id
        else:
            try:
                label = extract_park_name_before_pcc(col)
            except Exception:
                label = str(col)
        
        ax.set_title(f"{title_prefix}: {label}")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes_flat[:len(cols)]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')

    plt.tight_layout()

    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"

    saved_path = save_figure(
        fig=fig,
        title_prefix=title_prefix,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt=fmt,
        auto_version=auto_version,
        add_date=add_date,
    )

    # Generate findings
    findings = _generate_timeseries_findings(df)
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)

    plt.show()
    return saved_path


def histplot_distribution_per_column(
    df: pd.DataFrame,
    title_prefix: str = "Distribution",
    xlabel: str = "Value",
    ncols: int = 3,
    bins: int = 30,
    density: bool = False,
    dropna: bool = True,
    sharex: bool = False,
    sharey: bool = False,
    show_stats: bool = True,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = True,
    add_date: bool = True,
):
    """
    Plot one histogram per column in a grid of subplots and optionally save the figure.

    Parameters
    -----------
    df: pd.DataFrame
        DataFrame indexed by date; each column is a park/series
    title_prefix: str
        Prefix used in subplot titles and default filename
    xlabel: str
        X-axis label for all subplots (default: "Value")
    ncols: int
        Number of columns in the subplot grid
    bins: int
        Histogram bin count
    density: bool
        If True, normalize histogram to form a probability density
    dropna: bool
        If True, exclude NaNs from each column
    sharex/sharey: bool
        Share axes across subplots
    show_stats: bool
        If True, draw vertical lines for mean and median
    save: bool
        If True, saves the figure
    save_dir: str | Path | None
        Directory where the figure will be saved
    base_filename: str | None
        Base filename without extension; if None, derived from title_prefix
    dpi: int
        Resolution for the saved image
    fmt: str
        File format for saving (e.g., "png", "pdf", "svg")
        
    Returns
    -------
    Path or None
        Path to saved file if save=True
    """
    cols = list(df.columns)
    if len(cols) == 0:
        print("No columns to plot.")
        return None

    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(6 * ncols, 24), max(2.8 * nrows, 4))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    axes_flat = axes.ravel()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        values = df[col]
        if dropna:
            values = values.dropna()

        ax.hist(values, bins=bins, alpha=0.75, color='steelblue', edgecolor='white', density=density)

        try:
            label = extract_park_name_before_pcc(col)
        except Exception:
            label = col
        ax.set_title(f"{title_prefix}: {label}")
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Count")

        if show_stats and len(values) > 0:
            mean_val = float(values.mean())
            median_val = float(values.median())
            ax.axvline(mean_val, color='orange', linestyle='--', linewidth=1, label='Mean')
            ax.axvline(median_val, color='crimson', linestyle='--', linewidth=1, label='Median')
            ax.legend(fontsize=8)

    # Hide any unused axes
    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"

    saved_path = save_figure(
        fig=fig,
        title_prefix=title_prefix,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt=fmt,
        auto_version=auto_version,
        add_date=add_date,
    )

    # Generate findings
    findings = _generate_distribution_findings(df)
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)

    plt.show()
    return saved_path


def scatterplot_measured_vs_reference_per_column(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    title_prefix: str = "Measured vs Reference",
    xlabel: str = "Reference [kWh/day]",
    ylabel: str = "Measured [kWh/day]",
    ncols: int = 3,
    alpha: float = 0.5,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = True,
    add_date: bool = True,
):
    """
    Plot scatter plots comparing measured vs reference data for each column in a grid.

    Parameters
    -----------
    measured_df: pd.DataFrame
        Date-indexed DataFrame with measured data; each column is a park/series
    reference_df: pd.DataFrame
        Date-indexed DataFrame with reference (expected) data; same structure as measured_df
    title_prefix: str
        Prefix used in subplot titles and default filename
    xlabel: str
        X-axis label for all subplots
    ylabel: str
        Y-axis label for all subplots
    ncols: int
        Number of columns in the subplot grid
    alpha: float
        Transparency of scatter points
    config: WorkspaceConfig, optional
        Workspace configuration object
    save: bool
        If True, saves the figure
    save_dir: str | Path | None
        Directory where the figure will be saved
    base_filename: str | None
        Base filename without extension
    dpi: int
        Resolution for the saved image
    fmt: str
        File format for saving
    auto_version: bool
        If True, automatically adds version to filename
    add_date: bool
        If True with auto_version, adds YYYYMMDD to filename
        
    Returns
    -------
    Path or None
        Path to saved file if save=True
    """
    # Helper function to extract park_id from column name (handles tuples and strings)
    def get_column_key(col):
        """Extract the first element from tuple columns, or return string as-is."""
        if isinstance(col, tuple):
            return col[0]
        return col
    
    # Get unique park IDs from both dataframes
    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    
    # Get common park IDs
    common_parks = sorted(measured_parks & reference_parks)
    
    if len(common_parks) == 0:
        print("No common parks between measured and reference data.")
        return None
    
    # Build mapping of park_id to actual column names in each dataframe
    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}
    
    n = len(common_parks)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(6 * ncols, 24), max(3 * nrows, 4))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    axes_flat = axes.ravel()

    for i, park_id in enumerate(common_parks):
        ax = axes_flat[i]
        
        # Get the actual column names for this park
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]
        
        # Get measured and reference data for this park
        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()
        
        # Match by calendar day (month-day) instead of exact date
        # Extract month-day from both indices
        if len(measured) > 0 and len(reference) > 0:
            # Create month-day tuples
            measured_mds = pd.Series(
                [(d.month, d.day) for d in measured.index],
                index=measured.index
            )
            reference_mds = pd.Series(
                [(d.month, d.day) for d in reference.index],
                index=reference.index
            )
            
            # Find common month-days and align data
            measured_list = []
            reference_list = []
            
            for md in reference_mds.unique():
                meas_mask = measured_mds == md
                if meas_mask.any():
                    # Get all measured values for this calendar day
                    meas_vals = measured[meas_mask]
                    # Get the single reference value for this calendar day
                    ref_val = reference[reference_mds == md].iloc[0]
                    
                    measured_list.append(meas_vals.values)
                    reference_list.append(np.full_like(meas_vals.values, ref_val))
            
            if measured_list and reference_list:
                measured_vals = np.concatenate(measured_list)
                reference_vals = np.concatenate(reference_list)
            else:
                measured_vals = np.array([])
                reference_vals = np.array([])
        else:
            measured_vals = np.array([])
            reference_vals = np.array([])
        
        # Create scatter plot
        if len(measured_vals) > 0:
            ax.scatter(reference_vals, measured_vals, alpha=alpha, s=15, color='steelblue')
            
            # Add diagonal line (perfect match)
            max_val = max(reference_vals.max(), measured_vals.max())
            min_val = min(reference_vals.min(), measured_vals.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                   label='Perfect match', alpha=0.7)
            
            # Calculate R² and correlation
            if len(reference_vals) > 1:
                correlation = np.corrcoef(reference_vals, measured_vals)[0, 1]
                slope, intercept = np.polyfit(reference_vals, measured_vals, 1)
                r_squared = correlation ** 2
                ax.plot(reference_vals, slope * reference_vals + intercept, 'g-', 
                       linewidth=1.5, label=f'Linear fit (R²={r_squared:.3f})', alpha=0.7)
        
        # Format park label
        try:
            label = extract_park_name_before_pcc(measured_col)
        except Exception:
            label = str(park_id)
        
        ax.set_title(f"{title_prefix}: {label}")
        ax.grid(alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='upper left')

    # Hide any unused axes
    for j in range(len(common_parks), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"

    if base_filename is None:
        base_filename = title_prefix.lower().replace(" ", "_")

    saved_path = save_figure(
        fig=fig,
        title_prefix=title_prefix,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt=fmt,
        auto_version=auto_version,
        add_date=add_date,
    )

    # Generate findings
    findings = _generate_scatter_findings(measured_df, reference_df)
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)

    plt.show()
    return saved_path


def _auto_revenue_title(revenue_series: pd.Series) -> str:
    try:
        index = pd.Index(revenue_series.index)
        is_year_index = index.map(lambda v: str(v).isdigit() and len(str(v)) == 4).all()
    except Exception:
        is_year_index = False

    base = "Revenue by Year" if is_year_index else "Revenue over Time"

    attrs = getattr(revenue_series, "attrs", {}) or {}
    period_start = attrs.get("period_start") or attrs.get("mtd_start")
    period_end = attrs.get("period_end") or attrs.get("mtd_end")

    if period_start is not None and period_end is not None:
        start_ts = pd.to_datetime(period_start)
        end_ts = pd.to_datetime(period_end)
        period_label = f"{start_ts:%b %d} to {end_ts:%b %d}"
        return f"{base} - {period_label}"

    return base


def plot_revenue_by_year(
    revenue_series: pd.Series,
    title: str | None = None,
    price_per_kwh: float = 0.2,
    currency: str = "EUR",
    config=None,
    plot_name: str | None = None,
    save: bool = False,
    save_dir: Path | None = None,
    base_filename: str = "revenue_by_year",
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = True,
    add_date: bool = True,
):
    """
    Create an enhanced revenue chart showing revenue by year with styling.
    
    Parameters
    ----------
    revenue_series : pd.Series
        Series indexed by year with revenue values
    title : str | None
        Chart title. If None, a title is derived from the series name and index.
    price_per_kwh : float
        Price per kWh (for display only)
    currency : str
        Currency code
    save : bool
        Whether to save the figure
    save_dir : Path
        Directory for saving
    base_filename : str
        Base filename
    dpi : int
        Resolution
    fmt : str
        Image format
        
    Returns
    -------
    tuple of (Figure, Path or None)
        Figure object and path to saved file
    """
    if not title:
        title = _auto_revenue_title(revenue_series)

    attrs = getattr(revenue_series, "attrs", {}) or {}
    period_start = attrs.get("period_start") or attrs.get("mtd_start")
    period_end = attrs.get("period_end") or attrs.get("mtd_end")
    if period_start is not None and period_end is not None:
        start_ts = pd.to_datetime(period_start)
        end_ts = pd.to_datetime(period_end)
        print(f"Revenue by Year {start_ts:%b %d} - {end_ts:%b %d}")

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    # Calculate average for reference line
    avg_revenue = float(revenue_series.mean())
    
    # Color bars based on performance vs average
    colors = []
    for value in revenue_series.values:
        if value >= 1.10 * avg_revenue:
            colors.append('#27ae60')  # Dark green: >10% above avg
        elif value >= avg_revenue:
            colors.append('#2ecc71')  # Light green: above avg
        elif value >= 0.90 * avg_revenue:
            colors.append('#f39c12')  # Orange: slightly below avg
        else:
            colors.append('#e74c3c')  # Red: significantly below avg
    
    # Create bar chart
    bars = ax.bar(range(len(revenue_series)), revenue_series.values,
                   color=colors, alpha=0.85, edgecolor='#34495e', linewidth=1.5, width=0.6)
    
    # Add value labels on bars
    for i, (year, value) in enumerate(zip(revenue_series.index, revenue_series.values)):
        label_y = value + (max(revenue_series.values) * 0.02)
        ax.text(i, label_y, f'{value:,.0f}\n{currency}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3, edgecolor='none'))
    
    # Add average reference line
    ax.axhline(avg_revenue, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Average: {avg_revenue:,.0f} {currency}')
    
    # Styling
    ax.set_xticks(range(len(revenue_series)))
    ax.set_xticklabels(revenue_series.index.astype(str), fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Revenue [{currency}]', fontsize=12, fontweight='bold', color='#34495e')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_facecolor('#f8f9fa')
    ax.legend(fontsize=10, loc='upper left', frameon=True, shadow=True, fancybox=True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#34495e')
    ax.spines['bottom'].set_color('#34495e')
    
    plt.tight_layout()
    
    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"
    
    if base_filename == "revenue_by_year" and plot_name:
        base_filename = plot_name
    
    saved_path = save_figure(
        fig=fig,
        title_prefix=title,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt=fmt,
        auto_version=auto_version,
        add_date=add_date,
    )
    
    plt.show()
    return fig, saved_path


def plot_annual_production_and_revenue(
    production_series: pd.Series,
    revenue_series: pd.Series,
    title: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    production_unit: str = "kWh",
    currency: str = "EUR",
    config=None,
    plot_name: str | None = None,
    save: bool = False,
    save_dir: Path | None = None,
    base_filename: str = "annual_production_revenue",
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = True,
    add_date: bool = True,
):
    """
    Create a combined annual bar chart for total production and revenue.

    Parameters
    ----------
    production_series : pd.Series
        Series indexed by year with production values
    revenue_series : pd.Series
        Series indexed by year with revenue values
    title : str | None
        Chart title. If None, title is auto-generated.
    start_year : int | None
        Optional first year (inclusive) to include in chart.
    end_year : int | None
        Optional last year (inclusive) to include in chart.
    production_unit : str
        Unit label for production axis
    currency : str
        Currency code
    save : bool
        Whether to save the figure
    save_dir : Path
        Directory for saving
    base_filename : str
        Base filename
    dpi : int
        Resolution
    fmt : str
        Image format

    Returns
    -------
    tuple of (Figure, Path or None)
        Figure object and path to saved file
    """
    if production_series is None or revenue_series is None:
        raise ValueError("production_series and revenue_series must not be None")

    combined_df = pd.concat(
        [production_series.rename("production"), revenue_series.rename("revenue")],
        axis=1,
        join="inner",
    ).dropna()

    # Derived business metric requested for annual chart comparison.
    combined_df["ebidta"] = combined_df["revenue"] * 0.70

    if combined_df.empty:
        raise ValueError("No overlapping yearly data between production_series and revenue_series")

    try:
        combined_df = combined_df.sort_index()
    except Exception:
        pass

    if start_year is not None or end_year is not None:
        year_index = pd.to_numeric(pd.Index(combined_df.index), errors="coerce")
        year_mask = pd.Series(True, index=combined_df.index)

        if start_year is not None:
            year_mask &= year_index >= int(start_year)
        if end_year is not None:
            year_mask &= year_index <= int(end_year)

        combined_df = combined_df.loc[year_mask.values]

        if combined_df.empty:
            raise ValueError(
                f"No yearly data available in requested range: {start_year} to {end_year}"
            )

    if not title:
        attrs = getattr(production_series, "attrs", {}) or {}
        period_start = attrs.get("period_start") or attrs.get("mtd_start")
        period_end = attrs.get("period_end") or attrs.get("mtd_end")
        if period_start is not None and period_end is not None:
            start_ts = pd.to_datetime(period_start)
            end_ts = pd.to_datetime(period_end)
            title = f"Annual Production & Revenue - {start_ts:%b %d} to {end_ts:%b %d}"
        else:
            title = "Annual Production & Revenue"

    fig, ax_prod = plt.subplots(figsize=(13, 7), facecolor='white')
    ax_rev = ax_prod.twinx()

    years = combined_df.index.astype(str)
    x = np.arange(len(combined_df))
    width = 0.26

    prod_color = '#3498db'
    rev_color = '#2ecc71'
    ebidta_color = '#16a085'

    bars_prod = ax_prod.bar(
        x - width,
        combined_df["production"].values,
        width=width,
        color=prod_color,
        alpha=0.85,
        edgecolor='#34495e',
        linewidth=1.2,
        label=f'Production [{production_unit}]',
    )
    bars_rev = ax_rev.bar(
        x,
        combined_df["revenue"].values,
        width=width,
        color=rev_color,
        alpha=0.85,
        edgecolor='#34495e',
        linewidth=1.2,
        label=f'Revenue [{currency}]',
    )
    bars_ebidta = ax_rev.bar(
        x + width,
        combined_df["ebidta"].values,
        width=width,
        color=ebidta_color,
        alpha=0.9,
        edgecolor='#34495e',
        linewidth=1.2,
        label=f'ebidta [{currency}]',
    )

    max_prod = float(combined_df["production"].max()) if len(combined_df) else 0.0
    max_rev = float(combined_df["revenue"].max()) if len(combined_df) else 0.0
    prod_offset = 0.02 * max_prod if max_prod > 0 else 0.0
    rev_inside_offset = 0.04 * max_rev if max_rev > 0 else 0.0

    for rect, value in zip(bars_prod, combined_df["production"].values):
        ax_prod.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + prod_offset,
            f"{value:,.0f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

    for rect, value in zip(bars_rev, combined_df["revenue"].values):
        label_y = rect.get_height() - rev_inside_offset
        if label_y <= 0:
            label_y = rect.get_height() * 0.5
        ax_rev.text(
            rect.get_x() + rect.get_width() / 2,
            label_y,
            f"{value:,.0f}",
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#27ae60', alpha=0.6, edgecolor='none'),
        )

    for rect, value in zip(bars_ebidta, combined_df["ebidta"].values):
        label_y = rect.get_height() - rev_inside_offset
        if label_y <= 0:
            label_y = rect.get_height() * 0.5
        ax_rev.text(
            rect.get_x() + rect.get_width() / 2,
            label_y,
            f"{value:,.0f}",
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#138d75', alpha=0.6, edgecolor='none'),
        )

    ax_prod.set_xticks(x)
    ax_prod.set_xticklabels(years, fontsize=11, fontweight='bold')
    ax_prod.set_ylabel(f'Production [{production_unit}]', fontsize=12, fontweight='bold', color='#34495e')
    ax_rev.set_ylabel(f'Revenue [{currency}]', fontsize=12, fontweight='bold', color='#34495e')
    ax_prod.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

    ax_prod.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax_prod.set_facecolor('#f8f9fa')

    handles_1, labels_1 = ax_prod.get_legend_handles_labels()
    handles_2, labels_2 = ax_rev.get_legend_handles_labels()
    ax_prod.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        fontsize=10,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        shadow=True,
        fancybox=True,
    )

    ax_prod.spines['top'].set_visible(False)
    ax_prod.spines['left'].set_color('#34495e')
    ax_prod.spines['bottom'].set_color('#34495e')
    ax_rev.spines['top'].set_visible(False)
    ax_rev.spines['right'].set_color('#34495e')

    plt.tight_layout()

    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"

    if base_filename == "annual_production_revenue" and plot_name:
        base_filename = plot_name

    saved_path = save_figure(
        fig=fig,
        title_prefix=title,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt=fmt,
        auto_version=auto_version,
        add_date=add_date,
    )

    plt.show()
    return fig, saved_path


def plot_mtd_revenue_by_year_grid(
    daily_historical_df: pd.DataFrame,
    price_per_kwh: float | pd.Series | dict | None = 0.2,
    currency: str = "EUR",
    current_date: pd.Timestamp | None = None,
    power_mapping_df: pd.DataFrame | None = None,
    power_mapping_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    ncols: int = 3,
    save: bool = True,
    save_dir: Path | None = None,
    base_filename: str = "mtd_revenue_by_year_grid",
    dpi: int = 180,
    fmt: str = "png",
) -> Path | None:
    """
    Grid of revenue-by-year charts, one per park, showing month-to-date revenue per kWp.

    For each column (park), computes month-to-date energy per year via
    `annual_mtd_energy(..., agg='sum', per_park=True)`, converts to revenue,
    and renders a bar chart with average reference line and value annotations.
    
    Revenue is normalized per kWp: (energy_kwh * price_per_kwh) / power_kwp
    
    **Per-park pricing**: Supports individual prices per park from metadata.
    
    Parameters
    ----------
    daily_historical_df : pd.DataFrame
        Date-indexed DataFrame with parks as columns
    price_per_kwh : float, pd.Series, dict, or None
        Price per kWh. Can be:
        - float: Single price for all parks (default: 0.2)
        - pd.Series: Indexed by park_id with per-park prices
        - dict: Mapping park_id -> price
        - None: Auto-load from metadata_path
    currency : str
        Currency code
    metadata_path : Path or str, optional
        Path to park_metadata.csv for loading per-park prices when price_per_kwh=None
    current_date : pd.Timestamp
        Reference date for MTD calculation
    power_mapping_df : pd.DataFrame, optional
        Park metadata (not used, capacity extracted from column names)
    power_mapping_path : Path, optional
        Path to park metadata (not used)
    ncols : int
        Grid columns
    save : bool
        Whether to save
    save_dir : Path
        Save directory
    base_filename : str
        Base filename
    dpi : int
        Resolution
    fmt : str
        Image format
        
    Returns
    -------
    Path or None
        Path to saved file
    """
    import re as _re

    # Current date normalization
    if current_date is None:
        current_date = pd.Timestamp.now()
    else:
        current_date = pd.Timestamp(current_date)

    # Build a dictionary of park -> power_kwp using metadata
    # Column format: (park_id, signal, unit) - already cleaned by silver
    power_kwp_dict = {}
    
    # Load capacity from metadata
    park_capacity_map = {}
    if metadata_path is not None:
        try:
            meta_df = pd.read_csv(metadata_path)
            if 'park_id' in meta_df.columns and 'capacity_kwp' in meta_df.columns:
                # Normalize park_id to lowercase for matching
                meta_df['park_id_normalized'] = meta_df['park_id'].astype(str).str.strip().str.lower()
                park_capacity_map = dict(zip(meta_df['park_id_normalized'], meta_df['capacity_kwp']))
        except Exception as e:
            print(f"⚠️  Warning: Could not load capacity from metadata: {e}")
    
    # Map each column to its capacity
    for col in daily_historical_df.columns:
        # Column is a tuple like ('park_id', 'signal_name', 'unit')
        # Extract park_id (first element of tuple)
        if isinstance(col, tuple):
            park_id = str(col[0]).strip().lower()
            park_full = str(col[0])
        else:
            # Fallback: if column is string, try to extract park_id
            park_full = str(col)
            park_id = park_full.split('__')[0].strip().lower() if '__' in park_full else park_full.strip().lower()
        
        # Try metadata first (preferred method - uses authoritative capacity_kwp)
        if park_id in park_capacity_map:
            power_kwp_dict[col] = float(park_capacity_map[park_id])
        else:
            # Fallback: Extract kWp from pattern like "_XXXkwp" or "_XXXX_kwp"
            m = _re.search(r'_(\d+)_?kwp', park_full, _re.IGNORECASE)
            if m:
                try:
                    kwp = float(m.group(1))
                    power_kwp_dict[col] = kwp
                except ValueError:
                    power_kwp_dict[col] = 100.0
            else:
                # Default fallback (should rarely happen with metadata-based approach)
                power_kwp_dict[col] = 100.0

    # Ensure save_dir
    if save_dir is None:
        from pathlib import Path
        save_dir = Path("plots") / "financial_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

    # Columns (parks)
    parks = list(daily_historical_df.columns)
    nparks = len(parks)
    nrows = int(np.ceil(nparks / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.2 * nrows),
        constrained_layout=True,
        facecolor="white",
    )

    axes_list = axes.flatten() if hasattr(axes, "flatten") else np.ravel(axes)

    # Helper for short label
    def _short_label(col: str) -> str:
        try:
            return short_label(col)
        except Exception:
            m = _re.search(r"\[(.*?)\]", str(col))
            return m.group(1) if m else str(col)

    # Pre-compute MTD revenue per park/year using annual_mtd_revenue
    from src.metrics_calculator import annual_mtd_revenue
    
    # Prepare metadata_path for annual_mtd_revenue
    if metadata_path is None and price_per_kwh is None:
        raise ValueError("Either metadata_path or price_per_kwh must be provided")
    
    if metadata_path is not None:
        # Use annual_mtd_revenue with metadata path
        mtd_revenue_df = annual_mtd_revenue(
            daily_historical_df,
            metadata_path=metadata_path,
            agg="sum",
            aggregate_parks=False,
            current_date=current_date,
        )
    else:
        # Fallback: compute manually if only price_per_kwh is given
        from src.metrics_calculator import annual_mtd_energy
        mtd_energy_df = annual_mtd_energy(
            daily_historical_df,
            agg="sum",
            per_park=True,
            current_date=current_date,
        )
        
        # Convert energy to revenue
        if isinstance(price_per_kwh, dict):
            price_series = pd.Series(price_per_kwh)
        elif isinstance(price_per_kwh, pd.Series):
            price_series = price_per_kwh
        elif isinstance(price_per_kwh, (int, float)):
            price_series = pd.Series({park: price_per_kwh for park in parks})
        else:
            raise TypeError(f"Unsupported price_per_kwh type: {type(price_per_kwh)}")
        
        # Multiply energy by price for each park
        mtd_revenue_df = mtd_energy_df.copy()
        for park in mtd_energy_df.columns:
            # Extract park_id from column
            if isinstance(park, tuple):
                park_id = str(park[0]).split('__')[0]
            else:
                park_id = str(park).split('__')[0]
            
            # Get price, with fallback to default
            if park_id in price_series.index:
                park_price = float(price_series.loc[park_id])
            elif park in price_series.index:
                park_price = float(price_series.loc[park])
            else:
                park_price = 0.2
            mtd_revenue_df[park] = mtd_energy_df[park] * park_price

    # Build each subplot
    for idx, park in enumerate(parks):
        ax = axes_list[idx]
        # Month-to-date revenue per year for this park
        if isinstance(mtd_revenue_df, pd.DataFrame) and park in mtd_revenue_df.columns:
            mtd_revenue_raw = mtd_revenue_df[park].dropna()
        else:
            mtd_revenue_raw = pd.Series(dtype=float)

        # Normalize by power_kwp: revenue per kWp
        power_kwp = power_kwp_dict.get(park, 100.0)
        mtd_revenue = mtd_revenue_raw / power_kwp

        # Compute average and colors
        avg_val = float(mtd_revenue.mean()) if len(mtd_revenue) else 0.0
        colors = []
        for v in mtd_revenue.values:
            if v >= 1.10 * avg_val:
                colors.append('#27ae60')
            elif v >= avg_val:
                colors.append('#2ecc71')
            elif v >= 0.90 * avg_val:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        # Bar chart
        bars = ax.bar(range(len(mtd_revenue)), mtd_revenue.values,
                      color=colors, alpha=0.85, edgecolor='#34495e', linewidth=1.5, width=0.6)

        # Value labels
        if len(mtd_revenue):
            ymax = float(max(mtd_revenue.values))
        else:
            ymax = 0.0
        for i, (year, value) in enumerate(zip(mtd_revenue.index, mtd_revenue.values)):
            label_y = value + (ymax * 0.02)
            ax.text(i, label_y, f"{value:,.0f}\n{currency}/kWp", ha='center', va='bottom', fontsize=9,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3, edgecolor='none'))

        # Average line
        if len(mtd_revenue):
            ax.axhline(avg_val, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Average: {avg_val:,.0f} {currency}/kWp')

        # Axis styling
        ax.set_xticks(range(len(mtd_revenue)))
        ax.set_xticklabels(mtd_revenue.index.astype(str), fontsize=10, fontweight='bold', rotation=45)
        ax.set_ylabel(f'Revenue per kWp [{currency}/kWp]', fontsize=10, fontweight='bold', color='#34495e')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_facecolor('#f8f9fa')
        ax.legend(fontsize=9, loc='upper left', frameon=True, shadow=True, fancybox=True)

        # Title per park
        park_label = _short_label(park)
        ax.set_title(f"{park_label}\nMonth-to-Date Revenue per kWp", fontsize=11, fontweight='bold', color='#2c3e50')

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#34495e')
        ax.spines['bottom'].set_color('#34495e')

    # Hide unused axes
    for j in range(nparks, len(axes_list)):
        axes_list[j].axis('off')

    month_name = current_date.strftime('%B %Y')
    plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.3)
    
    fig.suptitle(f"Month-to-Date Revenue per kWp by Year — All Parks ({month_name})", fontsize=14, fontweight='bold', y=1.01)

    if save_dir is None:
        save_dir = Path("plots") / "weekly_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_path = save_figure(fig, title_prefix="MTD Revenue per kWp by Year Grid", save=save, save_dir=save_dir,
                             base_filename=base_filename, dpi=dpi, fmt=fmt, auto_version=True, add_date=True)
    plt.show()
    plt.close(fig)
    return saved_path
