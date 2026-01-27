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
from src.metrics_calculator import analyze_month_to_date_by_year


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

    plt.show()
    return saved_path


def plot_revenue_by_year(
    revenue_series: pd.Series,
    title: str = "Revenue by Year",
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
    title : str
        Chart title
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


def plot_mtd_revenue_by_year_grid(
    daily_historical_df: pd.DataFrame,
    price_per_kwh: float = 0.2,
    currency: str = "EUR",
    current_date: pd.Timestamp | None = None,
    power_mapping_df: pd.DataFrame | None = None,
    power_mapping_path: Path | str | None = None,
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
    `analyze_month_to_date_by_year(..., aggregation='sum')`, converts to revenue,
    and renders a bar chart with average reference line and value annotations.
    
    Revenue is normalized per kWp: (energy_kwh * price_per_kwh) / power_kwp
    
    Parameters
    ----------
    daily_historical_df : pd.DataFrame
        Date-indexed DataFrame with parks as columns
    price_per_kwh : float
        Price per kWh
    currency : str
        Currency code
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

    # Build a dictionary of park -> power_kwp by extracting kWp from column tuple
    # Column format: (park_name__XXXX_kwp, signal, unit)
    power_kwp_dict = {}
    for col in daily_historical_df.columns:
        # col is a tuple like ('fragiatoula_utilitas__4866_kwp', 'pcc_active_energy_export', 'kwh')
        park_full = str(col[0]) if isinstance(col, tuple) else str(col)
        
        # Extract kWp from pattern like "park_name__XXXX_kwp"
        m = _re.search(r'__(\d+)_kwp', park_full)
        if m:
            try:
                kwp = float(m.group(1))
                power_kwp_dict[col] = kwp
            except ValueError:
                print(f"⚠️  Warning: Could not parse kWp from {col}, defaulting to 100 kWp")
                power_kwp_dict[col] = 100.0
        else:
            print(f"⚠️  Warning: Could not find kWp pattern in {col}, defaulting to 100 kWp")
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

    # Build each subplot
    for idx, park in enumerate(parks):
        ax = axes_list[idx]
        # Month-to-date energy per year
        mtd_energy = analyze_month_to_date_by_year(
            daily_historical_df,
            park,
            aggregation='sum',
            current_date=current_date,
        )
        # Convert to revenue: (energy * price) / power_kwp
        power_kwp = power_kwp_dict.get(park, 100.0)
        mtd_revenue = (mtd_energy * price_per_kwh) / power_kwp

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

    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"
    
    if base_filename == "revenue_mtd_grid" and plot_name:
        base_filename = plot_name

    saved_path = save_figure(fig, title_prefix="MTD Revenue per kWp by Year Grid", save=save, save_dir=save_dir,
                             base_filename=base_filename, dpi=dpi, fmt=fmt, auto_version=auto_version, add_date=add_date)
    plt.show()
    plt.close(fig)
    return saved_path
