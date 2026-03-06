"""
Revenue plotting utilities for park financial analysis.

This module provides simplified plotting functions that work with pre-calculated
revenue DataFrames, avoiding redundant calculations.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.pvgis_pi_heatmap import short_label
from src.utils import save_figure


def plot_revenue_per_park_grid(
    revenue_per_park: pd.DataFrame,
    metadata_path: Path | str,
    currency: str = "EUR",
    normalize_per_kwp: bool = True,
    ncols: int = 3,
    save: bool = True,
    save_dir: Path | None = None,
    base_filename: str = "revenue_per_park_grid",
    dpi: int = 180,
    fmt: str = "png",
) -> Path | None:
    """
    Plot grid of revenue-by-year charts using pre-calculated revenue DataFrame.
    
    Optionally normalizes revenue per kWp: revenue / capacity_kwp
    
    This function accepts pre-calculated revenue data (e.g., from annual_mtd_revenue)
    and plots it, avoiding redundant calculations.
    
    Parameters
    ----------
    revenue_per_park : pd.DataFrame
        DataFrame with years as index and parks as columns (already in revenue units)
    metadata_path : Path or str
        Path to park_metadata.csv for loading capacity_kwp
    currency : str
        Currency code (default: "EUR")
    normalize_per_kwp : bool
        If True, normalize revenue by capacity (EUR/kWp). If False, show absolute revenue (EUR)
    ncols : int
        Number of columns in grid
    save : bool
        Whether to save the figure
    save_dir : Path, optional
        Save directory
    base_filename : str
        Base filename for saved figure
    dpi : int
        Resolution
    fmt : str
        Image format
        
    Returns
    -------
    Path or None
        Path to saved file if saved, else None
    """
    # Load capacity from metadata
    park_capacity_map = {}
    try:
        meta_df = pd.read_csv(metadata_path)
        if 'park_id' in meta_df.columns and 'capacity_kwp' in meta_df.columns:
            # Normalize park_id to lowercase for matching
            meta_df['park_id_normalized'] = meta_df['park_id'].astype(str).str.strip().str.lower()
            park_capacity_map = dict(zip(meta_df['park_id_normalized'], meta_df['capacity_kwp']))
    except Exception as e:
        print(f"⚠️ Warning: Could not load capacity from metadata: {e}")
        return None
    
    # Build capacity mapping for each column
    power_kwp_dict = {}
    for col in revenue_per_park.columns:
        # Extract park_id from column (handle tuple or string format)
        if isinstance(col, tuple):
            park_id = str(col[0]).strip().lower()
        else:
            park_full = str(col)
            park_id = park_full.split('__')[0].strip().lower() if '__' in park_full else park_full.strip().lower()
        
        # Get capacity from metadata
        if park_id in park_capacity_map:
            power_kwp_dict[col] = float(park_capacity_map[park_id])
        else:
            # Fallback: extract from column name pattern
            m = re.search(r'_(\d+)_?kwp', str(col), re.IGNORECASE)
            if m:
                power_kwp_dict[col] = float(m.group(1))
            else:
                power_kwp_dict[col] = 100.0  # Default fallback
    
    # Setup grid
    parks = list(revenue_per_park.columns)
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
            m = re.search(r"\[(.*?)\]", str(col))
            return m.group(1) if m else str(col)
    
    attrs = getattr(revenue_per_park, "attrs", {}) or {}
    period_start = attrs.get("period_start") or attrs.get("mtd_start")
    period_end = attrs.get("period_end") or attrs.get("mtd_end")
    period_label = ""
    if period_start is not None and period_end is not None:
        start_ts = pd.to_datetime(period_start)
        end_ts = pd.to_datetime(period_end)
        period_label = f" ({start_ts:%b %d} to {end_ts:%b %d})"
    
    # Plot each park
    for idx, park in enumerate(parks):
        ax = axes_list[idx]
        
        # Get revenue series for this park
        revenue_series = revenue_per_park[park]
        
        # Get capacity
        capacity = power_kwp_dict.get(park, 100.0)
        
        # Optionally normalize by capacity
        if normalize_per_kwp:
            plot_values = revenue_series / capacity
            ylabel = f'{currency}/kWp'
            title_suffix = f"({capacity:.0f} kWp)"
        else:
            plot_values = revenue_series
            ylabel = f'{currency}'
            title_suffix = f"({capacity:.0f} kWp)"
        
        # Calculate average
        avg_value = float(plot_values.mean())
        
        # Color bars based on performance
        colors = []
        for value in plot_values.values:
            if value >= 1.10 * avg_value:
                colors.append('#27ae60')  # Dark green
            elif value >= avg_value:
                colors.append('#2ecc71')  # Light green
            elif value >= 0.90 * avg_value:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        # Create bar chart
        years = plot_values.index
        bars = ax.bar(range(len(years)), plot_values.values,
                      color=colors, alpha=0.85, edgecolor='#34495e', linewidth=1.2, width=0.65)
        
        # Add value labels
        for i, (year, value) in enumerate(zip(years, plot_values.values)):
            label_y = value + (max(plot_values.values) * 0.02) if max(plot_values.values) > 0 else 0.1
            ax.text(i, label_y, f'{value:,.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3, edgecolor='none'))
        
        # Add average line
        ax.axhline(avg_value, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Styling
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years.astype(str), fontsize=9, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color='#34495e')
        ax.set_title(f"{_short_label(park)} {title_suffix}", fontsize=10, fontweight='bold', color='#2c3e50')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_facecolor('#f8f9fa')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#34495e')
        ax.spines['bottom'].set_color('#34495e')
    
    # Hide unused subplots
    for idx in range(nparks, len(axes_list)):
        axes_list[idx].set_visible(False)
    
    # Overall title (dynamic based on normalization)
    if normalize_per_kwp:
        title_text = f"Month-to-Date Revenue per kWp by Year - All Parks{period_label}"
    else:
        title_text = f"Month-to-Date Total Revenue by Year - All Parks{period_label}"
    
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=1.01)
    
    # Save
    if save_dir is None:
        save_dir = Path("plots") / "weekly_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_path = save_figure(
        fig, 
        title_prefix="MTD Revenue per kWp by Year Grid", 
        save=save, 
        save_dir=save_dir,
        base_filename=base_filename, 
        dpi=dpi, 
        fmt=fmt, 
        auto_version=True, 
        add_date=True
    )
    
    plt.show()
    plt.close(fig)
    return saved_path
