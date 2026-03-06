"""
Advanced Scatter Plot Visualizations for PV Analysis

Provides industry-standard visualization methods for comparing measured vs 
expected power generation, including hexbin plots, residual plots, and 
Bland-Altman plots.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.pvgis_pi_heatmap import short_label
from src.utils import save_figure


def extract_park_name_before_pcc(col):
    """Extract readable park name from column name."""
    try:
        return short_label(col)
    except Exception:
        return str(col)


def _align_measured_reference_by_calendar_day(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Align measured and reference values by park and calendar day.

    Returns flattened paired arrays (measured_vals, reference_vals).
    """
    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col

    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    common_parks = sorted(measured_parks & reference_parks)
    if len(common_parks) == 0:
        return np.array([]), np.array([])

    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}

    measured_all = []
    reference_all = []

    for park_id in common_parks:
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]

        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()

        if len(measured) == 0 or len(reference) == 0:
            continue

        measured_mds = pd.Series(
            [(d.month, d.day) for d in measured.index],
            index=measured.index,
        )
        reference_mds = pd.Series(
            [(d.month, d.day) for d in reference.index],
            index=reference.index,
        )

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


def _align_measured_reference_by_calendar_day_per_park(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Align measured and reference values by park and calendar day.

    Returns mapping {park_id: (measured_vals, reference_vals)}.
    """
    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col

    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    common_parks = sorted(measured_parks & reference_parks)
    if len(common_parks) == 0:
        return {}

    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}

    park_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for park_id in common_parks:
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]

        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()

        if len(measured) == 0 or len(reference) == 0:
            continue

        measured_mds = pd.Series(
            [(d.month, d.day) for d in measured.index],
            index=measured.index,
        )
        reference_mds = pd.Series(
            [(d.month, d.day) for d in reference.index],
            index=reference.index,
        )

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


def _most_problematic_parks(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    agreement_tol_pct: float = 10.0,
    top_n: int = 3,
) -> list[tuple[str, float, float]]:
    """Return the worst parks based on agreement percentage.

    Returns list of tuples: (park_label, agreement_pct, abs_bias_pct)
    """
    park_data = _align_measured_reference_by_calendar_day_per_park(measured_df, reference_df)
    worst = []

    for park_id, (m_vals, r_vals) in park_data.items():
        if len(m_vals) == 0:
            continue
        mean_vals = (m_vals + r_vals) / 2
        valid_mean = mean_vals > 0
        if not np.any(valid_mean):
            continue

        mean_vals = mean_vals[valid_mean]
        diff_vals = (m_vals[valid_mean] - r_vals[valid_mean]) / mean_vals * 100
        agreement_pct = 100 * np.mean(np.abs(diff_vals) <= agreement_tol_pct)
        bias_pct = float(np.mean(diff_vals))

        try:
            label = short_label(park_id)
        except Exception:
            label = str(park_id)

        worst.append((label, float(agreement_pct), abs(bias_pct)))

    worst.sort(key=lambda x: (x[1], -x[2]))
    return worst[:top_n]


def _generate_hexbin_findings(measured_df: pd.DataFrame, reference_df: pd.DataFrame) -> str:
    """Generate findings for hexbin density plots."""
    findings = []

    m_vals, r_vals = _align_measured_reference_by_calendar_day(measured_df, reference_df)
    
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
        findings.append(f"Density: {len(m_vals)} paired observations")
        
        # Correlation
        if len(m_vals) > 2:
            corr = np.corrcoef(m_vals, r_vals)[0, 1]
            if not np.isnan(corr):
                if corr > 0.9:
                    strength = "Very Strong"
                elif corr > 0.75:
                    strength = "Strong"
                elif corr > 0.5:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                findings.append(f"{strength} correlation: {corr:.3f} (predictability)")

        worst = _most_problematic_parks(measured_df, reference_df, agreement_tol_pct=10.0)
        if worst:
            findings.append("🔧 Most problematic parks (lowest agreement ±10%):")
            for label, agreement_pct, bias_abs in worst:
                findings.append(f"  - {label}: agreement={agreement_pct:.1f}%, |bias|={bias_abs:.1f}%")
    else:
        findings.append("⚠️  No paired observations after alignment")
        findings.append("   Check date coverage, timezone consistency, and park mapping")
    
    findings.append("💡 Interpretation: Darker hexagons = operating regime density")
    findings.append("Points near diagonal = PVGIS matches site; spread suggests variability from soiling, curtailment, shading, or sensor drift")
    
    return "\n".join(findings)


def _generate_residual_findings(measured_df: pd.DataFrame, reference_df: pd.DataFrame) -> str:
    """Generate findings for residual plots."""
    findings = []

    m_vals, r_vals = _align_measured_reference_by_calendar_day(measured_df, reference_df)
    
    if len(m_vals) > 0 and len(r_vals) > 0:
        residuals = m_vals - r_vals
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        bias = np.mean(residuals)
        
        findings.append(f"Bias (systematic error): {bias:+.2f} kWh/day")
        if abs(bias) < mae * 0.05:
            findings.append("  ✓ Minimal bias detected")
        elif bias > 0:
            findings.append("  ⚠️  System tends to OVERESTIMATE expectations")
        else:
            findings.append("  ⚠️  System tends to UNDERESTIMATE expectations")
        
        findings.append(f"Variability: RMSE={rmse:.2f}, MAE={mae:.2f}")
        
        # Outlier detection
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        outliers = np.sum(np.abs(residuals - np.median(residuals)) > outlier_threshold)
        if outliers > 0:
            pct = 100 * outliers / len(residuals)
            findings.append(f"⚠️  {outliers} outliers detected ({pct:.1f}%)")

        worst = _most_problematic_parks(measured_df, reference_df, agreement_tol_pct=10.0)
        if worst:
            findings.append("🔧 Most problematic parks (lowest agreement ±10%):")
            for label, agreement_pct, bias_abs in worst:
                findings.append(f"  - {label}: agreement={agreement_pct:.1f}%, |bias|={bias_abs:.1f}%")
    else:
        findings.append("⚠️  No paired observations after alignment")
        findings.append("   Check date coverage, timezone consistency, and park mapping")
    
    findings.append("💡 Interpretation: Residuals near 0 = expected performance")
    findings.append("Positive residuals = overperformance; negative residuals = underperformance (check availability, clipping, outages)")
    
    return "\n".join(findings)


def _generate_bland_altman_findings(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    diff_mode: str = "percent",
    agreement_tol_pct: float = 10.0,
) -> str:
    """Generate findings for Bland-Altman agreement plots."""
    findings = []

    m_vals, r_vals = _align_measured_reference_by_calendar_day(measured_df, reference_df)
    
    if len(m_vals) > 0 and len(r_vals) > 0:
        mean_vals = (m_vals + r_vals) / 2
        diff_vals = m_vals - r_vals

        if diff_mode == "percent":
            valid_mean = mean_vals > 0
            mean_vals = mean_vals[valid_mean]
            diff_vals = diff_vals[valid_mean]
            if len(diff_vals) == 0:
                findings.append("⚠️  Insufficient positive mean values for percent differences")
                return "\n".join(findings)
            diff_vals = (diff_vals / mean_vals) * 100
            unit = "%"
        else:
            unit = "kWh/day"

        bias = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        findings.append(f"Mean bias: {bias:+.2f} {unit} (avg difference)")
        findings.append(
            f"LoA (±1.96σ): {bias - 1.96*std_diff:+.2f} to {bias + 1.96*std_diff:+.2f} {unit}"
        )

        if abs(bias) < std_diff * 0.2:
            findings.append("✓ Good agreement: bias is negligible")
        else:
            findings.append("⚠️  Systematic difference detected")

        agreement_pct = 100 * np.mean(np.abs(diff_vals) <= agreement_tol_pct)
        findings.append(f"Agreement within ±{agreement_tol_pct:.0f}%: {agreement_pct:.1f}%")

        if len(diff_vals) > 2:
            slope, intercept, r_value, p_value, _ = stats.linregress(mean_vals, diff_vals)
            findings.append(
                f"Proportional bias: slope={slope:.4f}, r={r_value:.3f}, p={p_value:.3g}"
            )

        worst = _most_problematic_parks(
            measured_df,
            reference_df,
            agreement_tol_pct=agreement_tol_pct,
        )
        if worst:
            findings.append("🔧 Most problematic parks (lowest agreement ±{:.0f}%):".format(agreement_tol_pct))
            for label, agreement_pct, bias_abs in worst:
                findings.append(f"  - {label}: agreement={agreement_pct:.1f}%, |bias|={bias_abs:.1f}%")
            findings.append("📌 Engineer notes for worst parks:")

            per_park = _align_measured_reference_by_calendar_day_per_park(measured_df, reference_df)
            for label, _, _ in worst:
                park_id = label
                matched_key = None
                for key in per_park.keys():
                    try:
                        if short_label(key) == label:
                            matched_key = key
                            break
                    except Exception:
                        continue
                if matched_key is None:
                    continue

                m_vals, r_vals = per_park[matched_key]
                if len(m_vals) == 0:
                    continue

                mean_vals = (m_vals + r_vals) / 2
                valid_mean = mean_vals > 0
                if not np.any(valid_mean):
                    continue

                mean_vals = mean_vals[valid_mean]
                diff_vals = (m_vals[valid_mean] - r_vals[valid_mean]) / mean_vals * 100

                mean_diff = float(np.mean(diff_vals))
                std_diff = float(np.std(diff_vals))
                loa_upper = mean_diff + 1.96 * std_diff
                loa_lower = mean_diff - 1.96 * std_diff
                agreement_pct = 100 * np.mean(np.abs(diff_vals) <= agreement_tol_pct)

                if len(diff_vals) > 2:
                    slope, intercept, r_value, p_value, _ = stats.linregress(mean_vals, diff_vals)
                    prop_bias = f"slope={slope:.4f}, r={r_value:.3f}, p={p_value:.3g}"
                else:
                    prop_bias = "insufficient data"

                findings.append(
                    f"  - {label}: bias={mean_diff:+.2f}%, LoA=[{loa_lower:+.2f}%, {loa_upper:+.2f}%], "
                    f"agreement={agreement_pct:.1f}%, proportional bias: {prop_bias}"
                )
    else:
        findings.append("⚠️  No paired observations after alignment")
        findings.append("   Check date coverage, timezone consistency, and park mapping")

    findings.append(
        "💡 Interpretation: Bias near 0 and tight LoA indicate strong agreement with PVGIS"
    )
    findings.append(
        "A trend vs mean implies proportional bias (e.g., low-load or high-load losses, curtailment, or temperature effects)"
    )
    
    return "\n".join(findings)

def scatterplot_measured_vs_reference_hexbin(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    title_prefix: str = "Measured vs Reference (Density)",
    xlabel: str = "Reference [kWh/day]",
    ylabel: str = "Measured [kWh/day]",
    ncols: int = 3,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    cmap: str = "YlOrRd",
):
    """
    Create hexbin density plots comparing measured vs reference data.
    
    Hexbin plots are excellent for dense data, showing where most points
    cluster (darker = more data points).
    
    Parameters
    -----------
    measured_df: pd.DataFrame
        Date-indexed DataFrame with measured data
    reference_df: pd.DataFrame
        Date-indexed DataFrame with reference data
    title_prefix: str
        Prefix for subplot titles
    xlabel, ylabel: str
        Axis labels
    ncols: int
        Number of columns in subplot grid
    config: WorkspaceConfig, optional
        Workspace configuration for save directory
    save: bool
        Whether to save the figure
    save_dir: Path or str
        Directory for saving
    base_filename: str
        Base filename for saving
    dpi: int
        Resolution for saving
    cmap: str
        Colormap for hexbin (default: YlOrRd for warm colors)
    """
    
    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col
    
    # Get common parks
    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    common_parks = sorted(measured_parks & reference_parks)
    
    if len(common_parks) == 0:
        print("No common parks between measured and reference data.")
        return None
    
    # Build column mappings
    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}
    
    n = len(common_parks)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(6 * ncols, 24), max(3.5 * nrows, 4))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    
    axes_flat = axes.ravel()
    
    for i, park_id in enumerate(common_parks):
        ax = axes_flat[i]
        
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]
        
        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()
        
        # Match by calendar day
        if len(measured) > 0 and len(reference) > 0:
            measured_mds = pd.Series(
                [(d.month, d.day) for d in measured.index],
                index=measured.index
            )
            reference_mds = pd.Series(
                [(d.month, d.day) for d in reference.index],
                index=reference.index
            )
            
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
                measured_vals = np.concatenate(measured_list)
                reference_vals = np.concatenate(reference_list)
            else:
                measured_vals = np.array([])
                reference_vals = np.array([])
        else:
            measured_vals = np.array([])
            reference_vals = np.array([])
        
        # Create hexbin plot
        if len(measured_vals) > 0:
            hb = ax.hexbin(reference_vals, measured_vals, gridsize=20, cmap=cmap, mincnt=1)
            
            # Add perfect match line
            max_val = max(reference_vals.max(), measured_vals.max())
            min_val = min(reference_vals.min(), measured_vals.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect match', alpha=0.8)
            
            # Add colorbar for density
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Count', fontsize=8)
        
        try:
            label = short_label(measured_col)
        except Exception:
            label = str(park_id)
        
        ax.set_title(f"{title_prefix}: {label}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.2)
    
    # Hide unused axes
    for j in range(len(common_parks), len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"
    
    if base_filename is None:
        base_filename = "measured_vs_reference_hexbin"
    
    saved_path = save_figure(
        fig=fig,
        title_prefix="Measured vs Reference (Hexbin Density)",
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt="png",
        auto_version=True,
        add_date=True,
    )
    
    # Generate findings
    findings = _generate_hexbin_findings(measured_df, reference_df)
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)
    
    plt.show()
    return saved_path


def scatterplot_residuals(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    title_prefix: str = "Residual Analysis",
    ncols: int = 3,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
):
    """
    Create residual plots (measured - reference) vs reference.
    
    Residual plots are industry-standard for analyzing bias and variability.
    Shows how far actual performance deviates from expected.
    
    Parameters
    -----------
    measured_df: pd.DataFrame
        Date-indexed DataFrame with measured data
    reference_df: pd.DataFrame
        Date-indexed DataFrame with reference data
    title_prefix: str
        Prefix for subplot titles
    ncols: int
        Number of columns in subplot grid
    config: WorkspaceConfig, optional
        Workspace configuration
    save: bool
        Whether to save the figure
    save_dir: Path or str
        Directory for saving
    base_filename: str
        Base filename for saving
    dpi: int
        Resolution for saving
    """
    
    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col
    
    # Get common parks
    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    common_parks = sorted(measured_parks & reference_parks)
    
    if len(common_parks) == 0:
        print("No common parks between measured and reference data.")
        return None
    
    # Build column mappings
    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}
    
    n = len(common_parks)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(6 * ncols, 24), max(3.5 * nrows, 4))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    
    axes_flat = axes.ravel()
    
    for i, park_id in enumerate(common_parks):
        ax = axes_flat[i]
        
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]
        
        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()
        
        # Match by calendar day
        if len(measured) > 0 and len(reference) > 0:
            measured_mds = pd.Series(
                [(d.month, d.day) for d in measured.index],
                index=measured.index
            )
            reference_mds = pd.Series(
                [(d.month, d.day) for d in reference.index],
                index=reference.index
            )
            
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
                measured_vals = np.concatenate(measured_list)
                reference_vals = np.concatenate(reference_list)
            else:
                measured_vals = np.array([])
                reference_vals = np.array([])
        else:
            measured_vals = np.array([])
            reference_vals = np.array([])
        
        # Create residual plot
        if len(measured_vals) > 0:
            residuals = measured_vals - reference_vals
            
            # Scatter plot with colors based on residual magnitude
            scatter = ax.scatter(reference_vals, residuals, c=residuals, cmap='RdBu_r', 
                               alpha=0.5, s=15, edgecolors='none')
            
            # Add zero line (perfect match)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='No bias', alpha=0.8)
            
            # Add ±1 std dev bands
            std_residual = np.std(residuals)
            ax.axhline(y=std_residual, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=-std_residual, color='gray', linestyle=':', linewidth=1, alpha=0.5, 
                      label=f'±1 σ ({std_residual:.0f})')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Residual [kWh]', fontsize=8)
            
            # Calculate bias metrics
            mean_residual = np.mean(residuals)
            rmse = np.sqrt(np.mean(residuals**2))
            mape = np.mean(np.abs((measured_vals - reference_vals) / reference_vals)) * 100
            
            stats_text = f'Bias: {mean_residual:.0f}\nRMSE: {rmse:.0f}\nMAPE: {mape:.1f}%'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        try:
            label = short_label(measured_col)
        except Exception:
            label = str(park_id)
        
        ax.set_title(f"{title_prefix}: {label}", fontsize=10)
        ax.set_xlabel("Reference [kWh/day]", fontsize=9)
        ax.set_ylabel("Residual: Measured - Reference [kWh/day]", fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.2)
    
    # Hide unused axes
    for j in range(len(common_parks), len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"
    
    if base_filename is None:
        base_filename = "residual_analysis"
    
    saved_path = save_figure(
        fig=fig,
        title_prefix="Residual Analysis",
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt="png",
        auto_version=True,
        add_date=True,
    )
    
    # Generate findings
    findings = _generate_residual_findings(measured_df, reference_df)
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)
    
    plt.show()
    return saved_path


def scatterplot_bland_altman(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    title_prefix: str = "Bland-Altman Analysis",
    ncols: int = 3,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    diff_mode: str = "percent",
    agreement_tol_pct: float = 10.0,
):
    """
    Create Bland-Altman plots (industry standard for comparing measurement methods).
    
    X-axis: Average of measured and reference (mean performance)
    Y-axis: Difference (measured - reference, i.e., bias)
    
    Shows agreement between two methods with limits of agreement.
    
    Parameters
    -----------
    measured_df: pd.DataFrame
        Date-indexed DataFrame with measured data
    reference_df: pd.DataFrame
        Date-indexed DataFrame with reference data
    title_prefix: str
        Prefix for subplot titles
    ncols: int
        Number of columns in subplot grid
    config: WorkspaceConfig, optional
        Workspace configuration
    save: bool
        Whether to save the figure
    save_dir: Path or str
        Directory for saving
    base_filename: str
        Base filename for saving
    dpi: int
        Resolution for saving
    """
    
    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col
    
    # Get common parks
    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    common_parks = sorted(measured_parks & reference_parks)
    
    if len(common_parks) == 0:
        print("No common parks between measured and reference data.")
        return None
    
    # Build column mappings
    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}
    
    n = len(common_parks)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(6 * ncols, 24), max(3.5 * nrows, 4))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    
    axes_flat = axes.ravel()
    
    for i, park_id in enumerate(common_parks):
        ax = axes_flat[i]
        
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]
        
        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()
        
        # Match by calendar day
        if len(measured) > 0 and len(reference) > 0:
            measured_mds = pd.Series(
                [(d.month, d.day) for d in measured.index],
                index=measured.index
            )
            reference_mds = pd.Series(
                [(d.month, d.day) for d in reference.index],
                index=reference.index
            )
            
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
                measured_vals = np.concatenate(measured_list)
                reference_vals = np.concatenate(reference_list)
            else:
                measured_vals = np.array([])
                reference_vals = np.array([])
        else:
            measured_vals = np.array([])
            reference_vals = np.array([])
        
        # Create Bland-Altman plot
        if len(measured_vals) > 0:
            # Calculate mean and difference
            mean_vals = (measured_vals + reference_vals) / 2
            diff_vals = measured_vals - reference_vals

            if diff_mode == "percent":
                valid_mean = mean_vals > 0
                mean_vals = mean_vals[valid_mean]
                diff_vals = diff_vals[valid_mean]
                if len(diff_vals) == 0:
                    continue
                diff_vals = (diff_vals / mean_vals) * 100
                diff_unit = "%"
            else:
                diff_unit = "kWh/day"
            
            # Plot
            ax.scatter(mean_vals, diff_vals, alpha=0.5, s=15, color='steelblue', edgecolors='none')
            
            # Calculate mean difference and limits of agreement
            mean_diff = np.mean(diff_vals)
            std_diff = np.std(diff_vals)
            
            # Limits of agreement (±1.96 * SD)
            loa_upper = mean_diff + 1.96 * std_diff
            loa_lower = mean_diff - 1.96 * std_diff
            
            # Plot lines
            ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean diff: {mean_diff:.2f}{diff_unit}')
            ax.axhline(loa_upper, color='gray', linestyle='--', linewidth=1.5, 
                      label=f'±LoA: ±{1.96*std_diff:.2f}{diff_unit}')
            ax.axhline(loa_lower, color='gray', linestyle='--', linewidth=1.5)
            ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(agreement_tol_pct, color='green', linestyle=':', linewidth=1, alpha=0.7,
                      label=f'±{agreement_tol_pct:.0f}% agreement')
            ax.axhline(-agreement_tol_pct, color='green', linestyle=':', linewidth=1, alpha=0.7)
            
            # Stats box
            agreement_pct = 100 * np.mean(np.abs(diff_vals) <= agreement_tol_pct)
            stats_text = (
                f'Mean diff: {mean_diff:.2f}{diff_unit}\n'
                f'SD: {std_diff:.2f}{diff_unit}\n'
                f'LoA: [{loa_lower:.2f}, {loa_upper:.2f}]{diff_unit}\n'
                f'≤±{agreement_tol_pct:.0f}%: {agreement_pct:.1f}%'
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        try:
            label = short_label(measured_col)
        except Exception:
            label = str(park_id)
        
        ax.set_title(f"{title_prefix}: {label}", fontsize=10)
        ax.set_xlabel("Mean of Measured & Reference [kWh/day]", fontsize=9)
        if diff_mode == "percent":
            ax.set_ylabel("Difference: Measured - Reference [% of mean]", fontsize=9)
        else:
            ax.set_ylabel("Difference: Measured - Reference [kWh/day]", fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.2)
    
    # Hide unused axes
    for j in range(len(common_parks), len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"
    
    if base_filename is None:
        base_filename = "bland_altman_analysis"
    
    saved_path = save_figure(
        fig=fig,
        title_prefix="Bland-Altman Analysis",
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt="png",
        auto_version=True,
        add_date=True,
    )
    
    # Generate findings
    findings = _generate_bland_altman_findings(
        measured_df,
        reference_df,
        diff_mode=diff_mode,
        agreement_tol_pct=agreement_tol_pct,
    )
    print("\n" + "="*60)
    print("📋 FINDINGS & INTERPRETATION")
    print("="*60)
    print(findings)
    print("="*60)
    
    plt.show()
    return saved_path


def scatterplot_measured_vs_reference_joint(
    measured_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    title_prefix: str = "Measured vs Reference (Joint Plot)",
    xlabel: str = "Reference [kWh/day]",
    ylabel: str = "Measured [kWh/day]",
    ncols: int = 3,
    config=None,
    save: bool = False,
    save_dir: str | Path | None = None,
    base_filename: str | None = None,
    dpi: int = 150,
    kind: str = "hex",
    cmap: str = "YlOrRd",
    bins: int = 30,
):
    """
    Create grid of joint plots comparing measured vs reference data per park.

    Uses hexbin or scatter plots with marginal distributions to show density
    and distribution overlap for each park in a subplot grid.
    
    Parameters
    -----------
    measured_df: pd.DataFrame
        Date-indexed DataFrame with measured data
    reference_df: pd.DataFrame
        Date-indexed DataFrame with reference data
    title_prefix: str
        Prefix for subplot titles
    xlabel, ylabel: str
        Axis labels
    ncols: int
        Number of columns in subplot grid (default: 3)
    config: WorkspaceConfig, optional
        Workspace configuration for save directory
    save: bool
        Whether to save the figure
    save_dir: Path or str
        Directory for saving
    base_filename: str
        Base filename for saving
    dpi: int
        Resolution for saving
    kind: str
        Plot kind for hexbin ('hex', 'scatter', etc.)
    cmap: str
        Colormap for hexbin (default: YlOrRd for warm colors)
    bins: int
        Number of bins for hexbin
    """

    def get_column_key(col):
        if isinstance(col, tuple):
            return col[0]
        return col

    measured_parks = set(get_column_key(col) for col in measured_df.columns)
    reference_parks = set(get_column_key(col) for col in reference_df.columns)
    common_parks = sorted(measured_parks & reference_parks)

    if len(common_parks) == 0:
        print("No common parks between measured and reference data.")
        return None

    measured_cols = {get_column_key(col): col for col in measured_df.columns}
    reference_cols = {get_column_key(col): col for col in reference_df.columns}

    if save_dir is None and config is not None:
        save_dir = config.PLOTS_DIR / "weekly_analysis"

    if base_filename is None:
        base_filename = "measured_vs_reference_joint"

    # Build data for all parks
    park_data = []
    for park_id in common_parks:
        measured_col = measured_cols[park_id]
        reference_col = reference_cols[park_id]

        measured = measured_df[measured_col].dropna()
        reference = reference_df[reference_col].dropna()

        if len(measured) > 0 and len(reference) > 0:
            measured_mds = pd.Series(
                [(d.month, d.day) for d in measured.index],
                index=measured.index
            )
            reference_mds = pd.Series(
                [(d.month, d.day) for d in reference.index],
                index=reference.index
            )

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
                measured_vals = np.concatenate(measured_list)
                reference_vals = np.concatenate(reference_list)
            else:
                measured_vals = np.array([])
                reference_vals = np.array([])
        else:
            measured_vals = np.array([])
            reference_vals = np.array([])

        if len(measured_vals) > 0:
            try:
                label = short_label(measured_col)
            except Exception:
                label = str(park_id)
            
            park_data.append({
                'park_id': park_id,
                'label': label,
                'measured': measured_vals,
                'reference': reference_vals,
            })

    if len(park_data) == 0:
        print("No valid data available for any parks.")
        return None

    # Create grid
    n = len(park_data)
    nrows = int(np.ceil(n / ncols))
    figsize = (min(5 * ncols, 20), max(4 * nrows, 5))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    
    axes_flat = axes.flatten()

    for idx, data in enumerate(park_data):
        ax = axes_flat[idx]
        measured_vals = data['measured']
        reference_vals = data['reference']
        label = data['label']

        # Create scatter/hexbin plot on the axis
        if kind == "hex":
            hexbin = ax.hexbin(
                reference_vals, measured_vals,
                gridsize=bins, cmap=cmap, mincnt=1, alpha=0.8
            )
            fig.colorbar(hexbin, ax=ax, label="Count")
        else:
            ax.scatter(reference_vals, measured_vals, alpha=0.4, s=10, c=cmap)

        # Add 1:1 line
        max_val = max(reference_vals.max(), measured_vals.max())
        min_val = min(reference_vals.min(), measured_vals.min())
        ax.plot([min_val, max_val], [min_val, max_val], "b--", linewidth=1.5, alpha=0.8)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"{title_prefix}: {label}", fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(len(park_data), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = save_figure(
        fig=fig,
        title_prefix=title_prefix,
        save=save,
        save_dir=save_dir,
        base_filename=base_filename,
        dpi=dpi,
        fmt="png",
        auto_version=True,
        add_date=True,
    )

    if save_path is not None:
        plt.show()
        return save_path
    else:
        plt.show()
        return None
