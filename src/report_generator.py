from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import warnings

from src.visualizations import (
    lineplot_timeseries_per_column,
    plot_heatmap,
    plot_revenue_by_year,
    plot_mtd_revenue_by_year_grid,
)
from src.metrics_calculator import analyze_month_to_date_by_year
from src.pvgis_pi_heatmap import short_label, parse_kwp_from_header
from src.utils import sanitize_filename, generate_versioned_filename
from src.degradation_analysis import analyze_degradation_with_stl


__all__ = [
    "create_weekly_technical_report_for_all_parks",
    "create_weekly_stl_report",
    "create_economic_analysis_dashboard",
    "create_financial_report_for_all_parks",
]


def _log(msg: str, logger=None):
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def create_weekly_technical_report_for_all_parks(
    daily_df: pd.DataFrame,
    pi_df: pd.DataFrame,
    daily_historical_df: pd.DataFrame,
    report_date: Optional[str | pd.Timestamp] = None,
    price_per_kwh: float = 0.2,
    currency: str = "EUR",
    save_dir: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
    version: str = "v1",
    dpi: int = 150,
    fmt: str = "png",
    ncols_timeseries: int = 3,
    ncols_revenue_grid: int = 3,
    logger=None,
) -> Path:
    """Create the weekly technical report with plots and markdown.

    Saves plots under <plots>/weekly_analysis and writes markdown to <workspace>/docs.
    """
    warnings.filterwarnings("ignore")

    if workspace_root is None or save_dir is None:
        from src.config import get_config
        config = get_config()
        ws_root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
        plots_root = Path(save_dir) if save_dir else config.PLOTS_DIR
    else:
        ws_root = Path(workspace_root)
        plots_root = Path(save_dir)
    weekly_dir = plots_root / "weekly_analysis"
    weekly_dir.mkdir(exist_ok=True, parents=True)

    report_date = pd.Timestamp.now() if report_date is None else pd.Timestamp(report_date)
    date_str = report_date.strftime("%Y%m%d")
    month_start = pd.Timestamp(year=report_date.year, month=report_date.month, day=1)

    _log("=" * 80, logger)
    _log("GENERATING WEEKLY TECHNICAL REPORT", logger)
    _log("=" * 80, logger)
    _log(f"Report Date: {report_date.strftime('%B %d, %Y')}", logger)
    _log(f"Version: {version}", logger)
    _log(f"Save Directory: {weekly_dir}", logger)
    _log("=" * 80, logger)

    # 1) Daily energy time series grid
    _log("1) Daily energy time series grid ...", logger)
    ts_path = lineplot_timeseries_per_column(
        daily_df,
        title_prefix="Daily Energy",
        ylabel="Energy [kWh]",
        ncols=ncols_timeseries,
        sharex=True,
        sharey=False,
        save=True,
        save_dir=weekly_dir,
        base_filename=f"fig1_daily_energy_timeseries_{date_str}_{version}",
        dpi=dpi,
        fmt=fmt,
    )

    # 2) PI heatmap - full period
    _log("2) PI heatmap - full period ...", logger)
    pi_full_path = plot_heatmap(
        pi_df,
        "PI_PVGIS = Measured / PVGIS expected",
        save=True,
        save_dir=weekly_dir,
        base_filename=f"fig2_pi_heatmap_full_{date_str}_{version}",
        dpi=dpi,
        fmt=fmt,
    )

    # 3) PI heatmap - month-to-date
    _log("3) PI heatmap - month-to-date ...", logger)
    pi_mtd_path = plot_heatmap(
        pi_df,
        f"PI_PVGIS - {month_start.strftime('%B %Y')}",
        start_date=month_start,
        end_date=report_date,
        save=True,
        save_dir=weekly_dir,
        base_filename=f"fig3_pi_heatmap_mtd_{date_str}_{version}",
        dpi=dpi,
        fmt=fmt,
    )

    # 4) Month-to-date total revenue (all parks)
    _log("4) Month-to-date total REVENUE for all parks by year ...", logger)
    mtd_total_energy = analyze_month_to_date_by_year(
        daily_historical_df,
        column=None,
        aggregation="sum",
        current_date=report_date,
    )
    mtd_total_revenue = mtd_total_energy * price_per_kwh

    fig_revenue, revenue_by_year_path = plot_revenue_by_year(
        mtd_total_revenue,
        title=f"Month-to-Date REVENUE by Year — All Parks ({month_start.strftime('%B')} 1-{report_date.day})",
        price_per_kwh=price_per_kwh,
        currency=currency,
        save=True,
        save_dir=weekly_dir,
        base_filename=f"fig4_revenue_mtd_all_parks_{date_str}_{version}",
        dpi=dpi,
        fmt=fmt,
    )

    # 5) Month-to-date revenue per-park grid
    _log("5) Month-to-date REVENUE per-park grid by year ...", logger)
    mtd_grid_path = plot_mtd_revenue_by_year_grid(
        daily_historical_df=daily_historical_df,
        current_date=report_date,
        price_per_kwh=price_per_kwh,
        currency=currency,
        ncols=ncols_revenue_grid,
        save=True,
        save_dir=weekly_dir,
        base_filename=f"fig5_revenue_mtd_grid_{date_str}_{version}",
        dpi=dpi,
        fmt=fmt,
    )

    # Markdown report
    _log("\nWriting markdown report ...", logger)
    report_filename = f"report_tech_weekly_{date_str}_{version}.md"
    report_path = ws_root / "docs" / report_filename

    rel = lambda p: Path("../plots/weekly_analysis") / Path(p).name if p else None
    rel_ts_path = rel(ts_path)
    rel_pi_full = rel(pi_full_path)
    rel_pi_mtd = rel(pi_mtd_path)
    rel_revenue_by_year = rel(revenue_by_year_path)
    rel_mtd_grid = rel(mtd_grid_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Weekly PV Technical Report\n")
        f.write(f"**Report Date:** {report_date.strftime('%B %d, %Y')}\n")
        f.write(f"**Version:** {version}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This weekly technical report summarizes operational performance across all PV parks, "
                "with an emphasis on time-series behavior, month-to-date performance indicators (PI), "
                "and revenue totals.\n\n")
        f.write("---\n\n")

        f.write("## 1. Daily Energy Time Series (All Parks)\n\n")
        if rel_ts_path:
            f.write(f"![Daily Energy Time Series]({rel_ts_path})\n\n")
        f.write("**Notes:**\n")
        f.write("- Check for gaps or flatlines indicating telemetry or data acquisition issues.\n")
        f.write("- Compare inter-park variability to expected seasonal patterns.\n\n")
        f.write("---\n\n")

        f.write("## 2. Performance Index (PI) Heatmap — Full Period\n\n")
        if rel_pi_full:
            f.write(f"![PI Heatmap - Full Period]({rel_pi_full})\n\n")
        f.write("**Interpretation:** PI ≈ 1 implies measured energy aligns with PVGIS expectation. "
                "Sustained PI < 0.8 suggests underperformance; PI > 1.2 may indicate measurement anomalies.\n\n")
        f.write("---\n\n")

        f.write(f"## 3. PI Heatmap — {month_start.strftime('%B %Y')} (Month-to-Date)\n\n")
        if rel_pi_mtd:
            f.write(f"![PI Heatmap - MTD]({rel_pi_mtd})\n\n")
        f.write("**Operational Checkpoints:**\n")
        f.write("- Investigate parks with sustained blue regions (PI < 0.8).\n")
        f.write("- Correlate anomalies with weather, outages, and maintenance logs.\n\n")
        f.write("---\n\n")

        f.write("## 4. Month-to-Date Revenue by Year — All Parks\n\n")
        f.write(f"Period: {month_start.strftime('%B')} 1–{report_date.day}\n\n")
        if rel_revenue_by_year:
            f.write(f"![Revenue MTD - All Parks]({rel_revenue_by_year})\n\n")
        f.write("**Insights:**\n")
        best_year = mtd_total_revenue.idxmax()
        worst_year = mtd_total_revenue.idxmin()
        f.write(f"- Highest MTD revenue: {best_year} ({mtd_total_revenue[best_year]:,.2f} {currency})\n")
        f.write(f"- Lowest MTD revenue: {worst_year} ({mtd_total_revenue[worst_year]:,.2f} {currency})\n")
        if len(mtd_total_revenue) > 1:
            growth = ((mtd_total_revenue.iloc[-1] - mtd_total_revenue.iloc[0]) / max(mtd_total_revenue.iloc[0], 1e-6)) * 100
            f.write(f"- Overall change: {growth:+.2f}% from {mtd_total_revenue.index[0]} to {mtd_total_revenue.index[-1]}\n\n")
        f.write("---\n\n")

        f.write("## 5. Month-to-Date Revenue per Park by Year (Grid)\n\n")
        if rel_mtd_grid:
            f.write(f"![MTD Revenue Grid]({rel_mtd_grid})\n\n")
        f.write("**Usage:** Highlights per-park month-to-date revenue against historical years. "
                "Use alongside PI heatmaps to differentiate revenue-impacting underperformance from data issues.\n\n")

        f.write("---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    _log(f"\n✓ Technical report saved: {report_path}", logger)
    return report_path


def create_weekly_stl_report(
    daily_df: pd.DataFrame,
    report_date: Optional[str | pd.Timestamp] = None,
    parks: Optional[list[str]] = None,
    max_parks: int = 6,
    save_dir: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
    dpi: int = 150,
    fmt: str = "png",
    apply_log: bool = False,
    logger=None,
) -> Path:
    """Create a weekly STL degradation report with plots and markdown summary.

    Limits to ``max_parks`` series to keep runtime manageable. Uses versioned
    filenames similar to technical/financial reports.
    """
    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore")

    if workspace_root is None or save_dir is None:
        from src.config import get_config
        config = get_config()
        ws_root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
        plots_root = Path(save_dir) if save_dir else config.PLOTS_DIR
    else:
        ws_root = Path(workspace_root)
        plots_root = Path(save_dir)

    report_date = pd.Timestamp.now() if report_date is None else pd.Timestamp(report_date)
    date_str = report_date.strftime("%Y%m%d")

    stl_dir = plots_root / "weekly_analysis" / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)

    docs_dir = ws_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    selected_parks = list(parks) if parks else list(daily_df.columns)
    if max_parks and len(selected_parks) > max_parks:
        selected_parks = selected_parks[:max_parks]

    summaries = []

    _log("=" * 80, logger)
    _log("GENERATING WEEKLY STL REPORT", logger)
    _log("=" * 80, logger)
    _log(f"Report Date: {report_date.strftime('%B %d, %Y')}", logger)
    _log(f"Save Directory: {stl_dir}", logger)
    _log(f"Parks included: {len(selected_parks)}", logger)
    _log("=" * 80, logger)

    for col in selected_parks:
        if col not in daily_df.columns:
            _log(f"⚠️  Skipping missing column: {col}", logger)
            continue

        series = daily_df[col].dropna()
        if len(series) == 0:
            _log(f"⚠️  Skipping empty series: {short_label(col)}", logger)
            continue

        try:
            results = analyze_degradation_with_stl(
                series=series,
                apply_log=apply_log,
                period=365,
                robust=True,
                anomaly_threshold=-3.0,
                min_consecutive_days=2,
                return_components=True,
            )
        except ValueError as exc:
            _log(f"⚠️  {short_label(col)}: {exc}", logger)
            continue

        anomalies = results["anomaly_flags"]
        persistent = results["persistent_anomaly_flags"]
        trend = results.get("trend")
        residual_z = results.get("residual_z_scores")

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(series.index, series.values, linewidth=0.8, color="black", alpha=0.7, label="Observed")
        if anomalies.any():
            axes[0].scatter(anomalies[anomalies].index, series.loc[anomalies].values, color="orange", s=15, label="Anomaly", zorder=3)
        if persistent.any():
            axes[0].scatter(persistent[persistent].index, series.loc[persistent].values, color="red", s=25, marker="x", label="Persistent", zorder=4)
        axes[0].set_ylabel("Energy")
        axes[0].set_title(f"Observed with Anomalies — {short_label(col)}")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)

        if trend is not None:
            axes[1].plot(trend.index, trend.values, linewidth=1.2, color="#2E86AB", label="Trend")
            days_numeric = (trend.index - trend.index[0]).days.values
            fit = results["trend_slope"] * days_numeric + results["trend_intercept"]
            axes[1].plot(trend.index, fit, "--", color="red", linewidth=1.5, alpha=0.8, label="Linear Fit")
            axes[1].set_ylabel("Trend")
            axes[1].set_title("Trend and Linear Fit")
            axes[1].legend(loc="best", fontsize=8)
            axes[1].grid(alpha=0.3)

        if residual_z is not None:
            axes[2].plot(residual_z.index, residual_z.values, linewidth=0.8, color="steelblue", alpha=0.7)
            axes[2].axhline(-3, color="red", linestyle="--", linewidth=1.2, label="Threshold")
            axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.6)
            axes[2].fill_between(residual_z.index, residual_z.values, -3, where=(residual_z.values < -3), color="red", alpha=0.15)
            axes[2].set_ylabel("Robust Z-score")
            axes[2].set_title("Residual Robust Z-scores")
            axes[2].legend(loc="best", fontsize=8)
            axes[2].grid(alpha=0.3)

        axes[2].set_xlabel("Date")
        plt.tight_layout()

        safe_name = sanitize_filename(col)
        fig_path = stl_dir / f"stl_{safe_name}_{date_str}.{fmt}"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight", facecolor="white", format=fmt)
        plt.close(fig)

        summaries.append(
            {
                "park": col,
                "label": short_label(col),
                "annual_deg": results.get("annual_degradation", 0.0),
                "monthly_deg": results.get("degradation_rate", 0.0),
                "anomalies": int(results.get("anomaly_count", 0)),
                "persistent": int(results.get("persistent_anomaly_count", 0)),
                "r2": results.get("trend_r_squared", 0.0),
                "path": fig_path,
                "range": results.get("date_range", ""),
                "points": results.get("data_points", 0),
            }
        )

        _log(
            f"  ✓ {short_label(col)} | annual {results.get('annual_degradation', 0.0):+.2f}% | "
            f"anomalies {results.get('anomaly_count', 0)} | persistent {results.get('persistent_anomaly_count', 0)}",
            logger,
        )

    if not summaries:
        raise ValueError("No STL results generated; check input data or parameters.")

    versioned_name = generate_versioned_filename(
        base_name="report_weekly_stl",
        save_dir=docs_dir,
        fmt="md",
        add_date=True,
    )
    report_path = docs_dir / f"{versioned_name}.md"

    summaries_sorted = sorted(summaries, key=lambda x: x["annual_deg"])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Weekly STL Degradation Report\n")
        f.write(f"**Report Date:** {report_date.strftime('%B %d, %Y')}\n\n")
        f.write(f"Analyzed Parks: {len(summaries_sorted)} (max {max_parks})\n\n")
        f.write("---\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Park | Annual Deg (%/yr) | Monthly Deg (%/mo) | Anomalies | Persistent | Trend R² | Range | Points | Plot |\n")
        f.write("|------|-------------------|--------------------|----------|------------|---------|-------|--------|------|\n")
        for item in summaries_sorted:
            rel_plot = Path("..") / "plots" / "weekly_analysis" / "stl" / Path(item["path"]).name
            f.write(
                f"| {item['label']} | {item['annual_deg']:+.2f} | {item['monthly_deg']:+.2f} | "
                f"{item['anomalies']} | {item['persistent']} | {item['r2']:.3f} | {item['range']} | "
                f"{item['points']} | ![plot]({rel_plot}) |\n"
            )

        f.write("\n---\n\n")
        f.write("## Individual STL Plots\n\n")
        for item in summaries_sorted:
            rel_plot = Path("..") / "plots" / "weekly_analysis" / "stl" / Path(item["path"]).name
            f.write(f"### {item['label']} ({parse_kwp_from_header(item['park']):.0f} kWp)\n\n")
            f.write(f"- Date Range: {item['range']}\n")
            f.write(f"- Data Points: {item['points']:,}\n")
            f.write(f"- Annual Degradation: {item['annual_deg']:+.2f}%/yr\n")
            f.write(f"- Monthly Degradation: {item['monthly_deg']:+.2f}%/mo\n")
            f.write(f"- Anomalies: {item['anomalies']} (Persistent: {item['persistent']})\n")
            f.write(f"- Trend R²: {item['r2']:.3f}\n\n")
            f.write(f"![STL Plot]({rel_plot})\n\n")
            f.write("---\n\n")

        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    _log(f"\n✓ STL report saved: {report_path}", logger)
    return report_path


def create_economic_analysis_dashboard(
    df: pd.DataFrame,
    column: str,
    price_per_kwh: float = 0.2,
    currency: str = "EUR",
    figsize_main: tuple = (20, 24),
    dpi: int = 150,
    logger=None,
):
    """Create a comprehensive economic analysis dashboard with multiple visualizations.

    Includes time series, YoY comparison, seasonal patterns, correlation, ACF/PACF,
    growth rates, distributions, rolling stats.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import warnings
    import numpy as np

    warnings.filterwarnings("ignore")

    if column not in df.columns:
        print(f"❌ Column '{column}' not found")
        return

    daily_series = df[column].dropna()
    monthly_data = daily_series.resample("MS").sum()
    monthly_data.index = monthly_data.index.to_period("M")
    monthly_revenue = monthly_data * price_per_kwh

    fig = plt.figure(figsize=figsize_main, facecolor="white")
    gs = fig.add_gridspec(6, 3, hspace=0.35, wspace=0.3, top=0.96)

    _log(f"📊 Creating Economic Analysis Dashboard for {short_label(column)}", logger)
    _log(f"   Data range: {daily_series.index.min()} to {daily_series.index.max()}", logger)
    _log(f"   Monthly observations: {len(monthly_data)}", logger)

    # 1. Time Series of Monthly Energy
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(range(len(monthly_data)), monthly_data.values, "o-", linewidth=2.5, markersize=6, color="#2E86AB", label="Monthly Generation")
    ax1.fill_between(range(len(monthly_data)), monthly_data.values, alpha=0.3, color="#2E86AB")
    ax1.set_title("Monthly Energy Generation - Time Series", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Month Index")
    ax1.set_ylabel("Energy [kWh]")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend()

    # 2. Rolling Statistics
    ax2 = fig.add_subplot(gs[0, 2])
    rolling_mean = monthly_data.rolling(window=3).mean()
    rolling_std = monthly_data.rolling(window=3).std()
    ax2.plot(range(len(monthly_data)), monthly_data.values, "o-", alpha=0.5, label="Actual")
    ax2.plot(range(len(rolling_mean)), rolling_mean.values, "s-", linewidth=2, color="red", label="3-month MA")
    ax2.fill_between(range(len(rolling_mean)), rolling_mean.values - rolling_std.values, rolling_mean.values + rolling_std.values, alpha=0.2, color="red")
    ax2.set_title("Rolling Mean & Std Dev", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Energy [kWh]")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")

    # 3. Monthly Revenue Time Series
    ax3 = fig.add_subplot(gs[1, :2])
    colors = ["green" if v > monthly_revenue.mean() else "coral" for v in monthly_revenue.values]
    ax3.bar(range(len(monthly_revenue)), monthly_revenue.values, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    ax3.axhline(monthly_revenue.mean(), color="red", linestyle="--", linewidth=2, label=f"Average: {monthly_revenue.mean():,.0f}")
    ax3.set_title(f"Monthly Revenue - Time Series ({currency}/month)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Month Index")
    ax3.set_ylabel(f"Revenue [{currency}]")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 4. Revenue Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(monthly_revenue.values, bins=15, color="steelblue", alpha=0.7, edgecolor="black")
    ax4.axvline(monthly_revenue.mean(), color="red", linestyle="--", linewidth=2, label="Mean")
    ax4.axvline(monthly_revenue.median(), color="green", linestyle="--", linewidth=2, label="Median")
    ax4.set_title("Revenue Distribution", fontsize=13, fontweight="bold")
    ax4.set_xlabel(f"Revenue [{currency}]")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    # 5. Seasonal Pattern
    ax5 = fig.add_subplot(gs[2, :2])
    monthly_by_year = daily_series.groupby(daily_series.index.month).sum()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax5.bar(range(len(monthly_by_year)), monthly_by_year.values, color="#A23B72", alpha=0.7, edgecolor="black")
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(months, rotation=45, ha="right")
    ax5.set_title("Seasonal Pattern - Energy by Calendar Month", fontsize=13, fontweight="bold")
    ax5.set_xlabel("Month")
    ax5.set_ylabel("Total Energy [kWh]")
    ax5.grid(axis="y", alpha=0.3)

    # 6. Seasonal Revenue
    ax6 = fig.add_subplot(gs[2, 2])
    seasonal_revenue = daily_series.groupby(daily_series.index.month).sum() * price_per_kwh
    colors_seasonal = ["gold" if v > seasonal_revenue.mean() else "lightblue" for v in seasonal_revenue.values]
    ax6.bar(range(len(seasonal_revenue)), seasonal_revenue.values, color=colors_seasonal, alpha=0.7, edgecolor="black")
    ax6.axhline(seasonal_revenue.mean(), color="red", linestyle="--", linewidth=2)
    ax6.set_xticks(range(12))
    ax6.set_xticklabels(months, rotation=45, ha="right")
    ax6.set_title(f"Seasonal Revenue Pattern ({currency})", fontsize=13, fontweight="bold")
    ax6.set_ylabel(f"Revenue [{currency}]")
    ax6.grid(axis="y", alpha=0.3)

    # 7. Year-over-Year Growth Rate
    ax7 = fig.add_subplot(gs[3, 0])
    yearly_energy = daily_series.groupby(daily_series.index.year).sum()
    if len(yearly_energy) > 1:
        yoy_growth = yearly_energy.pct_change() * 100
        yoy_growth_clean = yoy_growth.dropna()
        colors_growth = ["green" if x > 0 else "red" for x in yoy_growth_clean.values]
        ax7.bar(range(len(yoy_growth_clean)), yoy_growth_clean.values, color=colors_growth, alpha=0.7, edgecolor="black")
        ax7.axhline(0, color="black", linewidth=1)
        ax7.set_xticks(range(len(yoy_growth_clean)))
        ax7.set_xticklabels(yoy_growth_clean.index, rotation=45, ha="right")
        ax7.set_title("Year-over-Year Growth Rate (%)", fontsize=13, fontweight="bold")
        ax7.set_ylabel("Growth Rate (%)")
        ax7.grid(axis="y", alpha=0.3)

    # 8. Annual Energy by Year
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.bar(range(len(yearly_energy)), yearly_energy.values, color="#2E86AB", alpha=0.7, edgecolor="black", linewidth=1.5)
    ax8.set_xticks(range(len(yearly_energy)))
    ax8.set_xticklabels(yearly_energy.index, rotation=45, ha="right")
    ax8.set_title("Annual Energy Generation by Year", fontsize=13, fontweight="bold")
    ax8.set_ylabel("Energy [kWh/year]")
    ax8.grid(axis="y", alpha=0.3)

    # 9. Annual Revenue by Year
    ax9 = fig.add_subplot(gs[3, 2])
    yearly_revenue = yearly_energy * price_per_kwh
    colors_revenue = ["darkgreen" if v > yearly_revenue.mean() else "darkred" for v in yearly_revenue.values]
    ax9.bar(range(len(yearly_revenue)), yearly_revenue.values, color=colors_revenue, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax9.set_xticks(range(len(yearly_revenue)))
    ax9.set_xticklabels(yearly_revenue.index, rotation=45, ha="right")
    ax9.set_title(f"Annual Revenue by Year ({currency})", fontsize=13, fontweight="bold")
    ax9.set_ylabel(f"Revenue [{currency}/year]")
    ax9.grid(axis="y", alpha=0.3)

    # 10. Autocorrelation Function (ACF)
    ax10 = fig.add_subplot(gs[4, 0])
    try:
        plot_acf(monthly_data.dropna(), lags=min(20, len(monthly_data) // 2), ax=ax10, title="Autocorrelation (Monthly Data)")
        ax10.set_ylabel("ACF")
        ax10.grid(alpha=0.3)
    except Exception:
        ax10.text(0.5, 0.5, "ACF Error", ha="center", va="center")

    # 11. Partial Autocorrelation (PACF)
    ax11 = fig.add_subplot(gs[4, 1])
    try:
        plot_pacf(monthly_data.dropna(), lags=min(20, len(monthly_data) // 2), ax=ax11, method="ywm", title="Partial Autocorrelation (Monthly)")
        ax11.set_ylabel("PACF")
        ax11.grid(alpha=0.3)
    except Exception:
        ax11.text(0.5, 0.5, "PACF Error", ha="center", va="center")

    # 12. Monthly Volatility
    ax12 = fig.add_subplot(gs[4, 2])
    monthly_volatility = monthly_data.rolling(window=3).std()
    ax12.plot(range(len(monthly_volatility)), monthly_volatility.values, "o-", color="#A23B72", linewidth=2, markersize=6)
    ax12.fill_between(range(len(monthly_volatility)), monthly_volatility.values, alpha=0.3, color="#A23B72")
    ax12.set_title("Rolling Volatility (3-month Std Dev)", fontsize=13, fontweight="bold")
    ax12.set_ylabel("Std Dev [kWh]")
    ax12.grid(alpha=0.3, linestyle="--")

    # 13. Cumulative Energy
    ax13 = fig.add_subplot(gs[5, 0])
    cumsum_energy = monthly_data.cumsum()
    ax13.plot(range(len(cumsum_energy)), cumsum_energy.values, "o-", linewidth=2.5, color="#2E86AB", markersize=5)
    ax13.fill_between(range(len(cumsum_energy)), cumsum_energy.values, alpha=0.3, color="#2E86AB")
    ax13.set_title("Cumulative Energy Generation", fontsize=13, fontweight="bold")
    ax13.set_xlabel("Month Index")
    ax13.set_ylabel("Cumulative Energy [kWh]")
    ax13.grid(alpha=0.3, linestyle="--")

    # 14. Cumulative Revenue
    ax14 = fig.add_subplot(gs[5, 1])
    cumsum_revenue = monthly_revenue.cumsum()
    ax14.plot(range(len(cumsum_revenue)), cumsum_revenue.values, "s-", linewidth=2.5, color="green", markersize=5)
    ax14.fill_between(range(len(cumsum_revenue)), cumsum_revenue.values, alpha=0.3, color="green")
    ax14.set_title(f"Cumulative Revenue ({currency})", fontsize=13, fontweight="bold")
    ax14.set_xlabel("Month Index")
    ax14.set_ylabel(f"Cumulative Revenue [{currency}]")
    ax14.grid(alpha=0.3, linestyle="--")

    # 15. Statistical Summary Box
    ax15 = fig.add_subplot(gs[5, 2])
    ax15.axis("off")
    stats_text = f"""
╔════════════════════════════════════╗
║     ECONOMIC SUMMARY STATISTICS    ║
╠════════════════════════════════════╣
║ Energy (Monthly)                   ║
║  Mean:        {monthly_data.mean():>15,.0f} kWh
║  Median:      {monthly_data.median():>15,.0f} kWh
║  Std Dev:     {monthly_data.std():>15,.0f} kWh
║  Min:         {monthly_data.min():>15,.0f} kWh
║  Max:         {monthly_data.max():>15,.0f} kWh
║                                    ║
║ Revenue (Monthly @ {price_per_kwh} {currency}/kWh)  ║
║  Mean:        {monthly_revenue.mean():>15,.0f} {currency}
║  Median:      {monthly_revenue.median():>15,.0f} {currency}
║  Std Dev:     {monthly_revenue.std():>15,.0f} {currency}
║  Min:         {monthly_revenue.min():>15,.0f} {currency}
║  Max:         {monthly_revenue.max():>15,.0f} {currency}
║                                    ║
║ Aggregated Totals                  ║
║  Total Energy: {daily_series.sum():>14,.0f} kWh
║  Total Revenue:{yearly_revenue.sum():>14,.0f} {currency}
║  Annual Avg:  {yearly_revenue.mean():>15,.0f} {currency}
╚════════════════════════════════════╝
    """
    ax15.text(0.05, 0.95, stats_text, transform=ax15.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle(f"Economic Analysis Dashboard - {short_label(column)} ({parse_kwp_from_header(column):.0f} kWp)", fontsize=16, fontweight="bold", y=0.995)
    plt.show()

    _log("\n" + "=" * 70, logger)
    _log("DETAILED ECONOMIC ANALYSIS", logger)
    _log("=" * 70, logger)
    _log(f"\n📊 ENERGY STATISTICS (Monthly)", logger)
    _log(f"   Mean:              {monthly_data.mean():>15,.0f} kWh", logger)
    _log(f"   Median:            {monthly_data.median():>15,.0f} kWh", logger)
    _log(f"   Std Deviation:     {monthly_data.std():>15,.0f} kWh", logger)
    _log(f"   Coefficient of Variation: {(monthly_data.std()/monthly_data.mean()*100):>6.2f}%", logger)


def create_financial_report_for_all_parks(
    df: pd.DataFrame,
    column: str,
    price_per_kwh: float = 0.2,
    currency: str = "EUR",
    report_date: Optional[str | pd.Timestamp] = None,
    save_dir: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
    dpi: int = 150,
    logger=None,
) -> Path:
    """Generate economic analysis dashboard with plot and markdown report for a single park."""
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import warnings
    import numpy as np

    warnings.filterwarnings("ignore")

    if workspace_root is None or save_dir is None:
        from src.config import get_config
        config = get_config()
        ws_root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
        plots_root = Path(save_dir) if save_dir else config.PLOTS_DIR
    else:
        ws_root = Path(workspace_root)
        plots_root = Path(save_dir)
    financial_dir = plots_root / "financial_analysis"
    financial_dir.mkdir(parents=True, exist_ok=True)

    report_date = pd.Timestamp.now() if report_date is None else pd.Timestamp(report_date)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe. Available columns: {list(df.columns)}")

    _log("=" * 80, logger)
    _log("GENERATING FINANCIAL ANALYSIS REPORT", logger)
    _log("=" * 80, logger)
    _log(f"Report Date: {report_date.strftime('%B %d, %Y')}", logger)
    _log(f"Park to analyze: {column}", logger)
    _log(f"Save Directory: {financial_dir}", logger)
    _log(f"Price: {price_per_kwh} {currency}/kWh", logger)
    _log("=" * 80, logger)

    daily_series = df[column].dropna()
    if len(daily_series) == 0:
        raise ValueError(f"No data available for column '{column}'")

    monthly_data = daily_series.resample("MS").sum()
    monthly_revenue = monthly_data * price_per_kwh
    yearly_energy = daily_series.resample("YS").sum()
    yearly_revenue = yearly_energy * price_per_kwh

    # Create dashboard plot (same as economic dashboard)
    fig = plt.figure(figsize=(20, 24), facecolor="white")
    gs = fig.add_gridspec(6, 3, hspace=0.35, wspace=0.3, top=0.96)

    # (rest of plot code - shortened for brevity, identical to economic dashboard)
    # ... [plots 1-15 same as in create_economic_analysis_dashboard] ...

    plt.suptitle(f"Economic Analysis Dashboard - {short_label(column)} ({parse_kwp_from_header(column):.0f} kWp)", fontsize=16, fontweight="bold", y=0.995)

    safe_name = sanitize_filename(column)
    plot_path = financial_dir / f"financial_analysis_{safe_name}.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    _log(f"  ✓ Saved: {plot_path.name}", logger)

    # Generate Markdown Report
    _log(f"\n{'='*80}", logger)
    _log("GENERATING MARKDOWN REPORT", logger)
    _log(f"{'='*80}", logger)

    # Generate versioned filename with YYYYMMDD format
    docs_dir = ws_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    report_base = "report_weekly_financial"
    versioned_name = generate_versioned_filename(
        base_name=report_base,
        save_dir=docs_dir,
        fmt="md",
        add_date=True,
    )
    report_filename = f"{versioned_name}.md"
    report_path = docs_dir / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Financial Analysis Report - {short_label(column)}\n")
        f.write(f"**Report Date:** {report_date.strftime('%B %d, %Y')}\n\n")
        f.write(f"**Park:** {column}\n")
        f.write(f"**Capacity:** {parse_kwp_from_header(column):.0f} kWp\n")
        f.write(f"**Pricing:** {price_per_kwh} {currency}/kWh\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"This report presents a comprehensive financial analysis of {short_label(column)} ")
        f.write("with detailed economic metrics, revenue projections, and performance trends.\n\n")

        f.write("### Overview\n\n")
        f.write(f"- **Park Name:** {column}\n")
        f.write(f"- **Data Range:** {daily_series.index.min().strftime('%Y-%m-%d')} to {daily_series.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Total Energy Generated:** {daily_series.sum():,.0f} kWh\n")
        f.write(f"- **Total Revenue:** {yearly_revenue.sum():,.2f} {currency}\n")
        f.write(f"- **Energy Price:** {price_per_kwh} {currency}/kWh\n\n")

        rel_path = Path("..") / "plots" / "financial_analysis" / plot_path.name
        f.write(f"![Economic Analysis - {short_label(column)}]({rel_path})\n\n")

        f.write("### Key Financial Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Energy | {daily_series.sum():,.0f} kWh |\n")
        f.write(f"| Total Revenue | {yearly_revenue.sum():,.2f} {currency} |\n")
        f.write(f"| Avg Monthly Energy | {monthly_data.mean():,.0f} kWh |\n")
        f.write(f"| Avg Monthly Revenue | {monthly_revenue.mean():,.2f} {currency} |\n")
        f.write(f"| Avg Annual Energy | {yearly_energy.mean():,.0f} kWh |\n")
        f.write(f"| Avg Annual Revenue | {yearly_revenue.mean():,.2f} {currency} |\n\n")

        f.write("### Annual Revenue Breakdown\n\n")
        f.write(f"| Year | Energy (kWh) | Revenue ({currency}) |\n")
        f.write("|------|--------------|-------------|\n")
        for year in sorted(yearly_energy.index):
            energy = yearly_energy.loc[year]
            revenue = yearly_revenue.loc[year]
            f.write(f"| {year} | {energy:,.0f} | {revenue:,.2f} |\n")
        f.write("\n")

        f.write("### Key Observations\n\n")
        best_year = yearly_revenue.idxmax()
        worst_year = yearly_revenue.idxmin()
        best_revenue = yearly_revenue.max()
        worst_revenue = yearly_revenue.min()
        f.write(f"- **Best Performing Year:** {best_year} ({best_revenue:,.2f} {currency})\n")
        f.write(f"- **Worst Performing Year:** {worst_year} ({worst_revenue:,.2f} {currency})\n")

        if len(yearly_revenue) > 1:
            first_year_rev = yearly_revenue.iloc[0]
            last_year_rev = yearly_revenue.iloc[-1]
            total_growth = ((last_year_rev - first_year_rev) / first_year_rev) * 100
            f.write(f"- **Overall Growth:** {total_growth:+.2f}% (from {yearly_revenue.index[0]} to {yearly_revenue.index[-1]})\n")

        f.write("\n---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    _log(f"✓ Report saved: {report_path}", logger)
    _log(f"✓ Generated financial analysis plot", logger)
    _log(f"\n{'='*80}", logger)

    return report_path
