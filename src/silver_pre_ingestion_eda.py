from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class EdaConfig:
    """Configuration for Silver EDA analysis.
    
    Args:
        output_dir: Directory for saved outputs (required if save_plots or save_stats=True)
        max_units: Maximum units to analyze in unit stats
        max_signals: Maximum signals to plot (limits number of signals analyzed)
        max_parks: Maximum parks per signal grid (None = all parks)
        grid_cols: Number of columns in park grid plots
        focus_signal: Plot only this specific signal
        focus_unit: Unit to use with focus_signal
        focus_signals: List of specific signals to plot
        plot_kinds: Types of plots to generate (timeseries, hist, box, coverage)
        max_days: Filter data to last N days (applies to ALL plots and stats)
        max_xticks: Maximum x-axis ticks in coverage heatmap
        smooth_window: Window size for smoothing in timeseries plots
        quantiles: Quantiles to compute in statistics
        sample_rows: Sample size for plots (None = use all data)
        save_plots: Save plot figures to disk
        save_stats: Save statistics tables to disk
    """
    output_dir: Optional[Path] = None
    max_units: int = 8
    max_signals: int = 12
    max_parks: Optional[int] = None
    grid_cols: int = 3
    focus_signal: Optional[str] = None
    focus_unit: Optional[str] = None
    focus_signals: Optional[List[str]] = None
    plot_kinds: Optional[List[str]] = None
    max_days: Optional[int] = 120
    max_xticks: int = 12
    smooth_window: int = 21
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)
    sample_rows: Optional[int] = 200_000
    save_plots: bool = False
    save_stats: bool = False



def _coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    if "interval_start_date" in df.columns:
        df["interval_start_date"] = pd.to_datetime(df["interval_start_date"], errors="coerce").dt.date
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in ["unit", "signal_name", "park_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
    return df


def _maybe_sample(df: pd.DataFrame, sample_rows: Optional[int]) -> pd.DataFrame:
    if sample_rows is None or len(df) <= sample_rows:
        return df
    return df.sample(sample_rows, random_state=42)


def _summary_overview(df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "rows": int(len(df)),
        "parks": int(df["park_id"].nunique()) if "park_id" in df.columns else 0,
        "signals": int(df["signal_name"].nunique()) if "signal_name" in df.columns else 0,
        "units": int(df["unit"].nunique()) if "unit" in df.columns else 0,
        "ts_utc_min": None,
        "ts_utc_max": None,
    }
    if "ts_utc" in df.columns and df["ts_utc"].notna().any():
        summary["ts_utc_min"] = df["ts_utc"].min().isoformat()
        summary["ts_utc_max"] = df["ts_utc"].max().isoformat()
    return summary


def _quantile_label(q: float) -> str:
    pct = int(round(q * 100))
    return f"p{pct:02d}"


def stats_by_unit(
    df: pd.DataFrame,
    quantiles: Iterable[float],
) -> pd.DataFrame:
    if df.empty or "unit" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    group = df.dropna(subset=["value"]).groupby("unit")["value"]
    stats = group.agg(count="size", min="min", max="max", mean="mean", std="std")

    q_list = list(quantiles)
    q_labels = [_quantile_label(q) for q in q_list]
    q_df = group.quantile(q_list).unstack()
    q_df.columns = q_labels

    out = stats.join(q_df, how="left").reset_index()
    return out


def stats_by_signal_unit(
    df: pd.DataFrame,
    quantiles: Iterable[float],
    max_signals: int,
) -> pd.DataFrame:
    if df.empty or "signal_name" not in df.columns or "unit" not in df.columns:
        return pd.DataFrame()

    counts = (
        df.groupby(["signal_name", "unit"], dropna=False)["value"]
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top = counts.head(max_signals)
    if top.empty:
        return top

    keys = set(tuple(x) for x in top[["signal_name", "unit"]].to_numpy())
    mask = df[["signal_name", "unit"]].apply(tuple, axis=1).isin(keys)
    subset = df[mask].copy()

    group = subset.groupby(["signal_name", "unit"])["value"]
    stats = group.agg(min="min", max="max", mean="mean", std="std")

    q_list = list(quantiles)
    q_labels = [_quantile_label(q) for q in q_list]
    q_df = group.quantile(q_list).unstack()
    q_df.columns = q_labels

    out = stats.join(q_df, how="left").reset_index()
    out = out.merge(top, on=["signal_name", "unit"], how="left")
    return out.sort_values("count", ascending=False)


def coverage_by_park_signal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "park_id" not in df.columns or "signal_name" not in df.columns:
        return pd.DataFrame()

    if "interval_start_date" in df.columns:
        group = df.groupby(["park_id", "signal_name"])["interval_start_date"]
        coverage = group.agg(
            first_date="min",
            last_date="max",
            unique_days=lambda s: int(s.nunique()),
        ).reset_index()
    elif "ts_utc" in df.columns:
        group = df.groupby(["park_id", "signal_name"])["ts_utc"]
        coverage = group.agg(
            first_date="min",
            last_date="max",
            unique_days=lambda s: int(s.dt.floor("D").nunique()),
        ).reset_index()
    else:
        return pd.DataFrame()

    counts = df.groupby(["park_id", "signal_name"]).size().reset_index(name="rows")
    return coverage.merge(counts, on=["park_id", "signal_name"], how="left")


def _select_focus_pair(
    df: pd.DataFrame,
    focus_signal: Optional[str],
    focus_unit: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if "signal_name" not in df.columns:
        return None, None

    if focus_signal:
        signal = str(focus_signal).strip().lower()
        if "unit" not in df.columns:
            return signal, None
        if focus_unit:
            return signal, str(focus_unit).strip().lower()
        sub = df[df["signal_name"] == signal]
        if sub.empty:
            return signal, None
        unit = sub["unit"].value_counts().index[0]
        return signal, unit

    if "unit" in df.columns:
        counts = df.groupby(["signal_name", "unit"]).size().sort_values(ascending=False)
        if counts.empty:
            return None, None
        signal, unit = counts.index[0]
        return signal, unit

    counts = df["signal_name"].value_counts()
    if counts.empty:
        return None, None
    return counts.index[0], None


def _select_unit_for_signal(df: pd.DataFrame, signal: str) -> Optional[str]:
    if "unit" not in df.columns:
        return None
    sub = df[df["signal_name"] == signal]
    if sub.empty:
        return None
    return sub["unit"].value_counts().index[0]


def _select_focus_pairs(
    df: pd.DataFrame,
    focus_signal: Optional[str],
    focus_unit: Optional[str],
    focus_signals: Optional[Sequence[str]],
    max_signals: int,
) -> List[Tuple[Optional[str], Optional[str]]]:
    if df.empty or "signal_name" not in df.columns:
        return []

    if focus_signal:
        signal = str(focus_signal).strip().lower()
        if focus_unit:
            return [(signal, str(focus_unit).strip().lower())]
        return [(signal, None)]

    if focus_signals:
        pairs = []
        for sig in focus_signals:
            signal = str(sig).strip().lower()
            unit = str(focus_unit).strip().lower() if focus_unit else None
            pairs.append((signal, unit))
        return pairs

    signals = df["signal_name"].value_counts().head(max_signals).index.tolist()
    pairs = []
    for signal in signals:
        pairs.append((signal, None))
    return pairs


def _focus_subset(
    df: pd.DataFrame,
    focus_signal: Optional[str],
    focus_unit: Optional[str],
    max_parks: Optional[int],
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], List[str]]:
    signal, unit = _select_focus_pair(df, focus_signal, focus_unit)
    sub = df.copy()
    if signal:
        sub = sub[sub["signal_name"] == signal]
    if unit and "unit" in sub.columns:
        sub = sub[sub["unit"] == unit]
    if sub.empty:
        return sub, signal, unit, []
    if max_parks is None or max_parks <= 0:
        parks = sub["park_id"].value_counts().index.tolist()
    else:
        parks = sub["park_id"].value_counts().head(max_parks).index.tolist()
    return sub, signal, unit, parks


def _focus_subset_for_pair(
    df: pd.DataFrame,
    signal: Optional[str],
    unit: Optional[str],
    max_parks: Optional[int],
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], List[str]]:
    sub = df.copy()
    if signal:
        sub = sub[sub["signal_name"] == signal]
    if unit and "unit" in sub.columns:
        sub = sub[sub["unit"] == unit]
    if sub.empty:
        return sub, signal, unit, []
    if max_parks is None or max_parks <= 0:
        parks = sub["park_id"].value_counts().index.tolist()
    else:
        parks = sub["park_id"].value_counts().head(max_parks).index.tolist()
    return sub, signal, unit, parks


def _downsample_timeseries(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    step = max(1, int(len(df) / max_rows))
    return df.iloc[::step]


def _smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(values) < window:
        return values
    min_periods = max(1, window // 2)
    return values.rolling(window=window, center=True, min_periods=min_periods).mean()


def _safe_label(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _plot_park_grid_timeseries(
    df: pd.DataFrame,
    focus_signal: Optional[str],
    focus_unit: Optional[str],
    max_parks: int,
    grid_cols: int,
    output_dir: Optional[Path],
    save: bool,
    sample_rows: Optional[int],
    smooth_window: int,
    label_suffix: Optional[str] = None,
    parks_override: Optional[List[str]] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path], Optional[str], Optional[str]]:
    if df.empty or "park_id" not in df.columns or "value" not in df.columns:
        return None, None, focus_signal, focus_unit

    if parks_override:
        sub = df
        signal = focus_signal
        unit = focus_unit
        parks = parks_override
    else:
        sub, signal, unit, parks = _focus_subset(df, focus_signal, focus_unit, max_parks)
    if sub.empty or not parks:
        return None, None, signal, unit

    ncols = max(1, grid_cols)
    nrows = int((len(parks) + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, park in zip(axes.flat, parks):
        park_df = sub[sub["park_id"] == park].copy()
        park_df = _downsample_timeseries(park_df, sample_rows)
        if "ts_utc" in park_df.columns:
            park_df = park_df.sort_values("ts_utc")
            x = park_df["ts_utc"]
        elif "interval_start_date" in park_df.columns:
            park_df = park_df.sort_values("interval_start_date")
            x = park_df["interval_start_date"]
        else:
            x = range(len(park_df))
        y = park_df["value"].astype(float)
        ax.plot(x[: len(park_df)], y, linewidth=0.8, alpha=0.5)
        smooth = _smooth_series(y, window=smooth_window)
        ax.plot(x[: len(park_df)], smooth, linewidth=1.6, alpha=0.9, color="#D62728")
        values = y.dropna()
        if not values.empty:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            ax.axhline(lower, color="#9E9E9E", linewidth=0.9, alpha=0.7, linestyle="--")
            ax.axhline(upper, color="#9E9E9E", linewidth=0.9, alpha=0.7, linestyle="--")
        ax.set_title(str(park))
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes.flat[len(parks):]:
        ax.set_visible(False)

    title_parts = ["per-park timeseries"]
    if signal:
        title_parts.append(f"signal={signal}")
    if unit:
        title_parts.append(f"unit={unit}")
    fig.suptitle(" | ".join(title_parts))
    fig.tight_layout()

    out_path = None
    if save and output_dir is not None:
        suffix = _safe_label(label_suffix or signal)
        out_path = output_dir / f"park_grid_timeseries_{suffix}.png"
        fig.savefig(out_path, dpi=150)
    return fig, out_path, signal, unit


def _plot_park_grid_histograms(
    df: pd.DataFrame,
    parks: List[str],
    signal: Optional[str],
    unit: Optional[str],
    grid_cols: int,
    output_dir: Optional[Path],
    save: bool,
    sample_rows: Optional[int],
    label_suffix: Optional[str] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    if df.empty or not parks:
        return None, None

    ncols = max(1, grid_cols)
    nrows = int((len(parks) + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, park in zip(axes.flat, parks):
        park_df = df[df["park_id"] == park].copy()
        park_df = _maybe_sample(park_df, sample_rows)
        values = park_df["value"].dropna()
        ax.hist(values, bins=40, color="#4C78A8", alpha=0.85)
        if not values.empty:
            p50 = values.quantile(0.5)
            p70 = values.quantile(0.7)
            p90 = values.quantile(0.9)
            ax.axvline(p50, color="#E45756", linewidth=1.2, alpha=0.9)
            ax.axvline(p70, color="#F58518", linewidth=1.0, alpha=0.9, linestyle="--")
            ax.axvline(p90, color="#54A24B", linewidth=1.0, alpha=0.9, linestyle="--")
        ax.set_title(str(park))
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    for ax in axes.flat[len(parks):]:
        ax.set_visible(False)

    title_parts = ["per-park distribution"]
    if signal:
        title_parts.append(f"signal={signal}")
    if unit:
        title_parts.append(f"unit={unit}")
    fig.suptitle(" | ".join(title_parts))
    fig.tight_layout()

    out_path = None
    if save and output_dir is not None:
        suffix = _safe_label(label_suffix or signal)
        out_path = output_dir / f"park_grid_hist_{suffix}.png"
        fig.savefig(out_path, dpi=150)
    return fig, out_path


def _plot_park_grid_boxplots(
    df: pd.DataFrame,
    parks: List[str],
    signal: Optional[str],
    unit: Optional[str],
    grid_cols: int,
    output_dir: Optional[Path],
    save: bool,
    sample_rows: Optional[int],
    label_suffix: Optional[str] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    if df.empty or not parks:
        return None, None

    ncols = max(1, grid_cols)
    nrows = int((len(parks) + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, park in zip(axes.flat, parks):
        park_df = df[df["park_id"] == park].copy()
        park_df = _maybe_sample(park_df, sample_rows)
        values = park_df["value"].dropna()
        if values.empty:
            ax.set_visible(False)
            continue
        ax.boxplot(values, vert=True, showfliers=False)
        ax.set_title(str(park))
        ax.set_xticks([])

    for ax in axes.flat[len(parks):]:
        ax.set_visible(False)

    title_parts = ["per-park boxplot"]
    if signal:
        title_parts.append(f"signal={signal}")
    if unit:
        title_parts.append(f"unit={unit}")
    fig.suptitle(" | ".join(title_parts))
    fig.tight_layout()

    out_path = None
    if save and output_dir is not None:
        suffix = _safe_label(label_suffix or signal)
        out_path = output_dir / f"park_grid_box_{suffix}.png"
        fig.savefig(out_path, dpi=150)
    return fig, out_path


def _plot_coverage_heatmap(
    df: pd.DataFrame,
    parks: List[str],
    signal: Optional[str],
    unit: Optional[str],
    max_days: Optional[int],
    max_xticks: int,
    output_dir: Optional[Path],
    save: bool,
    label_suffix: Optional[str] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    if df.empty or not parks:
        return None, None

    if "interval_start_date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["interval_start_date"], errors="coerce")
    elif "ts_utc" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True).dt.floor("D")
    else:
        return None, None

    df = df.dropna(subset=["date"])
    if df.empty:
        return None, None

    df = df[df["park_id"].isin(parks)]
    # Note: max_days filtering now applied globally in run_silver_pre_ingestion_eda

    pivot = df.groupby(["park_id", "date"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    if pivot.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(parks) + 2))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    if len(pivot.columns) > 0:
        step = max(1, int(len(pivot.columns) / max(1, max_xticks)))
        xticks = list(range(0, len(pivot.columns), step))
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [pivot.columns[i].strftime("%Y-%m-%d") for i in xticks],
            rotation=90,
            fontsize=7,
        )
    title_parts = ["coverage heatmap (rows per day)"]
    if signal:
        title_parts.append(f"signal={signal}")
    if unit:
        title_parts.append(f"unit={unit}")
    ax.set_title(" | ".join(title_parts))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    out_path = None
    if save and output_dir is not None:
        suffix = _safe_label(label_suffix or signal)
        out_path = output_dir / f"coverage_heatmap_{suffix}.png"
        fig.savefig(out_path, dpi=150)
    return fig, out_path


def stats_by_park(
    df: pd.DataFrame,
    parks: List[str],
    quantiles: Iterable[float],
) -> pd.DataFrame:
    if df.empty or "park_id" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    if parks:
        df = df[df["park_id"].isin(parks)]
    if df.empty:
        return pd.DataFrame()
    group = df.groupby("park_id")["value"]
    stats = group.agg(count="size", min="min", max="max", mean="mean", std="std")
    q_list = list(quantiles)
    q_labels = [_quantile_label(q) for q in q_list]
    q_df = group.quantile(q_list).unstack()
    q_df.columns = q_labels
    return stats.join(q_df, how="left").reset_index()


def run_silver_pre_ingestion_eda(
    df: pd.DataFrame,
    config: EdaConfig,
) -> Dict[str, object]:
    df = _coerce_frame(df)
    
    # Filter to last N days if max_days specified (applies to all plots)
    if config.max_days is not None and not df.empty:
        if "interval_start_date" in df.columns:
            df["_temp_date"] = pd.to_datetime(df["interval_start_date"], errors="coerce")
        elif "ts_utc" in df.columns:
            df["_temp_date"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True).dt.floor("D")
        else:
            df["_temp_date"] = None
        
        if "_temp_date" in df.columns and df["_temp_date"].notna().any():
            last_date = df["_temp_date"].max()
            df = df[df["_temp_date"] >= last_date - pd.Timedelta(days=config.max_days)].copy()
            df = df.drop(columns=["_temp_date"])
    
    if config.save_plots or config.save_stats:
        if config.output_dir is None:
            raise ValueError("output_dir must be set when save_plots or save_stats is True")
        config.output_dir.mkdir(parents=True, exist_ok=True)

    overview = _summary_overview(df)
    overview["computed_at_utc"] = datetime.now(timezone.utc).isoformat()

    unit_stats = stats_by_unit(df, config.quantiles)
    signal_stats = stats_by_signal_unit(df, config.quantiles, config.max_signals)
    coverage = coverage_by_park_signal(df)
    focus_pairs = _select_focus_pairs(
        df,
        config.focus_signal,
        config.focus_unit,
        config.focus_signals,
        config.max_signals,
    )
    park_stats_list: List[pd.DataFrame] = []

    overview_path = None
    if config.save_stats and config.output_dir is not None:
        overview_path = config.output_dir / "silver_eda_summary.json"
        overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")

    unit_stats_path = None
    signal_stats_path = None
    coverage_path = None
    if config.save_stats and config.output_dir is not None:
        if not unit_stats.empty:
            unit_stats_path = config.output_dir / "silver_eda_unit_stats.csv"
            unit_stats.to_csv(unit_stats_path, index=False)
        if not signal_stats.empty:
            signal_stats_path = config.output_dir / "silver_eda_signal_unit_stats.csv"
            signal_stats.to_csv(signal_stats_path, index=False)
        if not coverage.empty:
            coverage_path = config.output_dir / "silver_eda_coverage.csv"
            coverage.to_csv(coverage_path, index=False)

    figs: List[plt.Figure] = []
    plot_paths: List[Path] = []
    plot_kinds = {k.strip().lower() for k in (config.plot_kinds or ["timeseries"]) if k}

    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        all_parks = df["park_id"].value_counts().index.tolist() if "park_id" in df.columns else []

        for idx, (signal, unit) in enumerate(focus_pairs):
            focus_df, focus_signal, focus_unit, focus_parks = _focus_subset_for_pair(
                df, signal, unit, config.max_parks
            )
            if focus_df.empty or not focus_parks:
                continue

            if config.max_parks is None or config.max_parks <= 0:
                parks_to_use = all_parks
            else:
                parks_to_use = focus_parks

            park_stats = stats_by_park(focus_df, focus_parks, config.quantiles)
            if not park_stats.empty:
                park_stats["signal_name"] = focus_signal
                park_stats["unit"] = focus_unit
                park_stats_list.append(park_stats)

            if "timeseries" in plot_kinds:
                fig, path, _, _ = _plot_park_grid_timeseries(
                    focus_df,
                    focus_signal,
                    focus_unit,
                    config.max_parks,
                    config.grid_cols,
                    config.output_dir,
                    config.save_plots,
                    config.sample_rows,
                    config.smooth_window,
                    label_suffix=focus_signal,
                    parks_override=parks_to_use,
                )
                if fig:
                    plt.close(fig)
                    figs.append(fig)
                if path:
                    plot_paths.append(path)

            # Plot all plot types for all signals (controlled by max_signals)
            include_extras = True
            if include_extras and "hist" in plot_kinds:
                fig, path = _plot_park_grid_histograms(
                    focus_df,
                    parks_to_use,
                    focus_signal,
                    focus_unit,
                    config.grid_cols,
                    config.output_dir,
                    config.save_plots,
                    config.sample_rows,
                    label_suffix=focus_signal,
                )
                if fig:
                    plt.close(fig)
                    figs.append(fig)
                if path:
                    plot_paths.append(path)

            if include_extras and "box" in plot_kinds:
                fig, path = _plot_park_grid_boxplots(
                    focus_df,
                    parks_to_use,
                    focus_signal,
                    focus_unit,
                    config.grid_cols,
                    config.output_dir,
                    config.save_plots,
                    config.sample_rows,
                    label_suffix=focus_signal,
                )
                if fig:
                    plt.close(fig)
                    figs.append(fig)
                if path:
                    plot_paths.append(path)

            if include_extras and "coverage" in plot_kinds:
                fig, path = _plot_coverage_heatmap(
                    focus_df,
                    parks_to_use,
                    focus_signal,
                    focus_unit,
                    config.max_days,
                    config.max_xticks,
                    config.output_dir,
                    config.save_plots,
                    label_suffix=focus_signal,
                )
                if fig:
                    plt.close(fig)
                    figs.append(fig)
                if path:
                    plot_paths.append(path)
    finally:
        if was_interactive:
            plt.ion()

    park_stats_all = pd.concat(park_stats_list, ignore_index=True) if park_stats_list else pd.DataFrame()

    focus_signal = focus_pairs[0][0] if len(focus_pairs) == 1 else None
    focus_unit = focus_pairs[0][1] if len(focus_pairs) == 1 else None

    return {
        "overview": overview,
        "unit_stats": unit_stats,
        "signal_stats": signal_stats,
        "coverage": coverage,
        "park_stats": park_stats_all,
        "focus_pairs": focus_pairs,
        "focus_signal": focus_signal,
        "focus_unit": focus_unit,
        "overview_path": overview_path,
        "unit_stats_path": unit_stats_path,
        "signal_stats_path": signal_stats_path,
        "coverage_path": coverage_path,
        "plots": figs,
        "plot_paths": plot_paths,
    }


def build_cfg(args) -> EdaConfig:
    focus_signals = None
    if args.focus_signals:
        focus_signals = [s.strip() for s in args.focus_signals.split(",") if s.strip()]
    plot_kinds = None
    if args.plot_kinds:
        plot_kinds = [s.strip() for s in args.plot_kinds.split(",") if s.strip()]
    return EdaConfig(
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        max_units=args.max_units,
        max_signals=args.max_signals,
        max_parks=args.max_parks,
        grid_cols=args.grid_cols,
        focus_signal=args.focus_signal,
        focus_unit=args.focus_unit,
        focus_signals=focus_signals,
        plot_kinds=plot_kinds,
        max_days=args.max_days,
        max_xticks=args.max_xticks,
        smooth_window=args.smooth_window,
        quantiles=tuple(float(q) for q in args.quantiles.split(",")),
        sample_rows=args.sample_rows,
        save_plots=args.save_plots,
        save_stats=args.save_stats,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Silver pre-ingestion EDA")
    ap.add_argument("--input", required=True, help="Silver parquet (or staged parquet) file")
    ap.add_argument("--output_dir", default=None, help="Output directory for EDA artifacts")
    ap.add_argument("--max_units", type=int, default=8, help="Max units to plot")
    ap.add_argument("--max_signals", type=int, default=12, help="Max signal/unit pairs to summarize")
    ap.add_argument("--max_parks", type=int, default=0, help="Max parks to plot in grid (0 = all)")
    ap.add_argument("--grid_cols", type=int, default=3, help="Grid columns for per-park plots")
    ap.add_argument("--focus_signal", default=None, help="Optional signal_name to plot per-park")
    ap.add_argument("--focus_unit", default=None, help="Optional unit to plot per-park")
    ap.add_argument("--focus_signals", default=None, help="Comma-separated signal_name list")
    ap.add_argument("--plot_kinds", default=None, help="Comma-separated list: timeseries,hist,box,coverage")
    ap.add_argument("--max_days", type=int, default=120, help="Max days for coverage heatmap")
    ap.add_argument("--max_xticks", type=int, default=12, help="Max x-axis ticks for coverage heatmap")
    ap.add_argument("--smooth_window", type=int, default=21, help="Rolling window size for smoothing")
    ap.add_argument("--quantiles", default="0.01,0.05,0.5,0.95,0.99", help="Comma-separated quantiles")
    ap.add_argument("--sample_rows", type=int, default=200000, help="Sample rows per unit for plots")
    ap.add_argument("--save_plots", action="store_true", help="Save plots to output_dir")
    ap.add_argument("--save_stats", action="store_true", help="Save stats to output_dir")
    args = ap.parse_args()

    cfg = build_cfg(args)
    df = pd.read_parquet(Path(args.input))
    outputs = run_silver_pre_ingestion_eda(df, cfg)
    print("EDA overview:")
    print(outputs["overview"])
    if cfg.save_plots or cfg.save_stats:
        print("EDA saved outputs:")
        for k in ["overview_path", "unit_stats_path", "signal_stats_path", "coverage_path", "plot_paths"]:
            print(f"  {k}: {outputs.get(k)}")


__all__ = [
    "EdaConfig",
    "run_silver_pre_ingestion_eda",
    "stats_by_unit",
    "stats_by_signal_unit",
    "stats_by_park",
    "coverage_by_park_signal",
]


if __name__ == "__main__":
    main()
