from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class EdaConfig:
    output_dir: Optional[Path] = None
    max_units: int = 8
    max_signals: int = 12
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


def _plot_count_bar(
    df: pd.DataFrame,
    col: str,
    title: str,
    output_dir: Optional[Path],
    save: bool,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    if df.empty or col not in df.columns:
        return None, None
    counts = df[col].value_counts().head(20)
    if counts.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 4))
    counts.sort_values(ascending=True).plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("rows")
    fig.tight_layout()

    out_path = None
    if save and output_dir is not None:
        out_path = output_dir / f"counts_{col}.png"
        fig.savefig(out_path, dpi=150)
    return fig, out_path


def _plot_unit_histograms(
    df: pd.DataFrame,
    units: List[str],
    output_dir: Optional[Path],
    sample_rows: Optional[int],
    save: bool,
) -> Tuple[List[plt.Figure], List[Path]]:
    figs: List[plt.Figure] = []
    paths: List[Path] = []
    for unit in units:
        sub = df[df["unit"] == unit]
        if sub.empty:
            continue
        sub = _maybe_sample(sub, sample_rows)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sub["value"].dropna(), bins=50, color="#4C78A8", alpha=0.85)
        ax.set_title(f"value distribution: {unit}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        fig.tight_layout()

        out_path = None
        if save and output_dir is not None:
            out_path = output_dir / f"hist_{unit}.png"
            fig.savefig(out_path, dpi=150)
            paths.append(out_path)
        figs.append(fig)
    return figs, paths


def _plot_unit_boxplot(
    df: pd.DataFrame,
    units: List[str],
    output_dir: Optional[Path],
    save: bool,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    if df.empty or not units:
        return None, None
    sub = df[df["unit"].isin(units)]
    if sub.empty:
        return None, None
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [sub[sub["unit"] == u]["value"].dropna() for u in units]
    ax.boxplot(data, labels=units, vert=False, showfliers=False)
    ax.set_title("value spread by unit")
    ax.set_xlabel("value")
    fig.tight_layout()

    out_path = None
    if save and output_dir is not None:
        out_path = output_dir / "boxplot_units.png"
        fig.savefig(out_path, dpi=150)
    return fig, out_path


def run_silver_pre_ingestion_eda(
    df: pd.DataFrame,
    config: EdaConfig,
) -> Dict[str, object]:
    df = _coerce_frame(df)
    if config.save_plots or config.save_stats:
        if config.output_dir is None:
            raise ValueError("output_dir must be set when save_plots or save_stats is True")
        config.output_dir.mkdir(parents=True, exist_ok=True)

    overview = _summary_overview(df)
    overview["computed_at_utc"] = datetime.now(timezone.utc).isoformat()

    unit_stats = stats_by_unit(df, config.quantiles)
    signal_stats = stats_by_signal_unit(df, config.quantiles, config.max_signals)
    coverage = coverage_by_park_signal(df)

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
    fig, path = _plot_count_bar(
        df,
        "unit",
        "rows by unit (top 20)",
        config.output_dir,
        config.save_plots,
    )
    if fig:
        figs.append(fig)
    if path:
        plot_paths.append(path)

    fig, path = _plot_count_bar(
        df,
        "signal_name",
        "rows by signal (top 20)",
        config.output_dir,
        config.save_plots,
    )
    if fig:
        figs.append(fig)
    if path:
        plot_paths.append(path)

    if "unit" in df.columns:
        top_units = df["unit"].value_counts().head(config.max_units).index.tolist()
        unit_figs, unit_paths = _plot_unit_histograms(
            df,
            top_units,
            config.output_dir,
            config.sample_rows,
            config.save_plots,
        )
        figs.extend(unit_figs)
        plot_paths.extend(unit_paths)
        fig, path = _plot_unit_boxplot(df, top_units, config.output_dir, config.save_plots)
        if fig:
            figs.append(fig)
        if path:
            plot_paths.append(path)

    return {
        "overview": overview,
        "unit_stats": unit_stats,
        "signal_stats": signal_stats,
        "coverage": coverage,
        "overview_path": overview_path,
        "unit_stats_path": unit_stats_path,
        "signal_stats_path": signal_stats_path,
        "coverage_path": coverage_path,
        "plots": figs,
        "plot_paths": plot_paths,
    }


def build_cfg(args) -> EdaConfig:
    return EdaConfig(
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        max_units=args.max_units,
        max_signals=args.max_signals,
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
    "coverage_by_park_signal",
]


if __name__ == "__main__":
    main()
