from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


DatasetPath = Path


def _dataset_path(bronze_root: Path, dataset_name: str = "scada_1d_signal") -> Path:
    return Path(bronze_root) / dataset_name


def list_bronze_partitions(
    bronze_root: Optional[Path] = None,
    dataset_name: str = "scada_1d_signal",
    limit: int = 10,
    verbose: bool = True,
):
    """List bronze parquet partitions with file counts and sizes."""
    dataset_path = _dataset_path(bronze_root, dataset_name)
    if not dataset_path.exists():
        if verbose:
            print(f"❌ Dataset not found: {dataset_path}")
        return []

    parquet_files = list(dataset_path.rglob("*.parquet"))
    if verbose:
        print(f"📁 Bronze dataset: {dataset_path}")
        print(f"📂 Exists: {dataset_path.exists()}")
        print(f"📄 Parquet files found: {len(parquet_files)}")

    partitions = defaultdict(list)
    for f in parquet_files:
        parts = f.parts
        year = next((p for p in parts if p.startswith("year=")), "year=?")
        month = next((p for p in parts if p.startswith("month=")), "month=?")
        key = f"{year}/{month}"
        partitions[key].append(f)

    summary = []
    for key, files in partitions.items():
        total_mb = sum(p.stat().st_size for p in files) / (1024 ** 2)
        summary.append((key, len(files), total_mb))

    summary = sorted(summary, key=lambda x: x[0])
    if limit:
        summary = summary[-limit:]

    if verbose:
        print("\n📊 Data by partition (year/month):")
        for key, count, total_mb in summary:
            print(f"   {key}: {count} files, {total_mb:.2f} MB")

    # Show a sample file for quick visibility
    if parquet_files and verbose:
        sample_file = sorted(parquet_files)[-1]
        print("\n🔍 Sample file:")
        print(f"   Name: {sample_file.name}")
        print(f"   Size: {sample_file.stat().st_size / 1024:.2f} KB")
        print(f"   Path: .../{'/'.join(sample_file.parts[-4:])}")

    return summary


def sample_bronze_files(
    bronze_root: Optional[Path] = None,
    dataset_name: str = "scada_1d_signal",
    year: Optional[int] = None,
    limit: int = 10,
) -> List[Path]:
    """Return a list of bronze parquet files, optionally filtered by year."""
    if bronze_root is None:
        from src.config import get_config
        bronze_root = get_config().BRONZE_ROOT
    dataset_path = _dataset_path(bronze_root, dataset_name)
    if not dataset_path.exists():
        return []

    parquet_files = list(dataset_path.rglob("*.parquet"))
    if year is not None:
        parquet_files = [f for f in parquet_files if f"year={year}" in str(f)]
    parquet_files = sorted(parquet_files)
    if limit:
        parquet_files = parquet_files[:limit]
    return parquet_files


def load_bronze_sample(
    bronze_root: Optional[Path] = None,
    dataset_name: str = "scada_1d_signal",
    year: Optional[int] = None,
    n_files: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a sample of bronze parquet files into a DataFrame."""
    if bronze_root is None:
        from src.config import get_config
        bronze_root = get_config().BRONZE_ROOT
    files = sample_bronze_files(bronze_root, dataset_name=dataset_name, year=year, limit=n_files)
    if verbose:
        yr = year if year is not None else "any"
        print(f"📂 Loading {len(files)} files for year={yr} from {dataset_name} ...")
    if not files:
        if verbose:
            print("❌ No files found for requested filters")
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as exc:  # pragma: no cover - logging only
            if verbose:
                print(f"⚠️  Skipped {f.name}: {exc}")

    if not dfs:
        if verbose:
            print("❌ No data could be loaded")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True, sort=False)
    if verbose:
        print(f"\n📊 Loaded {len(df):,} rows; shape={df.shape}")
        print(f"🔍 Columns: {len(df.columns)}")
    return df


def describe_bronze(df: pd.DataFrame) -> None:
    """Print info and basic statistics for a bronze sample."""
    if df is None or df.empty:
        print("❌ No data to describe")
        return

    print("📊 DataFrame Info:")
    print("=" * 80)
    df.info(memory_usage="deep")

    print("\n\n📈 Basic Statistics:")
    print("=" * 80)
    with pd.option_context("display.max_columns", 20):
        print(df.describe(include="all"))


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing-value counts and percentages."""
    if df is None or df.empty:
        print("❌ No data to analyze for missing values")
        return pd.DataFrame()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing_Count": missing.values,
            "Missing_Pct": missing_pct.values,
        }
    )
    missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)

    if len(missing_df) > 0:
        print(f"\n⚠️  Found {len(missing_df)} columns with missing values:")
        print(missing_df.to_string(index=False))
    else:
        print("\n✅ No missing values found!")

    return missing_df


def summarize_parks_and_signals(df: pd.DataFrame) -> None:
    """Print park-level and signal-level summaries for bronze data."""
    if df is None or df.empty:
        print("❌ No data to summarize")
        return

    required_cols = {"park_id", "park_capacity_kwp", "value", "signal_name", "unit"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"⚠️  Missing expected columns: {sorted(missing_cols)}")

    print("🏭 Unique Parks:")
    print("=" * 80)
    if {"park_id", "park_capacity_kwp", "value", "signal_name"} <= set(df.columns):
        parks = df.groupby("park_id").agg(
            park_capacity_kwp=("park_capacity_kwp", "first"),
            records=("value", "count"),
            signals=("signal_name", "nunique"),
        ).sort_values("records", ascending=False)
        print(parks.to_string())
    else:
        print("Not enough columns to compute park summary.")

    print("\n\n📡 Unique Signals:")
    print("=" * 80)
    if {"signal_name", "unit"} <= set(df.columns):
        signals = df.groupby(["signal_name", "unit"]).size().reset_index(name="count")
        signals = signals.sort_values("count", ascending=False)
        print(signals.to_string(index=False))
    else:
        print("Not enough columns to compute signal summary.")

    if "park_id" in df.columns:
        print("\n\n📊 Summary:")
        print(f"   • Total parks: {df['park_id'].nunique()}")
    if "signal_name" in df.columns:
        print(f"   • Total signals: {df['signal_name'].nunique()}")
    print(f"   • Total measurements: {len(df)}")
    if {"park_id", "park_capacity_kwp"} <= set(df.columns):
        missing_cap = df[df["park_capacity_kwp"].isna()]["park_id"].nunique()
        print(f"   • Parks with missing capacity: {missing_cap}")


__all__ = [
    "list_bronze_partitions",
    "sample_bronze_files",
    "load_bronze_sample",
    "describe_bronze",
    "analyze_missing_values",
    "summarize_parks_and_signals",
    "run_bronze_inspection",
]


def run_bronze_inspection(
    bronze_root: Optional[Path] = None,
    dataset_name: str = "scada_1d_signal",
    year: Optional[int] = None,
    n_files: int = 10,
    partition_limit: int = 10,
    show_partitions: bool = True,
    show_schema: bool = True,
    show_missing: bool = True,
    show_summary: bool = True,
    verbose: bool = True,
):
    """One-call bronze inspection with selectable outputs.

    Returns a dict with loaded DataFrame, missing-values table, partition summary, and sampled file list.
    """
    if bronze_root is None:
        from src.config import get_config
        bronze_root = get_config().BRONZE_ROOT
    result = {
        "partitions": None,
        "files": None,
        "df": pd.DataFrame(),
        "missing": pd.DataFrame(),
    }

    if show_partitions:
        result["partitions"] = list_bronze_partitions(
            bronze_root,
            dataset_name=dataset_name,
            limit=partition_limit,
            verbose=verbose,
        )

    files = sample_bronze_files(
        bronze_root,
        dataset_name=dataset_name,
        year=year,
        limit=n_files,
    )
    result["files"] = files

    df = load_bronze_sample(
        bronze_root,
        dataset_name=dataset_name,
        year=year,
        n_files=n_files,
        verbose=verbose,
    )
    result["df"] = df

    if not df.empty:
        if show_schema:
            describe_bronze(df)
        if show_missing:
            result["missing"] = analyze_missing_values(df)
        if show_summary:
            summarize_parks_and_signals(df)
    else:
        if verbose:
            print("❌ No bronze data loaded; skipping schema/missing/summary.")

    return result
