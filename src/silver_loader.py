"""
Silver Data Loading Utilities

This module provides functions for loading and processing silver layer data
and PVGIS typical year data in wide format.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def _normalize_values(values):
    """Normalize a collection of values to lowercase strings."""
    if values is None:
        return None
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _normalize_status(value):
    """Normalize status value(s) to lowercase strings."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    if isinstance(value, bool):
        return {"true"} if value else {"false"}
    return {str(value).strip().lower()}


def _iter_silver_parts(silver_root, start_date=None, end_date=None):
    """Iterate over silver parquet partition files within date range."""
    root = Path(silver_root)
    if start_date is None and end_date is None:
        return sorted(root.glob("year=*/month=*/part-*.parquet"))

    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None
    if start is None and end is not None:
        start = end
    if start is not None and end is None:
        end = pd.Timestamp.utcnow().tz_localize(None)
    if start is None or end is None:
        return sorted(root.glob("year=*/month=*/part-*.parquet"))
    if start > end:
        start, end = end, start

    months = pd.period_range(start=start, end=end, freq="M")
    files = []
    for period in months:
        part_dir = root / f"year={period.year}" / f"month={period.month}"
        if part_dir.exists():
            files.extend(part_dir.glob("part-*.parquet"))
    return sorted(files)


def _read_parquet_columns(path, columns):
    """Read parquet file with specified columns, fallback to all columns."""
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)


def _align_timestamp(ts, series):
    """Align timestamp timezone with series timezone."""
    if ts is None:
        return None
    ts = pd.to_datetime(ts)
    series_tz = series.dt.tz
    if series_tz is None:
        if ts.tzinfo is not None:
            return ts.tz_convert(None)
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize(series_tz)
    return ts.tz_convert(series_tz)


def _replace_year_safe(ts, ref_year):
    """Replace year in timestamp, handling Feb 29 edge case."""
    if ts is None:
        return None
    ts = pd.to_datetime(ts)
    try:
        return ts.replace(year=int(ref_year))
    except ValueError:
        if ts.month == 2 and ts.day == 29:
            return ts.replace(year=int(ref_year), day=28)
        raise


def load_silver_wide(
    silver_root,
    metadata_path,
    park_ids=None,
    signals=None,
    start_date=None,
    end_date=None,
    status_effective="true",
    only_valid=True,
    flatten_columns=False,
    debug=False,
):
    """
    Load silver data in wide format (time series × parks).

    Parameters
    ----------
    silver_root : Path or str
        Root directory of silver layer (contains year=*/month=* partitions)
    metadata_path : Path or str
        Path to park_metadata.csv file
    park_ids : list-like, optional
        Filter to specific park IDs (case-insensitive)
    signals : list-like, optional
        Filter to specific signal names (case-insensitive)
    start_date : str or Timestamp, optional
        Start date for filtering (inclusive)
    end_date : str or Timestamp, optional
        End date for filtering (inclusive)
    status_effective : str or list-like, default "true"
        Filter parks by status_effective from metadata. Use "all" or None to disable filtering.
    only_valid : bool, default True
        If True, exclude rows with any flag_* columns set
    flatten_columns : bool, default False
        If True, flatten MultiIndex columns to strings with '__' separator
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by ts_local (or ts_utc), 
        columns are (park_id, signal_name, unit) tuples or flattened strings
    """
    from src.silver_prepair import load_park_metadata
    
    park_set = _normalize_values(park_ids)
    signal_set = _normalize_values(signals)
    # Default to active parks; allow opt-out with "all"/None
    if isinstance(status_effective, str) and status_effective.strip().lower() == "all":
        status_set = None
    else:
        status_set = _normalize_status(status_effective)

    part_files = _iter_silver_parts(silver_root, start_date=start_date, end_date=end_date)
    if debug:
        print(f"[load_silver_wide] part files: {len(part_files)}")
    if not part_files:
        return pd.DataFrame()

    base_cols = ["ts_local", "ts_utc", "park_id", "signal_name", "unit", "value"]
    flag_cols = ["flag_missing_required", "flag_invalid_value", "flag_invalid_unit_range", "flag_duplicate"]
    use_cols = base_cols + flag_cols

    frames = [_read_parquet_columns(p, use_cols) for p in part_files]
    df = pd.concat(frames, ignore_index=True)
    if debug:
        print(f"[load_silver_wide] rows loaded: {len(df)}")
    if df.empty:
        return df

    if "park_id" in df.columns:
        df["park_id"] = df["park_id"].astype("string").str.strip().str.lower()
    if "signal_name" in df.columns:
        df["signal_name"] = df["signal_name"].astype("string").str.strip().str.lower()

    if debug:
        park_sample = df["park_id"].dropna().unique().tolist()[:5]
        signal_sample = df["signal_name"].dropna().unique().tolist()[:5]
        print(f"[load_silver_wide] sample park_ids: {park_sample}")
        print(f"[load_silver_wide] sample signals: {signal_sample}")

    if status_set is not None:
        meta = load_park_metadata(Path(metadata_path))
        if meta is None or "status_effective" not in meta.columns:
            raise ValueError("status_effective filter requested but metadata missing status_effective")
        status_series = meta["status_effective"].astype("string").str.strip().str.lower()
        allowed_parks = set(meta.loc[status_series.isin(status_set), "park_id"].astype(str).str.lower())
        if debug:
            intersection = set(df["park_id"].dropna()) & allowed_parks
            print(f"[load_silver_wide] status_effective parks: {len(allowed_parks)}")
            print(f"[load_silver_wide] status_effective intersection: {len(intersection)}")
        if park_set:
            allowed_parks = allowed_parks.intersection(park_set)
        df = df[df["park_id"].isin(allowed_parks)]
        if debug:
            print(f"[load_silver_wide] rows after status_effective: {len(df)}")

    if park_set is not None:
        df = df[df["park_id"].isin(park_set)]
        if debug:
            print(f"[load_silver_wide] rows after park filter: {len(df)}")

    if signal_set is not None:
        df = df[df["signal_name"].isin(signal_set)]
        if debug:
            print(f"[load_silver_wide] rows after signal filter: {len(df)}")

    time_col = "ts_local" if "ts_local" in df.columns else "ts_utc"
    if time_col == "ts_local":
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if df[time_col].isna().all() and "ts_utc" in df.columns:
            time_col = "ts_utc"
    if time_col == "ts_utc":
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    start_ts = _align_timestamp(start_date, df[time_col]) if start_date is not None else None
    end_ts = _align_timestamp(end_date, df[time_col]) if end_date is not None else None
    if start_ts is not None:
        df = df[df[time_col] >= start_ts]
        if debug:
            print(f"[load_silver_wide] rows after start_date: {len(df)}")
    if end_ts is not None:
        df = df[df[time_col] <= end_ts]
        if debug:
            print(f"[load_silver_wide] rows after end_date: {len(df)}")

    if only_valid:
        if "value" in df.columns:
            df = df[df["value"].notna()]
        flag_cols = [c for c in df.columns if c.startswith("flag_")]
        if flag_cols:
            df = df[~df[flag_cols].fillna(False).any(axis=1)]
        if debug:
            print(f"[load_silver_wide] rows after validity filter: {len(df)}")

    if df.empty:
        return df

    col_keys = ["park_id", "signal_name"]
    if "unit" in df.columns:
        col_keys.append("unit")

    wide = df.pivot_table(
        index=time_col,
        columns=col_keys,
        values="value",
        aggfunc="mean",
    ).sort_index()

    if flatten_columns and isinstance(wide.columns, pd.MultiIndex):
        wide.columns = ["__".join([str(x) for x in col]) for col in wide.columns]

    if debug:
        print(f"[load_silver_wide] wide shape: {wide.shape}")
    return wide


def _resolve_pvgis_files(pvgis_path=None, cache_dir=None, workspace_root=None):
    """Resolve paths to PVGIS parquet files."""
    candidates = []
    if workspace_root:
        root = Path(workspace_root)
        candidates.append(root / "outputs" / "pvgis_typical_year" / "pvgis_typical_daily.parquet")
        candidates.append(root / "pvgis" / "pvgis_cache" / "typical_daily")

    if pvgis_path:
        candidates.append(Path(pvgis_path))
    if cache_dir:
        candidates.append(Path(cache_dir))

    candidates.append(Path("outputs") / "pvgis_typical_year" / "pvgis_typical_daily.parquet")
    candidates.append(Path("pvgis") / "pvgis_cache" / "typical_daily")

    files = []
    for item in candidates:
        if item.is_dir():
            files.extend(sorted(item.glob("*.parquet")))
        elif item.exists():
            files.append(item)

    return files


def load_pvgis_typical_wide(
    pvgis_path=None,
    cache_dir=None,
    workspace_root=None,
    metadata_path=None,
    park_ids=None,
    signals=None,
    start_date=None,
    end_date=None,
    status_effective="true",
    flatten_columns=False,
    local_timezone=None,
    debug=False,
):
    """
    Load PVGIS typical year data in wide format.

    Parameters
    ----------
    pvgis_path : Path or str, optional
        Direct path to PVGIS parquet file or directory
    cache_dir : Path or str, optional
        Directory containing PVGIS cache files
    workspace_root : Path or str, optional
        Workspace root to search for PVGIS data
    metadata_path : Path or str, optional
        Path to park_metadata.csv; defaults to workspace_root/mappings/park_metadata.csv
    park_ids : list-like, optional
        Filter to specific park IDs
    signals : list-like, optional
        Filter to specific signal names
    start_date : str or Timestamp, optional
        Start date for filtering (matched by month-day)
    end_date : str or Timestamp, optional
        End date for filtering (matched by month-day)
    status_effective : str or list-like, default "true"
        Filter parks by status_effective from metadata. Use "all" or None to disable filtering.
    flatten_columns : bool, default False
        If True, flatten MultiIndex columns to strings
    local_timezone : str, optional
        Timezone for ts_local column
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by ts_local
    """
    # Use the new load_pvgis function and convert to wide
    df = load_pvgis(
        pvgis_path=pvgis_path,
        cache_dir=cache_dir,
        workspace_root=workspace_root,
        metadata_path=metadata_path,
        park_ids=park_ids,
        signals=signals,
        start_date=start_date,
        end_date=end_date,
        status_effective=status_effective,
        local_timezone=local_timezone,
        debug=debug,
    )
    print(df.columns)
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert to wide
    col_keys = ["park_id", "signal_name"]
    if "unit" in df.columns:
        col_keys.append("unit")

    wide = df.pivot_table(
        index="ts_local",
        columns=col_keys,
        values="value",
        aggfunc="mean",
    ).sort_index()

    if flatten_columns and isinstance(wide.columns, pd.MultiIndex):
        wide.columns = ["__".join([str(x) for x in col]) for col in wide.columns]

    if debug:
        print(f"[load_pvgis_typical_wide] wide shape: {wide.shape}")
    return wide


def load_pvgis(
    pvgis_path=None,
    cache_dir=None,
    workspace_root=None,
    metadata_path=None,
    park_ids=None,
    signals=None,
    start_date=None,
    end_date=None,
    status_effective="true",
    local_timezone=None,
    debug=False,
):
    """
    Load PVGIS typical year data in long format.

    Parameters
    ----------
    pvgis_path : Path or str, optional
        Direct path to PVGIS parquet file or directory
    cache_dir : Path or str, optional
        Directory containing PVGIS cache files
    workspace_root : Path or str, optional
        Workspace root to search for PVGIS data
    metadata_path : Path or str, optional
        Path to park_metadata.csv; defaults to workspace_root/mappings/park_metadata.csv
    park_ids : list-like, optional
        Filter to specific park IDs (case-insensitive)
    signals : list-like, optional
        Filter to specific signal names (case-insensitive)
    start_date : str or Timestamp, optional
        Start date for filtering (matched by month-day)
    end_date : str or Timestamp, optional
        End date for filtering (matched by month-day)
    status_effective : str or list-like, default "true"
        Filter parks by status_effective from metadata. Use "all" or None to disable filtering.
    local_timezone : str, optional
        Timezone for ts_local column
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: interval_start_date, ts_utc, ts_local, 
        park_id, park_name, signal_name, unit, value, timezone
    """
    park_set = _normalize_values(park_ids)
    signal_set = _normalize_values(signals)
    if isinstance(status_effective, str) and status_effective.strip().lower() == "all":
        status_set = None
    else:
        status_set = _normalize_status(status_effective)

    files = _resolve_pvgis_files(pvgis_path, cache_dir, workspace_root)
    if debug:
        print(f"[load_pvgis] files: {len(files)}")
    if not files:
        return pd.DataFrame()

    cols = ["interval_start_date", "ts_utc", "park_id", "park_name", "signal_name", "unit", "value", "timezone"]
    frames = [_read_parquet_columns(p, cols) for p in files]
    df = pd.concat(frames, ignore_index=True)
    if debug:
        print(f"[load_pvgis] rows loaded: {len(df)}")
    if df.empty:
        return df

    df["park_id"] = df["park_id"].astype("string").str.strip().str.lower()
    df["signal_name"] = df["signal_name"].astype("string").str.strip().str.lower()

    if status_set is not None:
        from src.silver_prepair import load_park_metadata

        meta_path = Path(metadata_path) if metadata_path is not None else None
        if meta_path is None and workspace_root is not None:
            meta_path = Path(workspace_root) / "mappings" / "park_metadata.csv"
        if meta_path is None:
            raise ValueError("status_effective filter requested but metadata_path not provided")

        meta = load_park_metadata(meta_path)
        if meta is None or "status_effective" not in meta.columns:
            raise ValueError("status_effective filter requested but metadata missing status_effective")

        status_series = meta["status_effective"].astype("string").str.strip().str.lower()
        allowed_parks = set(meta.loc[status_series.isin(status_set), "park_id"].astype(str).str.lower())
        if debug:
            print(f"[load_pvgis] status_effective parks: {len(allowed_parks)}")
        if park_set:
            allowed_parks = allowed_parks.intersection(park_set)
        df = df[df["park_id"].isin(allowed_parks)]
        if debug:
            print(f"[load_pvgis] rows after status_effective: {len(df)}")

    if park_set is not None:
        df = df[df["park_id"].isin(park_set)]
        if debug:
            print(f"[load_pvgis] rows after park filter: {len(df)}")

    if signal_set is not None:
        df = df[df["signal_name"].isin(signal_set)]
        if debug:
            print(f"[load_pvgis] rows after signal filter: {len(df)}")

    df["ts_local"] = pd.to_datetime(df.get("interval_start_date"), errors="coerce")
    if local_timezone:
        df["ts_local"] = df["ts_local"].dt.tz_localize(local_timezone)
    elif "timezone" in df.columns:
        tz_vals = df["timezone"].dropna().unique().tolist()
        if len(tz_vals) == 1:
            df["ts_local"] = df["ts_local"].dt.tz_localize(tz_vals[0])

    if start_date is not None and end_date is None:
        end_date = pd.Timestamp.utcnow().tz_localize(None)

    ref_year = None
    if df["ts_local"].notna().any():
        years = df["ts_local"].dt.year.dropna().unique()
        if len(years) == 1:
            ref_year = int(years[0])
    if ref_year is not None:
        start_date = _replace_year_safe(start_date, ref_year) if start_date is not None else None
        end_date = _replace_year_safe(end_date, ref_year) if end_date is not None else None
        if debug:
            print(f"[load_pvgis] ref_year: {ref_year}")

    start_ts = _align_timestamp(start_date, df["ts_local"]) if start_date is not None else None
    end_ts = _align_timestamp(end_date, df["ts_local"]) if end_date is not None else None
    if start_ts is not None:
        df = df[df["ts_local"] >= start_ts]
        if debug:
            print(f"[load_pvgis] rows after start_date: {len(df)}")
    if end_ts is not None:
        df = df[df["ts_local"] <= end_ts]
        if debug:
            print(f"[load_pvgis] rows after end_date: {len(df)}")

    if debug:
        print(f"[load_pvgis] Final shape: {df.shape}")

    return df


def divide_wide_by_reference(
    measured_wide,
    reference_wide,
    join_type="left",
    match_by_calendar_day=True,
    multiply_by_100=True,
    min_reference_value=None,
    output_range=None,
    debug=False,
):
    """
    Divide measured wide data by reference wide data, aligning by calendar day.
    
    Useful for:
    - Calculating performance ratio: (measured / pvgis_typical) * 100
    - Comparing real vs expected: (actual / planned) * 100
    - Normalizing by reference: (data / reference_model)

    Parameters
    ----------
    measured_wide : pd.DataFrame
        Wide-format measured/actual data (many years)
    reference_wide : pd.DataFrame
        Wide-format reference/expected data (typically one year or typical year)
    join_type : str, default 'left'
        How to align: 'left', 'inner', 'outer', 'right'
    match_by_calendar_day : bool, default True
        If True, match by month-day (for typical year alignment)
    multiply_by_100 : bool, default True
        If True, return as percentage (0-100 for parity)
    min_reference_value : float, optional
        Set ratios to NaN where reference is below this threshold
    output_range : tuple, optional
        Clip output to (min, max) range
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Ratio DataFrame (measured / reference) with same structure as measured_wide

    Examples
    --------
    >>> wide = load_silver_filtered_wide(silver_root, start_date="2025-01-01")
    >>> wide_pvgis = load_pvgis_filtered_wide(workspace_root)
    >>> ratio = divide_wide_by_reference(wide, wide_pvgis, multiply_by_100=True)
    >>> # ratio contains 100 where actual==expected, 150 where actual is 50% higher, etc.
    """
    if measured_wide is None or reference_wide is None or measured_wide.empty or reference_wide.empty:
        if debug:
            print("[divide_wide_by_reference] Invalid inputs")
        return pd.DataFrame()
    
    measured = measured_wide.copy()
    reference = reference_wide.copy()
    
    if debug:
        print(f"[divide_wide_by_reference] Input: measured={measured.shape}, reference={reference.shape}")
        print(f"[divide_wide_by_reference] Measured index tz: {measured.index.tz}")
        print(f"[divide_wide_by_reference] Reference index tz: {reference.index.tz}")
    
    # Align column structures: if both have MultiIndex columns, match by park_id (level 0)
    # This handles case where signal names differ (e.g., 'pcc_active_energy_export' vs 'pvgis_expected_daily_kwh')
    if isinstance(measured.columns, pd.MultiIndex) and isinstance(reference.columns, pd.MultiIndex):
        measured_parks = measured.columns.get_level_values(0)
        reference_parks = reference.columns.get_level_values(0)
        common_parks = set(measured_parks) & set(reference_parks)

        if common_parks:
            measured = measured.loc[:, measured_parks.isin(common_parks)]
            reference = reference.loc[:, reference_parks.isin(common_parks)]

            # Align reference columns to measured by park_id only (level 0)
            reference_cols_by_park = {}
            for col in reference.columns:
                park_id = col[0]
                if park_id not in reference_cols_by_park:
                    reference_cols_by_park[park_id] = col

            aligned_reference_cols = []
            for col in measured.columns:
                park_id = col[0]
                ref_col = reference_cols_by_park.get(park_id)
                if ref_col is not None:
                    aligned_reference_cols.append(ref_col)

            if aligned_reference_cols:
                reference = reference[aligned_reference_cols]
                reference.columns = measured.columns[:len(aligned_reference_cols)]

            if debug:
                print(f"[divide_wide_by_reference] Aligned columns to common parks: {len(common_parks)}")
                print(f"[divide_wide_by_reference] After alignment: measured={measured.shape}, reference={reference.shape}")
    
    # Remove timezone if present for matching
    if hasattr(measured.index, 'tz') and measured.index.tz is not None:
        measured.index = measured.index.tz_localize(None)
    if hasattr(reference.index, 'tz') and reference.index.tz is not None:
        reference.index = reference.index.tz_localize(None)
    
    # Align indices by calendar day (month-day) for typical year
    if match_by_calendar_day and isinstance(reference.index, pd.DatetimeIndex):
        ref_years = reference.index.year.unique()
        if len(ref_years) == 1:
            if debug:
                print(f"[divide_wide_by_reference] Typical year detected (year {ref_years[0]}), matching by month-day...")
            
            # Create month-day strings for matching
            measured_mday = measured.index.strftime("%m-%d")
            reference_mday = reference.index.strftime("%m-%d")
            
            # Build a reference dict: month-day -> values
            reference_by_mday = {}
            for mday in reference_mday.unique():
                idx = reference_mday == mday
                reference_by_mday[mday] = reference.loc[idx].iloc[0] if idx.sum() > 0 else None
            
            # Create new reference with same index as measured
            reference_aligned = pd.DataFrame(index=measured.index, columns=reference.columns, dtype=float)
            for i, mday in enumerate(measured_mday):
                if mday in reference_by_mday and reference_by_mday[mday] is not None:
                    reference_aligned.iloc[i] = reference_by_mday[mday]
            
            reference = reference_aligned
            
            if debug:
                print(f"[divide_wide_by_reference] Reference aligned to measured dates")
                print(f"[divide_wide_by_reference] Reference non-null after alignment: {reference.notna().sum().sum()}")
        else:
            # Multi-year reference, just align by index
            measured, reference = measured.align(reference, join=join_type, axis=0)
    else:
        # No calendar matching, just align indices
        measured, reference = measured.align(reference, join=join_type, axis=0)
    
    if debug:
        print(f"[divide_wide_by_reference] After alignment: {measured.shape}")
    
    # Apply minimum threshold to reference
    if min_reference_value is not None:
        reference_masked = reference.where(reference >= min_reference_value, other=np.nan)
    else:
        reference_masked = reference
    
    # Calculate ratio
    ratio = measured / reference_masked
    
    # Multiply by 100 if requested (for percentage)
    if multiply_by_100:
        ratio = ratio * 100
    
    # Clip to range if specified
    if output_range is not None:
        ratio = ratio.clip(lower=output_range[0], upper=output_range[1])
    
    if debug:
        non_null_count = ratio.notna().sum().sum()
        total_count = ratio.shape[0] * ratio.shape[1]
        pct = (non_null_count / total_count * 100) if total_count > 0 else 0
        print(f"[divide_wide_by_reference] Output: {ratio.shape}, non-null: {pct:.1f}%")
        print(f"\n{ratio.describe()}")
    
    return ratio


def calculate_power_ratio_percent(
    measured_wide,
    expected_wide,
    join_type="left",
    min_expected_kwh=None,
    output_range=None,
    match_by_calendar_day=True,
    debug=True,
):
    """
    Calculate power ratio percentage: (measured / expected) * 100.
    
    Matches parks by extracting park_id from column names (before first __ separator).
    For typical year data (PVGIS): matches by calendar day (month/day).

    Parameters
    ----------
    measured_wide : pd.DataFrame
        Wide-format measured data (from silver)
    expected_wide : pd.DataFrame
        Wide-format expected data (from PVGIS)
    join_type : str, default 'left'
        Join type for alignment ('left', 'inner', 'outer', 'right')
    min_expected_kwh : float, optional
        Minimum expected value threshold (values below are set to NaN)
    output_range : tuple, optional
        (min, max) range for output clipping
    match_by_calendar_day : bool, default True
        If True, match typical year by month-day
    debug : bool, default True
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Power ratio in percent (100 = perfect match)
    """
    import re
    
    if measured_wide is None or expected_wide is None or measured_wide.empty or expected_wide.empty:
        if debug:
            print("[calculate_power_ratio_percent] Invalid inputs")
        return pd.DataFrame()
    
    left = measured_wide.copy()
    right = expected_wide.copy()
    
    if debug:
        print(f"[calculate_power_ratio_percent] Input: left={left.shape}, right={right.shape}")
    
    # Normalize park ID by removing capacity suffix and prefixes
    def normalize_park_id(park_id):
        park_id = str(park_id).lower().strip()
        # Remove leading 'p_' prefix
        park_id = re.sub(r'^p_', '', park_id)
        # Remove capacity patterns like _123_kwp or _123kwp anywhere in the string
        park_id = re.sub(r'_\d+_?kwp', '', park_id)
        return park_id
    
    # Extract park IDs from column names
    def extract_park_ids(df):
        parks = {}
        for col in df.columns:
            if isinstance(col, tuple):
                park_raw = str(col[0]).split("__")[0]
            else:
                park_raw = str(col).split("__")[0]
            
            park_id = normalize_park_id(park_raw)
            if park_id not in parks:
                parks[park_id] = []
            parks[park_id].append(col)
        return parks
    
    left_parks = extract_park_ids(left)
    right_parks = extract_park_ids(right)
    common_parks = set(left_parks.keys()) & set(right_parks.keys())
    
    if debug:
        print(f"  Left parks: {len(left_parks)}, Right parks: {len(right_parks)}, Common: {len(common_parks)}")
    
    if len(common_parks) == 0:
        return pd.DataFrame()
    
    # Match columns: prefer pcc_active_energy_export for measured, pvgis_expected for expected
    left_cols = []
    right_cols = []
    
    for park in sorted(common_parks):
        left_candidates = left_parks[park]
        right_candidates = right_parks[park]
        
        left_energy = [c for c in left_candidates if "pcc_active" in str(c).lower()]
        if not left_energy:
            left_energy = [c for c in left_candidates if "energy" in str(c).lower() and "irradiance" not in str(c).lower()]
        
        right_energy = [c for c in right_candidates if "pvgis" in str(c).lower() and ("expected" in str(c).lower() or "kwh" in str(c).lower())]
        
        if debug:
            print(f"    Park '{park}': left candidates={len(left_candidates)}, right candidates={len(right_candidates)}")
            print(f"      Left energy matches: {len(left_energy)}")
            print(f"      Right energy matches: {len(right_energy)}")
        
        if left_energy and right_energy:
            left_cols.append(left_energy[0])
            right_cols.append(right_energy[0])
            if debug:
                print(f"      ✓ Matched: {left_energy[0]} <-> {right_energy[0]}")
    
    if debug:
        print(f"  Matched columns: {len(left_cols)}")
    
    if len(left_cols) == 0:
        return pd.DataFrame()
    
    left = left[left_cols].copy()
    right = right[right_cols].copy()
    right.columns = left.columns  # Rename to match
    
    # Align by calendar day if typical year
    if match_by_calendar_day and isinstance(right.index, pd.DatetimeIndex):
        right_years = right.index.year.unique()
        if len(right_years) == 1:
            if debug:
                print(f"  Typical year detected, matching by calendar day...")
            
            left_mday = left.index.strftime("%m-%d")
            right_mday = right.index.strftime("%m-%d")
            
            right_aligned = pd.DataFrame(index=left.index, columns=right.columns, dtype=float)
            for col in right.columns:
                s = right[col].copy()
                s.index = right_mday
                s = s.groupby(level=0).mean()
                right_aligned[col] = s.reindex(left_mday, method='ffill').values
            right = right_aligned
    
    # Final alignment
    left, right = left.align(right, join=join_type, axis=0)
    
    if min_expected_kwh is not None:
        right = right.where(right >= min_expected_kwh, other=np.nan)
    
    ratio = (left / right) * 100.0
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    
    if output_range:
        ratio = ratio.clip(lower=output_range[0], upper=output_range[1])
    else:
        ratio = ratio.where((ratio >= 0) & (ratio <= 300), other=np.nan)
    
    if debug:
        print(f"  Output: {ratio.shape}, non-null: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1] or 1) * 100:.1f}%")
    
    return ratio


def load_silver(
    silver_root,
    start_date=None,
    end_date=None,
    park_ids=None,
    signals=None,
    flags="valid",
    columns="all",
    debug=False,
):
    """
    Load silver data in long format (one row per timestamp/park/signal).

    Parameters
    ----------
    silver_root : Path or str
        Root directory of silver layer (contains year=*/month=* partitions)
    start_date : str or Timestamp, optional
        Start date for filtering (inclusive)
    end_date : str or Timestamp, optional
        End date for filtering (inclusive)
    park_ids : list-like, optional
        Filter to specific park IDs (case-insensitive)
    signals : list-like, optional
        Filter to specific signal names (case-insensitive)
    flags : str, default "valid"
        Filter by flag status:
        - "valid": only rows without any flags set
        - "invalid" or "flagged": only rows with at least one flag set
        - "all": return all rows regardless of flags
    columns : str, default "all"
        Select column subset:
        - "measurement": only measurement/business columns (ts_utc, ts_local, park_id, signal_name, value, unit, etc.)
        - "quality" or "lineage": only data quality/pipeline columns (source_file, run_id, flags, etc.)
        - "all": return all columns
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: ts_utc, ts_local, park_id, signal_name, 
        signal_value, unit, and optional flag_* columns
    """
    park_set = _normalize_values(park_ids)
    signal_set = _normalize_values(signals)
    flags_filter = str(flags).strip().lower() if flags is not None else "valid"
    columns_filter = str(columns).strip().lower() if columns is not None else "all"
    
    # Get partition files within date range
    part_files = _iter_silver_parts(silver_root, start_date, end_date)
    
    if not part_files:
        if debug:
            print(f"[load_silver] No partition files found in {silver_root} for range {start_date} to {end_date}")
        return pd.DataFrame()
    
    if debug:
        print(f"[load_silver] Loading {len(part_files)} partition file(s)")
    
    # Load and concatenate all partitions
    dfs = []
    for pf in part_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            if debug:
                print(f"[load_silver] Warning: Failed to read {pf}: {e}")
            continue
    
    if not dfs:
        if debug:
            print(f"[load_silver] No data loaded from partitions")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    if debug:
        print(f"[load_silver] rows loaded: {len(df)}")
    
    # Apply date filters
    if start_date is not None:
        start = _align_timestamp(start_date, df["ts_utc"])
        df = df[df["ts_utc"] >= start]
        if debug:
            print(f"[load_silver] rows after start_date: {len(df)}")
    
    if end_date is not None:
        end = _align_timestamp(end_date, df["ts_utc"])
        df = df[df["ts_utc"] <= end]
        if debug:
            print(f"[load_silver] rows after end_date: {len(df)}")
    
    # Apply park filter
    if park_set is not None:
        df = df[df["park_id"].astype(str).str.lower().isin(park_set)]
        if debug:
            print(f"[load_silver] rows after park filter: {len(df)}")
    
    # Apply signal filter
    if signal_set is not None:
        df = df[df["signal_name"].astype(str).str.lower().isin(signal_set)]
        if debug:
            print(f"[load_silver] rows after signal filter: {len(df)}")
    
    # Apply flag filter
    flag_cols = [col for col in df.columns if col.startswith("flag_")]
    if flag_cols and flags_filter != "all":
        if flags_filter == "valid":
            df = df[~df[flag_cols].any(axis=1)]
            if debug:
                print(f"[load_silver] rows after flag filter (valid only): {len(df)}")
        elif flags_filter in ("invalid", "flagged"):
            df = df[df[flag_cols].any(axis=1)]
            if debug:
                print(f"[load_silver] rows after flag filter (flagged only): {len(df)}")
    
    # Apply column filter
    if columns_filter != "all":
        measurement_cols = [
            "ts_utc", "ts_local", "interval_start_date", "park_id", 
            "park_capacity_kwp", "signal_name", "unit", "value",
        ]
        quality_cols = [
            "source_file", "source_file_hash", "run_id", "ingested_at_utc",
            "ingest_key", "prepared_at_utc", "year", "month",
        ] + flag_cols
        
        if columns_filter == "measurement":
            keep_cols = [c for c in measurement_cols if c in df.columns]
            df = df[keep_cols]
            if debug:
                print(f"[load_silver] columns after measurement filter: {len(df.columns)}")
        elif columns_filter in ("quality", "lineage"):
            # Keep park_id and timestamp for context
            context_cols = ["ts_utc", "park_id", "signal_name"]
            keep_cols = [c for c in context_cols + quality_cols if c in df.columns]
            df = df[keep_cols]
            if debug:
                print(f"[load_silver] columns after quality filter: {len(df.columns)}")
    
    if debug:
        print(f"[load_silver] Final shape: {df.shape}")
    
    return df


def filter_silver(
    df,
    park_id_contains=None,
    park_capacity_min=None,
    park_capacity_max=None,
    signal_name_contains=None,
    units=None,
    debug=False,
):
    """
    Filter silver DataFrame with flexible criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Silver DataFrame to filter (from load_silver)
    park_id_contains : str, optional
        Substring to match in park_id (case-insensitive). Use None or "all" to include all.
    park_capacity_min : float, optional
        Minimum park capacity (inclusive)
    park_capacity_max : float, optional
        Maximum park capacity (inclusive)
    signal_name_contains : str, optional
        Substring to match in signal_name (case-insensitive)
    units : str or list-like, optional
        Unit(s) to filter (exact match, case-insensitive)
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame

    Examples
    --------
    >>> df = load_silver(silver_root, start_date="2025-01-01", end_date="2025-12-31")
    >>> df_filtered = filter_silver(
    ...     df,
    ...     park_id_contains="energeiaki",
    ...     park_capacity_min=1000,
    ...     park_capacity_max=5000,
    ...     signal_name_contains="power",
    ...     units=["kW", "kWh"]
    ... )
    """
    result = df.copy()
    initial_rows = len(result)
    
    # Filter by park_id contains
    if park_id_contains is not None and str(park_id_contains).strip().lower() not in ("all", ""):
        if "park_id" in result.columns:
            mask = result["park_id"].astype(str).str.lower().str.contains(
                str(park_id_contains).strip().lower(), 
                case=False, 
                na=False
            )
            result = result[mask]
            if debug:
                print(f"[filter_silver] rows after park_id contains '{park_id_contains}': {len(result)} (removed {initial_rows - len(result)})")
                initial_rows = len(result)
    
    # Filter by park capacity min
    if park_capacity_min is not None and "park_capacity_kwp" in result.columns:
        result = result[result["park_capacity_kwp"] >= park_capacity_min]
        if debug:
            print(f"[filter_silver] rows after capacity >= {park_capacity_min}: {len(result)} (removed {initial_rows - len(result)})")
            initial_rows = len(result)
    
    # Filter by park capacity max
    if park_capacity_max is not None and "park_capacity_kwp" in result.columns:
        result = result[result["park_capacity_kwp"] <= park_capacity_max]
        if debug:
            print(f"[filter_silver] rows after capacity <= {park_capacity_max}: {len(result)} (removed {initial_rows - len(result)})")
            initial_rows = len(result)
    
    # Filter by signal_name contains
    if signal_name_contains is not None and str(signal_name_contains).strip().lower() not in ("all", ""):
        if "signal_name" in result.columns:
            mask = result["signal_name"].astype(str).str.lower().str.contains(
                str(signal_name_contains).strip().lower(),
                case=False,
                na=False
            )
            result = result[mask]
            if debug:
                print(f"[filter_silver] rows after signal_name contains '{signal_name_contains}': {len(result)} (removed {initial_rows - len(result)})")
                initial_rows = len(result)
    
    # Filter by units
    if units is not None and "unit" in result.columns:
        if isinstance(units, str):
            units = [units]
        units_lower = {str(u).strip().lower() for u in units}
        mask = result["unit"].astype(str).str.lower().str.strip().isin(units_lower)
        result = result[mask]
        if debug:
            print(f"[filter_silver] rows after unit in {list(units)}: {len(result)} (removed {initial_rows - len(result)})")
    
    if debug:
        print(f"[filter_silver] Final shape: {result.shape}")
    
    return result


def to_wide(
    df,
    timestamp_col="ts_utc",
    flatten_columns=False,
    debug=False,
):
    """
    Convert long-format silver DataFrame to wide format.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame (from load_silver or filter_silver) with columns:
        ts_utc/ts_local, park_id, signal_name, value, unit
    timestamp_col : str, default "ts_utc"
        Column to use as index. Options: "ts_utc", "ts_local"
    flatten_columns : bool, default False
        If True, flatten MultiIndex columns to strings with '__' separator.
        If False, keep as MultiIndex (park_id, signal_name, unit)
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by timestamp_col, columns are (park_id, signal_name, unit) 
        tuples or flattened strings

    Examples
    --------
    >>> df = load_silver(silver_root, start_date="2025-01-01", columns="measurement")
    >>> df_filtered = filter_silver(df, park_id_contains="energeiaki")
    >>> wide = to_wide(df_filtered, timestamp_col="ts_utc", flatten_columns=True)
    """
    if df.empty:
        if debug:
            print("[to_wide] Empty DataFrame, returning empty wide format")
        return pd.DataFrame()
    
    # Auto-detect timestamp column if not in DataFrame
    if timestamp_col not in df.columns:
        if "ts_utc" in df.columns:
            timestamp_col = "ts_utc"
        elif "ts_local" in df.columns:
            timestamp_col = "ts_local"
        else:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found. Available: {list(df.columns)}")
    
    # Check required columns
    required_cols = [timestamp_col, "park_id", "signal_name", "value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for wide conversion: {missing}")
    
    if debug:
        print(f"[to_wide] Input shape: {df.shape}")
        print(f"[to_wide] Unique parks: {df['park_id'].nunique()}")
        print(f"[to_wide] Unique signals: {df['signal_name'].nunique()}")
    
    # Create pivot columns from park_id, signal_name, and unit (if available)
    if "unit" in df.columns:
        index_cols = ["park_id", "signal_name", "unit"]
    else:
        index_cols = ["park_id", "signal_name"]
    
    # Pivot to wide format
    try:
        wide = df.pivot_table(
            index=timestamp_col,
            columns=index_cols,
            values="value",
            aggfunc="first",  # Take first value if duplicates exist
        )
    except Exception as e:
        if debug:
            print(f"[to_wide] Pivot failed: {e}")
            print(f"[to_wide] Checking for duplicates...")
            dupes = df.groupby([timestamp_col] + index_cols).size()
            dupes = dupes[dupes > 1]
            if not dupes.empty:
                print(f"[to_wide] Found {len(dupes)} duplicate groups:")
                print(dupes.head())
        raise
    
    # Flatten columns if requested
    if flatten_columns:
        if isinstance(wide.columns, pd.MultiIndex):
            wide.columns = ["__".join(str(c) for c in col).strip("__") for col in wide.columns]
        if debug:
            print(f"[to_wide] Flattened columns to strings")
    
    if debug:
        print(f"[to_wide] Output shape: {wide.shape}")
        if isinstance(wide.columns, pd.MultiIndex):
            print(f"[to_wide] Column levels: {wide.columns.nlevels}")
        print(f"[to_wide] Index: {wide.index.name} ({len(wide)} timestamps)")
    
    return wide


def load_silver_filtered_wide(
    silver_root,
    start_date=None,
    end_date=None,
    park_ids=None,
    signals=None,
    flags="valid",
    columns="measurement",
    park_id_contains=None,
    park_capacity_min=None,
    park_capacity_max=None,
    signal_name_contains=None,
    units=None,
    timestamp_col="ts_utc",
    flatten_columns=False,
    debug=False,
):
    """
    Load, filter, and convert silver data to wide format in one call.
    
    Convenience function that combines load_silver(), filter_silver(), and to_wide().

    Parameters
    ----------
    silver_root : Path or str
        Root directory of silver layer
    start_date, end_date : str or Timestamp, optional
        Date range for loading (inclusive)
    park_ids : list-like, optional
        Exact park IDs to load (case-insensitive)
    signals : list-like, optional
        Exact signal names to load (case-insensitive)
    flags : str, default "valid"
        Filter by flag status: "valid", "invalid"/"flagged", or "all"
    columns : str, default "measurement"
        Column subset: "measurement", "quality"/"lineage", or "all"
    park_id_contains : str, optional
        Substring to match in park_id (applied after loading)
    park_capacity_min : float, optional
        Minimum park capacity in kWp
    park_capacity_max : float, optional
        Maximum park capacity in kWp
    signal_name_contains : str, optional
        Substring to match in signal_name (applied after loading)
    units : str or list-like, optional
        Unit(s) to filter (exact match)
    timestamp_col : str, default "ts_utc"
        Column to use as index. Options: "ts_utc", "ts_local"
    flatten_columns : bool, default False
        If True, flatten MultiIndex columns to strings
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by timestamp_col

    Examples
    --------
    >>> # Load all 2025 power data for parks with capacity 1000-5000 kWp
    >>> wide = load_silver_filtered_wide(
    ...     silver_root=config.SILVER_ROOT,
    ...     start_date="2025-01-01",
    ...     end_date="2025-12-31",
    ...     signal_name_contains="power",
    ...     park_capacity_min=1000,
    ...     park_capacity_max=5000,
    ...     flatten_columns=True,
    ...     debug=True
    ... )
    """
    if debug:
        print("="*80)
        print("LOAD SILVER FILTERED WIDE")
        print("="*80)
    
    # Step 1: Load data
    df = load_silver(
        silver_root=silver_root,
        start_date=start_date,
        end_date=end_date,
        park_ids=park_ids,
        signals=signals,
        flags=flags,
        columns=columns,
        debug=debug,
    )
    
    if df.empty:
        if debug:
            print("[load_silver_filtered_wide] No data loaded, returning empty DataFrame")
        return pd.DataFrame()
    
    # Step 2: Apply additional filters
    needs_filter = any([
        park_id_contains is not None,
        park_capacity_min is not None,
        park_capacity_max is not None,
        signal_name_contains is not None,
        units is not None,
    ])
    
    if needs_filter:
        df = filter_silver(
            df=df,
            park_id_contains=park_id_contains,
            park_capacity_min=park_capacity_min,
            park_capacity_max=park_capacity_max,
            signal_name_contains=signal_name_contains,
            units=units,
            debug=debug,
        )
        
        if df.empty:
            if debug:
                print("[load_silver_filtered_wide] No data after filtering, returning empty DataFrame")
            return pd.DataFrame()
    
    # Step 3: Convert to wide format
    wide = to_wide(
        df=df,
        timestamp_col=timestamp_col,
        flatten_columns=flatten_columns,
        debug=debug,
    )
    
    if debug:
        print("="*80)
        print(f"[load_silver_filtered_wide] Final wide shape: {wide.shape}")
        print("="*80)
    
    return wide


def load_pvgis_filtered_wide(
    pvgis_path=None,
    cache_dir=None,
    workspace_root=None,
    metadata_path=None,
    start_date=None,
    end_date=None,
    park_ids=None,
    signals=None,
    status_effective="true",
    park_id_contains=None,
    park_capacity_min=None,
    park_capacity_max=None,
    signal_name_contains=None,
    units=None,
    timestamp_col="ts_local",
    flatten_columns=False,
    local_timezone=None,
    debug=False,
):
    """
    Load, filter, and convert PVGIS data to wide format in one call.
    
    Convenience function that combines load_pvgis(), filter_silver(), and to_wide().

    Parameters
    ----------
    pvgis_path : Path or str, optional
        Direct path to PVGIS parquet file or directory
    cache_dir : Path or str, optional
        Directory containing PVGIS cache files
    workspace_root : Path or str, optional
        Workspace root to search for PVGIS data
    metadata_path : Path or str, optional
        Path to park_metadata.csv
    start_date : str or Timestamp, optional
        Start date for filtering (matched by month-day)
    end_date : str or Timestamp, optional
        End date for filtering (matched by month-day)
    park_ids : list-like, optional
        Exact park IDs to load (case-insensitive)
    signals : list-like, optional
        Exact signal names to load (case-insensitive)
    status_effective : str or list-like, default "true"
        Filter parks by status_effective from metadata
    park_id_contains : str, optional
        Substring to match in park_id (applied after loading)
    park_capacity_min : float, optional
        Minimum park capacity in kWp
    park_capacity_max : float, optional
        Maximum park capacity in kWp
    signal_name_contains : str, optional
        Substring to match in signal_name (applied after loading)
    units : str or list-like, optional
        Unit(s) to filter (exact match)
    timestamp_col : str, default "ts_local"
        Column to use as index. Options: "ts_utc", "ts_local"
    flatten_columns : bool, default False
        If True, flatten MultiIndex columns to strings
    local_timezone : str, optional
        Timezone for ts_local column
    debug : bool, default False
        If True, print diagnostic information

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by timestamp_col

    Examples
    --------
    >>> # Load PVGIS data for summer months, large parks only
    >>> wide = load_pvgis_filtered_wide(
    ...     workspace_root=config.WORKSPACE_ROOT,
    ...     start_date="2001-06-01",
    ...     end_date="2001-08-31",
    ...     park_id_contains="energeiaki",
    ...     park_capacity_min=1000,
    ...     timestamp_col="ts_local",
    ...     flatten_columns=True,
    ...     debug=True
    ... )
    """
    if debug:
        print("="*80)
        print("LOAD PVGIS FILTERED WIDE")
        print("="*80)
    
    # Step 1: Load PVGIS data
    df = load_pvgis(
        pvgis_path=pvgis_path,
        cache_dir=cache_dir,
        workspace_root=workspace_root,
        metadata_path=metadata_path,
        park_ids=park_ids,
        signals=signals,
        start_date=start_date,
        end_date=end_date,
        status_effective=status_effective,
        local_timezone=local_timezone,
        debug=debug,
    )
    
    if df.empty:
        if debug:
            print("[load_pvgis_filtered_wide] No data loaded, returning empty DataFrame")
        return pd.DataFrame()
    
    # Step 2: Apply additional filters
    needs_filter = any([
        park_id_contains is not None,
        park_capacity_min is not None,
        park_capacity_max is not None,
        signal_name_contains is not None,
        units is not None,
    ])
    
    if needs_filter:
        df = filter_silver(
            df=df,
            park_id_contains=park_id_contains,
            park_capacity_min=park_capacity_min,
            park_capacity_max=park_capacity_max,
            signal_name_contains=signal_name_contains,
            units=units,
            debug=debug,
        )
        
        if df.empty:
            if debug:
                print("[load_pvgis_filtered_wide] No data after filtering, returning empty DataFrame")
            return pd.DataFrame()
    
    # Step 3: Convert to wide format
    wide = to_wide(
        df=df,
        timestamp_col=timestamp_col,
        flatten_columns=flatten_columns,
        debug=debug,
    )
    
    if debug:
        print("="*80)
        print(f"[load_pvgis_filtered_wide] Final wide shape: {wide.shape}")
        print("="*80)
    
    return wide
