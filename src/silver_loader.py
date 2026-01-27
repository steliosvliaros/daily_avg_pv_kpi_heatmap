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
    status_effective=None,
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
    status_effective : str or list-like, optional
        Filter parks by status_effective from metadata
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
        allowed_parks = set(meta.loc[status_series.isin(status_set), "park_id"].astype(str))
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
    park_ids=None,
    signals=None,
    start_date=None,
    end_date=None,
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
    park_ids : list-like, optional
        Filter to specific park IDs
    signals : list-like, optional
        Filter to specific signal names
    start_date : str or Timestamp, optional
        Start date for filtering (matched by month-day)
    end_date : str or Timestamp, optional
        End date for filtering (matched by month-day)
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
    park_set = _normalize_values(park_ids)
    signal_set = _normalize_values(signals)

    files = _resolve_pvgis_files(pvgis_path, cache_dir, workspace_root)
    if debug:
        print(f"[load_pvgis_typical_wide] files: {len(files)}")
    if not files:
        return pd.DataFrame()

    cols = ["interval_start_date", "ts_utc", "park_id", "park_name", "signal_name", "unit", "value", "timezone"]
    frames = [_read_parquet_columns(p, cols) for p in files]
    df = pd.concat(frames, ignore_index=True)
    if debug:
        print(f"[load_pvgis_typical_wide] rows loaded: {len(df)}")
    if df.empty:
        return df

    df["park_id"] = df["park_id"].astype("string").str.strip().str.lower()
    df["signal_name"] = df["signal_name"].astype("string").str.strip().str.lower()

    if park_set is not None:
        df = df[df["park_id"].isin(park_set)]
        if debug:
            print(f"[load_pvgis_typical_wide] rows after park filter: {len(df)}")

    if signal_set is not None:
        df = df[df["signal_name"].isin(signal_set)]
        if debug:
            print(f"[load_pvgis_typical_wide] rows after signal filter: {len(df)}")

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
            print(f"[load_pvgis_typical_wide] ref_year: {ref_year}")

    start_ts = _align_timestamp(start_date, df["ts_local"]) if start_date is not None else None
    end_ts = _align_timestamp(end_date, df["ts_local"]) if end_date is not None else None
    if start_ts is not None:
        df = df[df["ts_local"] >= start_ts]
        if debug:
            print(f"[load_pvgis_typical_wide] rows after start_date: {len(df)}")
    if end_ts is not None:
        df = df[df["ts_local"] <= end_ts]
        if debug:
            print(f"[load_pvgis_typical_wide] rows after end_date: {len(df)}")

    if df.empty:
        return df

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
