from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_DATASET_NAME = "scada_1d_signal"
DEFAULT_SILVER_COLUMNS = [
    "ts_utc",
    "ts_local",
    "interval_start_date",
    "park_id",
    "park_capacity_kwp",
    "signal_name",
    "unit",
    "value",
    "source_file",
    "source_file_hash",
    "run_id",
    "ingested_at_utc",
    "ingest_key",
    "prepared_at_utc",
]
FLAG_COLUMNS = [
    "flag_missing_required",
    "flag_invalid_value",
    "flag_invalid_unit_range",
    "flag_duplicate",
]


@dataclass
class PrepStats:
    rows_in: int = 0
    rows_missing_required: int = 0
    rows_invalid_value: int = 0
    rows_deduped: int = 0
    rows_out: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "rows_in": self.rows_in,
            "rows_missing_required": self.rows_missing_required,
            "rows_invalid_value": self.rows_invalid_value,
            "rows_deduped": self.rows_deduped,
            "rows_out": self.rows_out,
        }


def load_new_bronze_parts_from_runlogs(
    bronze_root: Path,
    silver_watermark_path: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load parquet parts from new runs using run logs as the source of truth.
    Uses silver watermark to determine what has been committed to silver.
    Does NOT update watermark - use commit_silver_watermark() after successful silver write.
    
    Returns:
        Tuple of (DataFrame, list of run_ids loaded)
    """
    runlog_dir = bronze_root / "_ops" / "run_logs"
    if not runlog_dir.exists():
        return pd.DataFrame(), []

    # Read last committed run_id from silver watermark
    last_committed_run_id = "00000000T000000Z"
    if silver_watermark_path.exists():
        last_committed_run_id = silver_watermark_path.read_text(encoding="utf-8").strip() or last_committed_run_id

    runlog_files = sorted(runlog_dir.glob("run_*.json"))
    # Load everything > last committed (allows retry if silver write failed)
    new_runlogs = [p for p in runlog_files if p.stem.replace("run_", "") > last_committed_run_id]
    if not new_runlogs:
        return pd.DataFrame(), []

    parquet_files: List[str] = []
    loaded_run_ids: List[str] = []
    for p in new_runlogs:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("dataset") != dataset_name:
            continue
        run_id = payload.get("run_id")
        if run_id:
            loaded_run_ids.append(run_id)
        
        # Handle both list and integer formats for files_written
        # (legacy bronze logs stored count as int, newer logs store list of files)
        files_written = payload.get("files_written", [])
        if isinstance(files_written, int):
            # Integer count: discover files by scanning dataset directory for recent parquet files
            # This handles legacy runlogs that only stored the count, not the file list
            dataset_path = bronze_root / dataset_name
            if dataset_path.exists():
                recent_files = sorted(dataset_path.rglob("*.parquet"), key=lambda x: x.stat().st_mtime, reverse=True)
                # Take the most recent N files where N = files_written count
                discovered_files = [str(f) for f in recent_files[:files_written]]
                parquet_files.extend(discovered_files)
        elif isinstance(files_written, list):
            # Already a list of file paths
            parquet_files.extend(files_written)

    if not parquet_files:
        return pd.DataFrame(), []

    # Read parquet files, skipping those that don't exist (e.g., after reset)
    print(f"[load_new_bronze_parts] Reading {len(parquet_files)} parquet file(s)...")
    dfs = []
    missing_files = []
    for idx, f in enumerate(parquet_files, 1):
        file_path = Path(f)
        if file_path.exists():
            print(f"  [{idx}/{len(parquet_files)}] Reading {file_path.name}...", end=" ", flush=True)
            df_read = pd.read_parquet(f)
            print(f"({len(df_read):,} rows)")
            dfs.append(df_read)
        else:
            missing_files.append(f)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} parquet file(s) referenced in run logs but not found (likely from reset):")
        for f in missing_files[:3]:  # Show first 3
            print(f"  {f}")
        if len(missing_files) > 3:
            print(f"  ... and {len(missing_files) - 3} more")
    
    if not dfs:
        return pd.DataFrame(), []

    # Concatenate dataframes sequentially with garbage collection
    # to avoid memory accumulation from large datasets
    import gc
    import time
    print(f"[load_new_bronze_parts] Concatenating {len(dfs)} dataframes...")
    df_new = dfs[0]
    total_rows = len(df_new)
    
    for idx, df in enumerate(dfs[1:], 2):
        rows_before = len(df_new)
        t0 = time.time()
        df_new = pd.concat([df_new, df], ignore_index=True)
        elapsed = time.time() - t0
        total_rows += len(df)
        print(f"  [{idx}/{len(dfs)}] Merged ({rows_before:,} → {len(df_new):,} rows) in {elapsed:.1f}s")
        del df
        gc.collect()
    
    print(f"[load_new_bronze_parts] ✓ Total rows: {total_rows:,}")
    return df_new, loaded_run_ids


def commit_silver_watermark(
    silver_watermark_path: Path,
    run_ids: List[str],
) -> None:
    """
    Commit silver watermark after successful silver stage write.
    Only call this after data has been safely written to silver.
    
    Args:
        silver_watermark_path: Path to silver watermark file
        run_ids: List of run_ids that were successfully committed
    """
    if not run_ids:
        return
    
    newest_run_id = max(run_ids)
    silver_watermark_path.parent.mkdir(parents=True, exist_ok=True)
    silver_watermark_path.write_text(newest_run_id, encoding="utf-8")


def read_last_ingested_stage(silver_root: Path) -> Optional[str]:
    """Return the last ingested stage file name if tracked."""
    marker_path = silver_root / "_ops" / "last_ingested_stage.txt"
    if not marker_path.exists():
        return None
    value = marker_path.read_text(encoding="utf-8").strip()
    return value or None


def write_last_ingested_stage(silver_root: Path, stage_path: Path) -> None:
    """Persist the last ingested stage file name for fast reruns."""
    marker_path = silver_root / "_ops" / "last_ingested_stage.txt"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(stage_path.name, encoding="utf-8")


def load_park_metadata(
    metadata_file: Optional[Path],
) -> Optional[pd.DataFrame]:
    """
    Load park metadata dimension (separate from silver fact table).
    Use this to enrich silver data at query time with park attributes.
    
    Args:
        metadata_file: Path to park_metadata.csv
        
    Returns:
        DataFrame with park_id and all metadata columns, or None if file missing
    """
    if not metadata_file or not metadata_file.exists():
        return None
    
    df = pd.read_csv(metadata_file)
    if "park_id" not in df.columns:
        raise ValueError("park_metadata.csv must contain 'park_id' column")
    
    df["park_id"] = df["park_id"].astype("string").str.strip().str.lower()
    # Keep latest version per park (in case of updates)
    df = df.drop_duplicates(subset=["park_id"], keep="last")
    
    return df


def ingest_silver_stage(
    stage_path: Path,
    silver_root: Path,
    *,
    max_invalid_pct: float = 20.0,
    compression: str = "zstd",
) -> Dict[str, object]:
    """
    Ingest a silver-staged parquet file into the persistent silver layer.
    
    Quality gates:
    - Schema validation: required columns must exist
    - Invalid row rate: aborts if flag_* indicates > max_invalid_pct invalid rows
    - Deduplication: removes rows already ingested (by ingest_key + run_id)
    - Partitioning: writes by year/month for efficient querying
    - Idempotency: re-running same run_id is safe (deduped)
    
    Args:
        stage_path: Path to silver_stage_*.parquet file
        silver_root: Root directory for partitioned silver output (e.g., silver/)
        max_invalid_pct: Abort if invalid rows > this % (default 20%)
        compression: Parquet compression (default zstd)
        
    Returns:
        Dict with ingestion metadata: rows_ingested, rows_deduped, quality_report, etc.
    """
    result = {
        "success": False,
        "stage_file": str(stage_path),
        "rows_input": 0,
        "rows_deduped": 0,
        "rows_ingested": 0,
        "invalid_pct": 0.0,
        "quality_gate_passed": False,
        "errors": [],
    }
    
    # Load staged file with chunked reading to avoid memory overflow
    if not stage_path.exists():
        result["errors"].append(f"Stage file not found: {stage_path}")
        return result
    
    import pyarrow.parquet as pq
    import time
    
    try:
        print(f"[ingest] Starting ingestion from stage: {stage_path}")

        # Read parquet file metadata first
        parquet_file = pq.ParquetFile(stage_path)
        result["rows_input"] = parquet_file.metadata.num_rows
        
        if result["rows_input"] == 0:
            result["errors"].append("Stage file is empty")
            return result
        
        # Verify schema early
        required_cols = ["ts_utc", "park_id", "signal_name", "value"]
        schema_names = set(parquet_file.schema.names)
        missing_cols = [c for c in required_cols if c not in schema_names]
        if missing_cols:
            result["errors"].append(f"Missing required columns: {missing_cols}")
            return result
        
        # Build list of existing dedup keys once
        silver_root.mkdir(parents=True, exist_ok=True)
        existing_keys = set()
        dedupe_cols = []
        if "ingest_key" in schema_names:
            dedupe_cols.append("ingest_key")
        if "run_id" in schema_names:
            dedupe_cols.append("run_id")
        
        if dedupe_cols:
            existing_files = list(silver_root.glob("**/part-*.parquet"))
            print(f"[ingest] Dedup columns: {dedupe_cols}")
            print(f"[ingest] Scanning {len(existing_files):,} existing silver parts for dedupe keys")
            scan_start = time.perf_counter()
            for idx, parquet_file_existing in enumerate(existing_files, 1):
                try:
                    existing = pd.read_parquet(parquet_file_existing, columns=dedupe_cols)
                    existing_keys.update(existing[dedupe_cols].apply(tuple, axis=1).unique())
                except Exception:
                    pass
                if idx == 1 or idx % 50 == 0 or idx == len(existing_files):
                    elapsed = time.perf_counter() - scan_start
                    print(
                        f"[ingest] Dedup scan {idx:,}/{len(existing_files):,} files; "
                        f"keys: {len(existing_keys):,}; {elapsed:.1f}s"
                    )
        else:
            print("[ingest] No dedupe columns found in stage; ingestion may not be idempotent")
        
        # Process parquet file in batches
        chunk_size = 500_000  # Read ~500K rows at a time
        batch_reader = parquet_file.iter_batches(batch_size=chunk_size)
        total_batches = max(1, (result["rows_input"] + chunk_size - 1) // chunk_size)
        print(f"[ingest] Stage rows: {result['rows_input']:,} | batch size: {chunk_size:,} | batches: {total_batches:,}")
        
        total_invalid = 0
        total_rows = 0
        dup_count = 0
        
        for batch_idx, batch in enumerate(batch_reader, 1):
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"[ingest] Processing batch {batch_idx:,}/{total_batches:,}...")
            batch_start = time.perf_counter()
            df_batch = batch.to_pandas()
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"[ingest] Batch {batch_idx:,}: to_pandas in {time.perf_counter() - batch_start:.1f}s")
            
            # Ensure year/month columns exist
            if "year" not in df_batch.columns or "month" not in df_batch.columns:
                if "interval_start_date" in df_batch.columns:
                    dt_col = pd.to_datetime(df_batch["interval_start_date"], errors="coerce")
                else:
                    dt_col = pd.to_datetime(df_batch["ts_utc"], errors="coerce", utc=True)
                df_batch["year"] = dt_col.dt.year.astype("Int64")
                df_batch["month"] = dt_col.dt.month.astype("Int64")
            
            # Quality check: count invalid rows
            flag_cols = [c for c in df_batch.columns if c.startswith("flag_")]
            if flag_cols:
                invalid_mask = df_batch[flag_cols].any(axis=1)
                total_invalid += int(invalid_mask.sum())
            
            # Deduplication for this batch
            if dedupe_cols and existing_keys:
                dedupe_start = time.perf_counter()
                current_keys = df_batch[dedupe_cols].apply(tuple, axis=1)
                dup_mask = current_keys.isin(existing_keys)
                dup_count += int(dup_mask.sum())
                df_batch = df_batch[~dup_mask].copy()
                # Add new keys to set
                new_keys = current_keys[~dup_mask]
                existing_keys.update(new_keys.unique())
                if batch_idx == 1 or batch_idx % 10 == 0:
                    print(f"[ingest] Batch {batch_idx:,}: dedupe in {time.perf_counter() - dedupe_start:.1f}s")
            
            if not df_batch.empty:
                total_rows += len(df_batch)
                
                # Write batch partitioned by year/month
                write_start = time.perf_counter()
                for (year, month), group in df_batch.groupby(["year", "month"], dropna=False):
                    year_val = int(year) if pd.notna(year) else 0
                    month_val = int(month) if pd.notna(month) else 0
                    
                    partition_dir = silver_root / f"year={year_val}" / f"month={month_val}"
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Use timestamp for part filename
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    part_file = partition_dir / f"part-{timestamp}.parquet"
                    
                    group.to_parquet(part_file, index=False, compression=compression)
                if batch_idx == 1 or batch_idx % 10 == 0:
                    print(f"[ingest] Batch {batch_idx:,}: write in {time.perf_counter() - write_start:.1f}s")
            
            del df_batch
            if batch_idx == total_batches or batch_idx % 10 == 0:
                print(f"[ingest] Progress: {total_rows:,} rows ingested so far; {dup_count:,} deduped; {total_invalid:,} invalid")
        
        # Quality gate check
        if result["rows_input"] > 0:
            invalid_pct = (total_invalid / result["rows_input"]) * 100
            result["invalid_pct"] = round(invalid_pct, 2)
            
            if invalid_pct > max_invalid_pct:
                result["errors"].append(
                    f"Quality gate failed: {invalid_pct:.1f}% invalid rows > {max_invalid_pct}% threshold"
                )
                return result
            result["quality_gate_passed"] = True
        else:
            result["quality_gate_passed"] = True
        
        if total_rows == 0:
            result["errors"].append("No new rows after deduplication/quality filtering")
            return result
        
        result["rows_ingested"] = total_rows
        result["rows_deduped"] = dup_count
        result["success"] = True
        result["silver_root"] = str(silver_root)
        write_last_ingested_stage(silver_root, stage_path)
        
    except Exception as e:
        import traceback
        result["errors"].append(f"Failed during chunked ingestion: {e}")
        traceback.print_exc()
        return result
    
    return result


def _normalize_string_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    df[col] = df[col].astype("string").str.strip().str.lower()


def _ensure_interval_start_date(df: pd.DataFrame) -> None:
    if "interval_start_date" in df.columns:
        interval = pd.to_datetime(df["interval_start_date"], errors="coerce").dt.date
        df["interval_start_date"] = interval
        return

    if "ts_local" in df.columns and df["ts_local"].notna().any():
        interval = pd.to_datetime(df["ts_local"], errors="coerce").dt.date
    else:
        interval = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True).dt.date
    df["interval_start_date"] = interval


def _duplicate_masks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if df.empty:
        empty = pd.Series(False, index=df.index)
        return empty, empty

    if "ingest_key" in df.columns:
        subset = ["ingest_key"]
    else:
        subset = [c for c in ["park_id", "ts_utc", "signal_name", "unit", "source_file_hash"] if c in df.columns]

    if not subset:
        empty = pd.Series(False, index=df.index)
        return empty, empty

    dup_any = df.duplicated(subset=subset, keep=False)
    dup_drop = df.duplicated(subset=subset, keep="first")
    return dup_any, dup_drop


def _load_unit_benchmarks(
    unit_benchmarks: Optional[pd.DataFrame],
    unit_benchmarks_path: Optional[Path],
) -> Optional[pd.DataFrame]:
    if unit_benchmarks is None and unit_benchmarks_path is None:
        return None

    if unit_benchmarks is not None:
        df = unit_benchmarks.copy()
    else:
        df = pd.read_csv(unit_benchmarks_path)

    if "unit" not in df.columns:
        return None

    df["unit"] = df["unit"].astype("string").str.strip().str.lower()
    if "min_allowed" in df.columns:
        df["min_allowed"] = pd.to_numeric(df["min_allowed"], errors="coerce")
    else:
        df["min_allowed"] = pd.NA
    if "max_allowed" in df.columns:
        df["max_allowed"] = pd.to_numeric(df["max_allowed"], errors="coerce")
    else:
        df["max_allowed"] = pd.NA

    return df[["unit", "min_allowed", "max_allowed"]].drop_duplicates(subset=["unit"], keep="last")


def clean_bronze_for_silver(
    df: pd.DataFrame,
    *,
    allowed_signals: Optional[Iterable[str]] = None,
    allowed_units: Optional[Iterable[str]] = None,
    dedupe: bool = True,
    keep_invalid: bool = False,
    unit_benchmarks: Optional[pd.DataFrame] = None,
    unit_benchmarks_path: Optional[Path] = None,
    keep_columns: Optional[Iterable[str]] = None,
    inplace: bool = True,  # Work in-place by default to avoid memory spikes with large DataFrames
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean bronze long-form data and return a silver-ready frame plus stats.
    When keep_invalid is True, no rows are dropped; invalid/duplicate rows are flagged instead.
    If unit_benchmarks or unit_benchmarks_path is provided, values outside min/max are flagged.
    
    Parameters
    ----------
    inplace : bool, default True
        If True, modifies DataFrame in-place to avoid memory spikes.
        If False, creates a copy first (safer but uses more memory).
    """
    stats = PrepStats()
    if df.empty:
        return df.copy(), stats.as_dict()

    stats.rows_in = len(df)
    
    # Only copy if explicitly requested (for memory efficiency with large DataFrames)
    if not inplace:
        df = df.copy()

    required = ["ts_utc", "park_id", "signal_name", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for silver prep: {missing}")

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    if "ts_local" in df.columns:
        df["ts_local"] = pd.to_datetime(df["ts_local"], errors="coerce")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["value"] = df["value"].replace([float("inf"), float("-inf")], pd.NA)

    if "park_capacity_kwp" in df.columns:
        df["park_capacity_kwp"] = pd.to_numeric(df["park_capacity_kwp"], errors="coerce")

    _normalize_string_col(df, "park_id")
    _normalize_string_col(df, "signal_name")
    _normalize_string_col(df, "unit")

    missing_required = df["ts_utc"].isna() | df["park_id"].isna() | df["signal_name"].isna()
    invalid_value = df["value"].isna()
    stats.rows_missing_required = int(missing_required.sum())
    stats.rows_invalid_value = int(invalid_value.sum())

    if keep_invalid:
        dup_any, dup_drop = _duplicate_masks(df)
        df["flag_missing_required"] = missing_required
        df["flag_invalid_value"] = invalid_value
        df["flag_duplicate"] = dup_any
        unit_bounds = _load_unit_benchmarks(unit_benchmarks, unit_benchmarks_path)
        if unit_bounds is not None and "unit" in df.columns:
            bounds = unit_bounds.set_index("unit")
            min_allowed = df["unit"].map(bounds["min_allowed"])
            max_allowed = df["unit"].map(bounds["max_allowed"])
            low_mask = min_allowed.notna() & (df["value"] < min_allowed)
            high_mask = max_allowed.notna() & (df["value"] > max_allowed)
            df["flag_invalid_unit_range"] = low_mask | high_mask
        else:
            df["flag_invalid_unit_range"] = False
    else:
        keep_mask = ~(missing_required | invalid_value)
        df = df[keep_mask].copy()

        if allowed_signals is not None:
            allowed = {s.strip().lower() for s in allowed_signals if s and str(s).strip()}
            if allowed:
                df = df[df["signal_name"].isin(allowed)].copy()

        if allowed_units is not None and "unit" in df.columns:
            allowed = {u.strip().lower() for u in allowed_units if u and str(u).strip()}
            if allowed:
                df = df[df["unit"].isin(allowed)].copy()

    _ensure_interval_start_date(df)

    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["interval_start_date"], errors="coerce").dt.year.astype("Int64")
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["interval_start_date"], errors="coerce").dt.month.astype("Int64")

    if not keep_invalid and dedupe:
        _, dup_drop = _duplicate_masks(df)
        stats.rows_deduped = int(dup_drop.sum())
        df = df[~dup_drop].copy()

    df["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    df = df.sort_values(["ts_utc", "park_id", "signal_name"], kind="mergesort")

    if keep_columns is None:
        keep_columns = list(DEFAULT_SILVER_COLUMNS)
    else:
        keep_columns = list(keep_columns)

    if keep_invalid:
        for col in FLAG_COLUMNS:
            if col not in keep_columns:
                keep_columns.append(col)

    keep_columns = [c for c in keep_columns if c in df.columns]
    if keep_columns:
        df = df[keep_columns].copy()

    stats.rows_out = len(df)
    return df, stats.as_dict()


def write_silver_stage(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    filename: Optional[str] = None,
    compression: str = "zstd",
    retention: str = "last_n",
    retain_n: int = 3,
) -> Optional[Path]:
    """
    Write a silver-ready parquet stage file for downstream ingestion.

    Retention modes:
    - keep:    Keep all stage files (no pruning)
    - latest:  Write to a stable filename (silver_stage_latest.parquet) and remove other stage files
    - last_n:  Keep the last N stage files by modified time (default retain_n=3)
    """
    if df.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        if retention == "latest":
            filename = "silver_stage_latest.parquet"
        else:
            run_id = None
            if "run_id" in df.columns:
                run_ids = [r for r in df["run_id"].dropna().unique() if str(r).strip()]
                if run_ids:
                    run_id = sorted(run_ids)[-1]
            if not run_id:
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"silver_stage_{run_id}.parquet"

    out_path = output_dir / filename
    df.to_parquet(out_path, index=False, compression=compression)

    # Apply retention policy
    mode = (retention or "keep").strip().lower()
    try:
        if mode == "keep":
            return out_path
        
        # List all stage files
        stage_files = sorted(output_dir.glob("silver_stage_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if mode == "latest":
            # Remove all except the stable latest filename
            for p in stage_files:
                if p.name != "silver_stage_latest.parquet":
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
            return out_path
        
        if mode == "last_n":
            n = max(1, int(retain_n))
            # Keep the N most recent files; delete the rest
            to_delete = stage_files[n:]
            for p in to_delete:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            return out_path
    finally:
        # Fall-through returns the written path
        ...

    return out_path


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_SILVER_COLUMNS",
    "FLAG_COLUMNS",
    "load_new_bronze_parts_from_runlogs",
    "commit_silver_watermark",
    "read_last_ingested_stage",
    "write_last_ingested_stage",
    "clean_bronze_for_silver",
    "write_silver_stage",
    "load_park_metadata",
    "ingest_silver_stage",
]
