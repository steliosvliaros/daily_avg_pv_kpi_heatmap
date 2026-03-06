"""
Gold Layer Ingestion Utilities

This module provides safe ingestion logic for gold layer tables,
preventing data duplication and ensuring datetime integrity.

Pattern: Similar to bronze/silver layers, uses watermark tracking
to avoid reprocessing already committed data.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import hashlib


class GoldIngestionConfig:
    """Configuration for gold ingestion operations."""
    
    def __init__(self, gold_root, table_name):
        self.gold_root = Path(gold_root)
        self.table_name = table_name
        self.table_dir = self.gold_root / table_name
        self.ops_dir = self.table_dir / "_ops"
        self.metadata_file = self.ops_dir / "metadata.json"
        self.watermark_file = self.ops_dir / "last_gold_committed.txt"
        self.hash_file = self.ops_dir / f"{table_name}_data_hash.txt"
        
        # Ensure directories exist
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.ops_dir.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        return f"GoldIngestionConfig(table={self.table_name}, dir={self.table_dir})"


def read_watermark(config: GoldIngestionConfig) -> pd.Timestamp:
    """
    Read the last committed date watermark.
    
    Returns
    -------
    pd.Timestamp
        Last committed date, or pd.Timestamp.min if no watermark exists
    """
    if not config.watermark_file.exists():
        return pd.Timestamp.min
    
    try:
        date_str = config.watermark_file.read_text(encoding="utf-8").strip()
        if not date_str:
            return pd.Timestamp.min
        # Parse timestamp with timezone if present (ISO format preserves tz)
        return pd.to_datetime(date_str, utc=False)
    except Exception as e:
        print(f"[gold_ingest] Warning: Failed to read watermark from {config.watermark_file}: {e}")
        return pd.Timestamp.min


def write_watermark(config: GoldIngestionConfig, last_date: pd.Timestamp) -> None:
    """
    Write the last committed date watermark.
    
    Parameters
    ----------
    config : GoldIngestionConfig
        Configuration object
    last_date : pd.Timestamp
        Last processed date to write
    """
    # Write as ISO format string to preserve timezone if present
    config.watermark_file.write_text(last_date.isoformat(), encoding="utf-8")


def read_hash(config: GoldIngestionConfig) -> str:
    """
    Read the last committed data hash.
    
    Returns
    -------
    str
        Last committed hash, or empty string if no hash exists
    """
    if not config.hash_file.exists():
        return ""
    
    try:
        return config.hash_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[gold_ingest] Warning: Failed to read hash from {config.hash_file}: {e}")
        return ""


def write_hash(config: GoldIngestionConfig, data_hash: str) -> None:
    """
    Write the data hash.
    
    Parameters
    ----------
    config : GoldIngestionConfig
        Configuration object
    data_hash : str
        Hash to write
    """
    config.hash_file.write_text(data_hash, encoding="utf-8")


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the DataFrame content for change detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash
        
    Returns
    -------
    str
        SHA256 hex digest of the DataFrame content
    """
    # Create a deterministic string representation
    # Use index, columns, and values
    parts = [
        str(df.index.tolist()),
        str(df.columns.tolist()),
        str(df.shape),
    ]
    
    # Sample some data to avoid huge memory usage
    if len(df) > 1000:
        sample = df.iloc[::len(df)//1000]
    else:
        sample = df
    
    parts.append(str(sample.values.tolist()))
    
    content = "|".join(parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def ingest_daily_energy_safe(
    daily_energy_df: pd.DataFrame,
    config: GoldIngestionConfig,
    force_full_replace: bool = False,
    debug: bool = False,
) -> dict:
    """
    Safely ingest daily energy DataFrame to gold layer.
    
    This function prevents duplicate ingestion by tracking the last committed date.
    Only new dates (beyond the watermark) are appended to the existing data.
    
    Parameters
    ----------
    daily_energy_df : pd.DataFrame
        Daily energy DataFrame with DatetimeIndex and park_id columns
    config : GoldIngestionConfig
        Gold ingestion configuration
    force_full_replace : bool, default False
        If True, replace entire dataset (ignore watermark)
    debug : bool, default False
        If True, print diagnostic information
        
    Returns
    -------
    dict
        Ingestion result with keys:
        - status: "appended", "replaced", "skipped", or "error"
        - rows_written: number of new rows written
        - last_date: last date in the committed data
        - message: human-readable message
    """
    if daily_energy_df.empty:
        return {
            "status": "skipped",
            "rows_written": 0,
            "last_date": None,
            "message": "Input DataFrame is empty"
        }
    
    # Ensure index is datetime
    if not isinstance(daily_energy_df.index, pd.DatetimeIndex):
        return {
            "status": "error",
            "rows_written": 0,
            "last_date": None,
            "message": f"Index must be DatetimeIndex, got {type(daily_energy_df.index)}"
        }
    
    # Sort by index
    daily_energy_df = daily_energy_df.sort_index()
    
    # Determine output path
    output_path = config.table_dir / "daily_energy.parquet"
    
    if force_full_replace:
        # Full replace mode
        daily_energy_df.to_parquet(output_path)
        last_date = daily_energy_df.index.max()
        write_watermark(config, last_date)
        
        if debug:
            print(f"[gold_ingest] Full replace: {len(daily_energy_df)} rows, last_date={last_date}")
        
        return {
            "status": "replaced",
            "rows_written": len(daily_energy_df),
            "last_date": last_date,
            "message": f"Full replace: {len(daily_energy_df)} rows"
        }
    
    # Incremental append mode
    last_committed_date = read_watermark(config)
    
    # Handle timezone awareness: align watermark with data timezone
    data_tz = daily_energy_df.index.tz
    if data_tz is not None and last_committed_date.tz is None:
            # Data is tz-aware but watermark is naive
            # Special case: pd.Timestamp.min cannot be localized (causes underflow)
            if last_committed_date == pd.Timestamp.min:
                # Use a safe minimum date for the data's timezone
                last_committed_date = pd.Timestamp('1900-01-01', tz=data_tz)
            else:
                last_committed_date = last_committed_date.tz_localize(data_tz)
    
    if debug:
        print(f"[gold_ingest] Last committed date: {last_committed_date}")
        print(f"[gold_ingest] New data date range: {daily_energy_df.index.min()} to {daily_energy_df.index.max()}")
    
    # Filter to only new dates
    new_data = daily_energy_df[daily_energy_df.index > last_committed_date]
    
    if new_data.empty:
        if debug:
            print(f"[gold_ingest] No new data beyond watermark {last_committed_date}")
        return {
            "status": "skipped",
            "rows_written": 0,
            "last_date": last_committed_date,
            "message": f"No new data beyond {last_committed_date}"
        }
    
    # Load existing data if it exists
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            if debug:
                print(f"[gold_ingest] Loaded existing data: {len(existing_df)} rows")
            
            # Combine existing + new data
            combined_df = pd.concat([existing_df, new_data])
            
            # Remove duplicates (keep last occurrence)
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            combined_df = combined_df.sort_index()
            
            combined_df.to_parquet(output_path)
            
            last_date = combined_df.index.max()
            write_watermark(config, last_date)
            
            if debug:
                print(f"[gold_ingest] Appended {len(new_data)} new rows, total={len(combined_df)}")
            
            return {
                "status": "appended",
                "rows_written": len(new_data),
                "last_date": last_date,
                "message": f"Appended {len(new_data)} rows (total={len(combined_df)})"
            }
            
        except Exception as e:
            if debug:
                print(f"[gold_ingest] Failed to load existing data: {e}")
            # Fallback: write new data directly
            new_data.to_parquet(output_path)
            last_date = new_data.index.max()
            write_watermark(config, last_date)
            
            return {
                "status": "replaced",
                "rows_written": len(new_data),
                "last_date": last_date,
                "message": f"Failed to append, wrote {len(new_data)} new rows"
            }
    else:
        # No existing data, write new data
        new_data.to_parquet(output_path)
        last_date = new_data.index.max()
        write_watermark(config, last_date)
        
        if debug:
            print(f"[gold_ingest] Initial write: {len(new_data)} rows")
        
        return {
            "status": "appended",
            "rows_written": len(new_data),
            "last_date": last_date,
            "message": f"Initial write: {len(new_data)} rows"
        }


def ingest_pvgis_reference_safe(
    pvgis_df: pd.DataFrame,
    config: GoldIngestionConfig,
    force_replace: bool = False,
    debug: bool = False,
) -> dict:
    """
    Safely ingest PVGIS reference DataFrame to gold layer.
    
    Uses hash-based change detection. Only writes if data has changed.
    
    Parameters
    ----------
    pvgis_df : pd.DataFrame
        PVGIS reference DataFrame with DatetimeIndex and park_id columns
    config : GoldIngestionConfig
        Gold ingestion configuration
    force_replace : bool, default False
        If True, replace regardless of hash match
    debug : bool, default False
        If True, print diagnostic information
        
    Returns
    -------
    dict
        Ingestion result with keys:
        - status: "replaced", "skipped", or "error"
        - rows_written: number of rows written
        - data_hash: hash of the data
        - message: human-readable message
    """
    if pvgis_df.empty:
        return {
            "status": "skipped",
            "rows_written": 0,
            "data_hash": "",
            "message": "Input DataFrame is empty"
        }
    
    # Ensure index is datetime
    if not isinstance(pvgis_df.index, pd.DatetimeIndex):
        return {
            "status": "error",
            "rows_written": 0,
            "data_hash": "",
            "message": f"Index must be DatetimeIndex, got {type(pvgis_df.index)}"
        }
    
    # Sort by index
    pvgis_df = pvgis_df.sort_index()
    
    # Compute data hash
    new_hash = compute_dataframe_hash(pvgis_df)
    
    if debug:
        print(f"[gold_ingest] New data hash: {new_hash}")
    
    if not force_replace:
        # Check if data has changed
        last_hash = read_hash(config)
        if debug:
            print(f"[gold_ingest] Last hash: {last_hash}")
        
        if last_hash == new_hash:
            if debug:
                print(f"[gold_ingest] Data unchanged (hash match), skipping write")
            return {
                "status": "skipped",
                "rows_written": 0,
                "data_hash": new_hash,
                "message": "Data unchanged (hash match)"
            }
    
    # Write new data
    output_path = config.table_dir / "pvgis_reference.parquet"
    pvgis_df.to_parquet(output_path)
    write_hash(config, new_hash)
    
    if debug:
        print(f"[gold_ingest] Wrote {len(pvgis_df)} rows with hash {new_hash}")
    
    return {
        "status": "replaced",
        "rows_written": len(pvgis_df),
        "data_hash": new_hash,
        "message": f"Replaced with {len(pvgis_df)} rows"
    }


def load_gold_table(config: GoldIngestionConfig, debug: bool = False) -> pd.DataFrame:
    """
    Load a gold table if it exists.
    
    Parameters
    ----------
    config : GoldIngestionConfig
        Gold ingestion configuration
    debug : bool, default False
        If True, print diagnostic information
        
    Returns
    -------
    pd.DataFrame
        Gold table data, or empty DataFrame if not found
    """
    if config.table_name == "daily_energy":
        path = config.table_dir / "daily_energy.parquet"
    elif config.table_name == "pvgis_reference":
        path = config.table_dir / "pvgis_reference.parquet"
    else:
        if debug:
            print(f"[gold_ingest] Unknown table name: {config.table_name}")
        return pd.DataFrame()
    
    if not path.exists():
        if debug:
            print(f"[gold_ingest] Gold table does not exist: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(path)
        if debug:
            print(f"[gold_ingest] Loaded gold table: {len(df)} rows from {path}")
        return df
    except Exception as e:
        if debug:
            print(f"[gold_ingest] Failed to load gold table {path}: {e}")
        return pd.DataFrame()
