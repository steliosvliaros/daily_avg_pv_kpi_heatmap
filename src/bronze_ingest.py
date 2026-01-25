#!/usr/bin/env python3
"""
FINAL Bronze ingestion script (folder-based, wide->long, hash-skip, archive)

What it does (daily SCADA exports):
- Watches a landing folder (data_root/inbox) for Excel/CSV exports
- Moves each file to data_root/processing (atomic move = "lock")
- Computes SHA256 file hash
- If the exact same hash was already ingested => SKIP and move to archived/duplicates
- Else:
    - Reads the file
    - Renames columns using your mapping CSV (original -> sanitized SQL-safe identifiers)
    - Converts wide -> long (keeps ALL signals)
    - Adds: source_file_hash, source_file, ingested_at_utc, run_id, deterministic ingest_key
    - Writes Bronze as a consolidated dataset partitioned by year/month:
        bronze_root/scada_1d_signal/year=YYYY/month=MM/part-run=<run_id>-hash=<hash12>.parquet
    - Registers the file hash in bronze_root/_ops/ingest_registry_files.csv
    - Moves file to data_root/archived (renamed with hash)
- If parsing/validation fails => move to data_root/rejected with a reason

Assumptions:
- Your sanitized measurement columns look like:
    <park_token>__<measurement_token>__<unit_token>
  Example:
    p_4e_energeiaki_lexaina_4472kwp__pcc_active_energy_export__u_kwh

- Timestamp column will be mapped to "ts" (recommended).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    # Data folders
    data_root: Path
    inbox: Path
    processing: Path
    archived: Path
    rejected: Path

    # Bronze folders
    bronze_root: Path
    dataset_name: str = "scada_1d_signal"
    ops_dirname: str = "_ops"
    registry_filename: str = "ingest_registry_files.csv"
    runlog_dirname: str = "run_logs"

    # Mappings (versioned)
    mappings_root: Path = None  # Will be set to data_root.parent / "mappings"
    
    # Parsing
    timezone_local: str = "Europe/Athens"
    daily_interval_end_is_midnight: bool = True

    # Parquet
    parquet_compression: str = "zstd"

    # Inbox safety
    min_age_seconds: int = 90        # only pick up files older than this (download completed)
    stable_check_seconds: int = 20   # verify size stable across this interval

    # Lock
    lockfile: Path = Path("bronze_ingest.lock")

    # Excel
    sheet_name: Optional[str] = None

    # CSV
    csv_sep: Optional[str] = None
    csv_encoding: Optional[str] = None

    # Runtime toggles
    allow_duplicates: bool = False


# -----------------------------
# Patterns
# -----------------------------

# park_id__signal_name__unit  (unit optional)
# Format uses two-step sanitization: park_id from metadata, signal_name independent
# Example: 4e_energeiaki_176_kwp_likovouni__pcc_active_energy_export__kwh
# Groups: park_id, meas (signal), unit
# Note: Capacity is embedded in park_id (e.g., 4e_energeiaki_176_kwp_likovouni includes "176_kwp")
COL_RE = re.compile(r"^(?P<park_id>.+?)__(?P<meas>.+?)__(?P<unit>[a-z0-9_]+)$")


# -----------------------------
# Basic utils
# -----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def run_id_utc() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def atomic_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dst))  # atomic if same volume

def slug_reason(s: str, max_len: int = 60) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return (s[:max_len] if s else "error")

def ensure_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
        return
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return
        except Exception:
            raise RuntimeError(
                "Parquet engine not found. Install 'pyarrow' (recommended) or 'fastparquet'."
            )


def ensure_under(target: Path, root: Path) -> None:
    """Guard destructive operations by ensuring target is within root."""
    target_res = target.resolve()
    root_res = root.resolve()
    if target_res == root_res:
        return
    if root_res not in target_res.parents:
        raise ValueError(f"Refusing to operate outside root: {target} (root={root})")


def safe_rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


# -----------------------------
# Mapping reader: original -> sanitized (with versioning support)
# -----------------------------

def get_current_mapping_path(cfg: Config) -> Path:
    """Read mappings/current.txt to find the active mapping file."""
    current_file = cfg.mappings_root / "current.txt"
    if not current_file.exists():
        raise FileNotFoundError(
            f"Mapping pointer file not found: {current_file}\n"
            f"Create {cfg.mappings_root}/current.txt with the filename of the active mapping (e.g., park_power_mapping_v001.csv)"
        )
    mapping_filename = current_file.read_text(encoding="utf-8").strip()
    mapping_path = cfg.mappings_root / mapping_filename
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Active mapping file not found: {mapping_path}\n"
            f"current.txt references: {mapping_filename}"
        )
    return mapping_path

def compute_mapping_hash(mapping_path: Path) -> str:
    """Compute SHA256 hash of mapping file for traceability."""
    sha256 = hashlib.sha256()
    with open(mapping_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def read_mapping(mapping_csv: Path) -> Dict[str, str]:
    mdf = pd.read_csv(mapping_csv)
    cols = [c.lower() for c in mdf.columns]

    if "original" in cols and "sanitized" in cols:
        ocol = mdf.columns[cols.index("original")]
        scol = mdf.columns[cols.index("sanitized")]
    else:
        if mdf.shape[1] < 2:
            raise ValueError("Mapping CSV must have at least 2 columns: original,sanitized")
        ocol, scol = mdf.columns[0], mdf.columns[1]

    return dict(zip(mdf[ocol].astype(str), mdf[scol].astype(str)))


# -----------------------------
# Registry (hash-skip)
# -----------------------------

def registry_path(cfg: Config) -> Path:
    return cfg.bronze_root / cfg.ops_dirname / cfg.registry_filename

def load_registry(cfg: Config) -> pd.DataFrame:
    rp = registry_path(cfg)
    if rp.exists():
        df = pd.read_csv(rp, dtype=str)
        if "source_file_hash" not in df.columns:
            raise ValueError(f"Registry missing 'source_file_hash': {rp}")
        return df
    return pd.DataFrame(columns=[
        "dataset", "source_file_hash", "source_file", "run_id",
        "status", "rows_long", "files_written", "archived_path",
        "mapping_filename", "mapping_file_hash",
        "ingested_at_utc", "message"
    ])

def registry_has_hash(registry_df: pd.DataFrame, dataset: str, file_hash: str) -> bool:
    if registry_df.empty:
        return False
    sub = registry_df[(registry_df["dataset"] == dataset) & (registry_df["source_file_hash"] == file_hash)]
    return not sub.empty

def append_registry_row(cfg: Config, row: Dict[str, str]) -> None:
    rp = registry_path(cfg)
    rp.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if rp.exists():
        df.to_csv(rp, mode="a", header=False, index=False)
    else:
        df.to_csv(rp, mode="w", header=True, index=False)


# -----------------------------
# Run log
# -----------------------------

def write_runlog(cfg: Config, run_id: str, payload: dict) -> None:
    d = cfg.bronze_root / cfg.ops_dirname / cfg.runlog_dirname
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run_{run_id}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# File readiness checks
# -----------------------------

def is_file_old_enough(p: Path, min_age_seconds: int) -> bool:
    try:
        mtime = p.stat().st_mtime
        age = time.time() - mtime
        return age >= min_age_seconds
    except FileNotFoundError:
        return False

def is_file_size_stable(p: Path, stable_check_seconds: int) -> bool:
    try:
        s1 = p.stat().st_size
        time.sleep(stable_check_seconds)
        s2 = p.stat().st_size
        return s1 == s2 and s2 > 0
    except FileNotFoundError:
        return False


# -----------------------------
# Read wide data file
# -----------------------------

def read_wide_file(path: Path, cfg: Config) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, sheet_name=cfg.sheet_name or 0)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, sep=cfg.csv_sep or ",", encoding=cfg.csv_encoding)
    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# Wide -> Long (all signals)
# -----------------------------

def wide_to_long(df_wide: pd.DataFrame, cfg: Config, source_file: str, source_file_hash: str, run_id: str) -> pd.DataFrame:
    # Timestamp column detection
    ts_col = None
    for c in df_wide.columns:
        if str(c).lower() == "ts":
            ts_col = c
            break
    if ts_col is None:
        for c in df_wide.columns:
            if str(c).lower() in ("timestamp", "time", "datetime"):
                ts_col = c
                break
    if ts_col is None:
        raise ValueError("No timestamp column found (expected 'ts' after mapping).")

    ts = pd.to_datetime(df_wide[ts_col], errors="coerce")
    if ts.isna().all():
        raise ValueError(f"Could not parse datetime column '{ts_col}'.")

    # Localize/convert timezone
    if ts.dt.tz is None:
        ts_local = ts.dt.tz_localize(cfg.timezone_local, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts_local = ts.dt.tz_convert(cfg.timezone_local)
    ts_utc = ts_local.dt.tz_convert("UTC")

    df = df_wide.copy()
    df["ts_local"] = ts_local
    df["ts_utc"] = ts_utc

    # Measurement columns = those matching expected token pattern
    candidate_cols = [str(c) for c in df.columns if "__" in str(c)]
    meas_cols = [c for c in candidate_cols if COL_RE.match(c)]

    if not meas_cols:
        raise ValueError("No measurement columns matched pattern: park_id__signal_name__unit")

    # Melt (keep only timestamp ids; extra id columns can be added if you want)
    id_vars = [ts_col, "ts_local", "ts_utc"]
    long_df = df.melt(id_vars=id_vars, value_vars=meas_cols, var_name="col", value_name="value")
    long_df = long_df.dropna(subset=["value"])

    extracted = long_df["col"].str.extract(COL_RE)
    long_df["park_id"] = extracted["park_id"]
    long_df["signal_name"] = extracted["meas"]
    long_df["unit"] = extracted["unit"]

    # Extract capacity from park_id (e.g., "4e_energeiaki_176_kwp_likovouni" -> 176)
    # Look for pattern: digits followed by "_kwp" anywhere in the park_id
    capacity_match = extracted["park_id"].str.extract(r"(\d+(?:\.\d+)?)_kwp", expand=False)
    long_df["park_capacity_kwp"] = pd.to_numeric(capacity_match, errors="coerce")

    # Determine interval_start_date for daily values
    ts_local_series = pd.to_datetime(long_df["ts_local"])
    if cfg.daily_interval_end_is_midnight:
        interval_start_date = (ts_local_series.dt.floor("D") - pd.Timedelta(days=1)).dt.date
    else:
        interval_start_date = ts_local_series.dt.date

    long_df["interval_start_date"] = interval_start_date
    long_df["year"] = pd.to_datetime(interval_start_date).dt.year.astype(int)
    long_df["month"] = pd.to_datetime(interval_start_date).dt.month.astype(int)

    ingested_at = utc_now().isoformat()
    long_df["source_file"] = source_file
    long_df["source_file_hash"] = source_file_hash
    long_df["run_id"] = run_id
    long_df["ingested_at_utc"] = ingested_at

    # Deterministic ingest_key (includes file hash so corrections create new rows)
    def make_key(r) -> str:
        base = f"{r['park_id']}|{r['ts_utc']}|{r['signal_name']}|{r.get('unit') or ''}|{source_file_hash}"
        return sha256_text(base)

    long_df["ingest_key"] = long_df.apply(make_key, axis=1)

    keep = [
        "ts_local", "ts_utc",
        "interval_start_date", "year", "month",
        "park_id", "park_capacity_kwp",
        "signal_name", "unit",
        "value",
        "source_file", "source_file_hash",
        "run_id", "ingested_at_utc",
        "ingest_key",
    ]
    return long_df[keep]


# -----------------------------
# Bronze writer: consolidated dataset partitioned by year/month
# (one parquet per (year,month) per source file/run)
# -----------------------------

def write_bronze_monthly(long_df: pd.DataFrame, cfg: Config, run_id: str, file_hash: str) -> List[str]:
    ensure_parquet_engine()
    base_dir = cfg.bronze_root / cfg.dataset_name
    written: List[str] = []

    for (y, m), g in long_df.groupby(["year", "month"], dropna=False):
        out_dir = base_dir / f"year={int(y):04d}" / f"month={int(m):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"part-run={run_id}-hash={file_hash[:12]}.parquet"
        if out_file.exists():
            # Strict immutability: never overwrite
            raise FileExistsError(f"Refusing to overwrite Bronze file: {out_file}")

        g.to_parquet(out_file, index=False, compression=cfg.parquet_compression)
        written.append(str(out_file))

    return written


# -----------------------------
# Archiving paths
# -----------------------------

def dated_subdir(root: Path, dt: datetime) -> Path:
    return root / f"year={dt:%Y}" / f"month={dt:%m}" / f"day={dt:%d}"

def archived_name(original: Path, file_hash: str) -> str:
    # Keep original stem, append hash
    return f"{original.stem}__hash={file_hash[:12]}{original.suffix}"

def move_to_archive(src_processing: Path, cfg: Config, file_hash: str, bucket: str) -> Path:
    """
    bucket in {"archived", "duplicates", "rejected"}; for rejected we use cfg.rejected root.
    """
    now_local = datetime.now()  # local machine time OK for folder naming
    if bucket == "archived":
        root = cfg.archived
    elif bucket == "duplicates":
        root = cfg.archived / "duplicates"
    elif bucket == "rejected":
        root = cfg.rejected
    else:
        raise ValueError(f"Unknown archive bucket: {bucket}")

    dst_dir = dated_subdir(root, now_local)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_name = archived_name(src_processing, file_hash)
    dst = dst_dir / dst_name
    atomic_move(src_processing, dst)
    return dst


# -----------------------------
# Locking (single instance)
# -----------------------------

def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\nstart_utc={utc_now().isoformat()}\n")
    except FileExistsError:
        raise RuntimeError(f"Lock exists: {lock_path}. Another run may be active.")

def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)  # py3.8+: missing_ok available in 3.12 yes
    except Exception:
        pass


# -----------------------------
# Reset utilities
# -----------------------------

def reset_ops(cfg: Config, remove_run_logs: bool = True) -> None:
    reg = registry_path(cfg)
    runlog_dir = cfg.bronze_root / cfg.ops_dirname / cfg.runlog_dirname
    lock = cfg.lockfile
    ensure_under(reg, cfg.bronze_root)
    ensure_under(runlog_dir, cfg.bronze_root)
    ensure_under(lock, cfg.data_root)
    safe_unlink(reg)
    if remove_run_logs:
        safe_rmtree(runlog_dir)
    safe_unlink(lock)


def reset_dataset(cfg: Config, remove_run_logs: bool = True) -> None:
    dataset_dir = cfg.bronze_root / cfg.dataset_name
    ensure_under(dataset_dir, cfg.bronze_root)
    safe_rmtree(dataset_dir)
    reset_ops(cfg, remove_run_logs=remove_run_logs)


# -----------------------------
# Main folder ingestion
# -----------------------------

def list_inbox_files(cfg: Config) -> List[Path]:
    allowed = {".xlsx", ".xls", ".xlsm", ".csv", ".txt"}
    files = []
    for p in cfg.inbox.iterdir():
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p)
    # oldest first
    files.sort(key=lambda x: x.stat().st_mtime)
    return files

def ingest_one_file(inbox_file: Path, mapping: Dict[str, str], cfg: Config, registry_df: pd.DataFrame,
                     mapping_filename: str = "", mapping_file_hash: str = "", allow_duplicates: bool = False) -> Tuple[str, str]:
    """
    Returns (status, message) where status in {"ingested","skipped_duplicate","rejected"}.
    Always moves file out of processing to archive/rejected.
    
    Args:
        inbox_file: File to ingest
        mapping: Column mapping dict
        cfg: Config object
        registry_df: Current registry dataframe
        mapping_filename: Name of mapping file (e.g., park_power_mapping_v001.csv)
        mapping_file_hash: SHA256 hash of mapping file
    """
    # Safety checks before we move
    if not is_file_old_enough(inbox_file, cfg.min_age_seconds):
        return "rejected", f"file too new (<{cfg.min_age_seconds}s): {inbox_file.name}"
    if not is_file_size_stable(inbox_file, cfg.stable_check_seconds):
        return "rejected", f"file size not stable: {inbox_file.name}"

    # Move to processing (ownership/lock)
    proc_path = cfg.processing / inbox_file.name
    try:
        atomic_move(inbox_file, proc_path)
    except Exception as e:
        return "rejected", f"failed to move to processing: {e}"

    run = run_id_utc()
    file_hash = ""
    archived_path: Optional[Path] = None

    try:
        file_hash = sha256_file(proc_path)

        duplicate_seen = registry_has_hash(registry_df, cfg.dataset_name, file_hash)

        # Duplicate hash => skip unless explicitly allowed
        if duplicate_seen and not allow_duplicates:
            archived_path = move_to_archive(proc_path, cfg, file_hash, bucket="duplicates")
            append_registry_row(cfg, {
                "dataset": cfg.dataset_name,
                "source_file_hash": file_hash,
                "source_file": str(archived_path),
                "run_id": run,
                "status": "skipped_duplicate",
                "rows_long": "0",
                "files_written": "0",
                "archived_path": str(archived_path),
                "ingested_at_utc": utc_now().isoformat(),
                "message": "same file hash already ingested; skipped",
            })
            return "skipped_duplicate", f"duplicate hash; moved to {archived_path}"

        # Read + rename
        df_wide = read_wide_file(proc_path, cfg)
        df_wide = df_wide.rename(columns={str(c): mapping.get(str(c), str(c)) for c in df_wide.columns})

        # Transform
        long_df = wide_to_long(
            df_wide=df_wide,
            cfg=cfg,
            source_file=str(proc_path.resolve()),
            source_file_hash=file_hash,
            run_id=run,
        )

        # Write Bronze
        written_files = write_bronze_monthly(long_df, cfg, run_id=run, file_hash=file_hash)

        # Archive source file (success)
        archived_path = move_to_archive(proc_path, cfg, file_hash, bucket="archived")

        status_note = "duplicate hash re-ingested (allow_duplicates)" if duplicate_seen else "ok"

        # Register + runlog (with mapping metadata)
        append_registry_row(cfg, {
            "dataset": cfg.dataset_name,
            "source_file_hash": file_hash,
            "source_file": str(archived_path),
            "run_id": run,
            "status": "ingested",
            "rows_long": str(len(long_df)),
            "files_written": str(len(written_files)),
            "archived_path": str(archived_path),
            "mapping_filename": mapping_filename,
            "mapping_file_hash": mapping_file_hash,
            "ingested_at_utc": utc_now().isoformat(),
            "message": status_note,
        })

        write_runlog(cfg, run, {
            "run_id": run,
            "dataset": cfg.dataset_name,
            "input_file_processing_path": str(proc_path),
            "archived_path": str(archived_path),
            "source_file_hash": file_hash,
            "mapping_filename": mapping_filename,
            "mapping_file_hash": mapping_file_hash,
            "rows_long": int(len(long_df)),
            "files_written": written_files,
            "ingested_at_utc": utc_now().isoformat(),
            "duplicate_seen": bool(duplicate_seen),
            "allow_duplicates": bool(allow_duplicates),
            "status_note": status_note,
        })

        suffix = " (allow_duplicates)" if duplicate_seen else ""
        return "ingested", f"ingested{suffix}; archived to {archived_path}; wrote {len(written_files)} parquet file(s)"

    except Exception as e:
        # Move to rejected and log reason
        reason = slug_reason(str(e))
        # keep hash if computed, else placeholder
        if not file_hash:
            try:
                file_hash = sha256_file(proc_path)
            except Exception:
                file_hash = "nohash"

        # rename rejected file to include reason + hash
        try:
            rej_dir = dated_subdir(cfg.rejected, datetime.now())
            rej_dir.mkdir(parents=True, exist_ok=True)
            dst = rej_dir / f"{proc_path.stem}__reason={reason}__hash={file_hash[:12]}{proc_path.suffix}"
            atomic_move(proc_path, dst)
            archived_path = dst
        except Exception:
            # worst case: leave in processing
            archived_path = proc_path

        append_registry_row(cfg, {
            "dataset": cfg.dataset_name,
            "source_file_hash": file_hash,
            "source_file": str(archived_path),
            "run_id": run,
            "status": "rejected",
            "rows_long": "0",
            "files_written": "0",
            "archived_path": str(archived_path),
            "mapping_filename": mapping_filename,
            "mapping_file_hash": mapping_file_hash,
            "ingested_at_utc": utc_now().isoformat(),
            "message": f"error: {e}",
        })

        return "rejected", f"rejected ({e}); moved to {archived_path}"


def ingest_folder(cfg: Config, mapping_filename: Optional[str] = None) -> None:
    """
    Ingest files from inbox folder using versioned mapping.
    
    Args:
        cfg: Configuration object
        mapping_filename: Optional override of mapping file (otherwise read from current.txt)
    """
    # Ensure dirs exist
    cfg.inbox.mkdir(parents=True, exist_ok=True)
    cfg.processing.mkdir(parents=True, exist_ok=True)
    cfg.archived.mkdir(parents=True, exist_ok=True)
    cfg.rejected.mkdir(parents=True, exist_ok=True)
    cfg.bronze_root.mkdir(parents=True, exist_ok=True)
    cfg.mappings_root.mkdir(parents=True, exist_ok=True)

    # Lock
    acquire_lock(cfg.lockfile)
    try:
        # Determine mapping file path
        if mapping_filename:
            mapping_csv = cfg.mappings_root / mapping_filename
        else:
            mapping_csv = get_current_mapping_path(cfg)
        
        mapping_filename = mapping_csv.name
        mapping_file_hash = compute_mapping_hash(mapping_csv)
        
        mapping = read_mapping(mapping_csv)
        registry_df = load_registry(cfg)

        files = list_inbox_files(cfg)
        if not files:
            print("No files found in inbox.")
            return

        print(f"Found {len(files)} file(s) in inbox.")
        print(f"Using mapping: {mapping_filename} (hash: {mapping_file_hash[:12]})")
        
        for f in files:
            status, msg = ingest_one_file(
                f,
                mapping,
                cfg,
                registry_df,
                mapping_filename,
                mapping_file_hash,
                allow_duplicates=cfg.allow_duplicates,
            )
            print(f"[{status}] {f.name} -> {msg}")

            # Reload registry after each file so hash-skip is up-to-date within the same run
            registry_df = load_registry(cfg)

    finally:
        release_lock(cfg.lockfile)


# -----------------------------
# CLI
# -----------------------------

def build_cfg(args) -> Config:
    data_root = Path(args.data_root).resolve()
    bronze_root = Path(args.bronze_root).resolve()
    mappings_root = Path(args.mappings_root).resolve() if args.mappings_root else data_root.parent / "mappings"

    cfg = Config(
        data_root=data_root,
        inbox=(data_root / "inbox").resolve(),
        processing=(data_root / "processing").resolve(),
        archived=(data_root / "archived").resolve(),
        rejected=(data_root / "rejected").resolve(),
        bronze_root=bronze_root,
        mappings_root=mappings_root,
        dataset_name=args.dataset_name,
        timezone_local=args.timezone,
        daily_interval_end_is_midnight=not args.midnight_is_start,  # default True unless explicitly flipped
        parquet_compression=args.compression,
        min_age_seconds=args.min_age_seconds,
        stable_check_seconds=args.stable_check_seconds,
        sheet_name=args.sheet,
        csv_sep=args.sep,
        csv_encoding=args.encoding,
        allow_duplicates=args.allow_duplicates,
    )

    # lockfile in data_root/_locks
    lockdir = data_root / "_locks"
    cfg.lockfile = lockdir / "bronze_ingest.lock"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Bronze ingestion with versioned mapping support")
    ap.add_argument("--data_root", required=True, help="Root data folder containing inbox/processing/archived/rejected")
    ap.add_argument("--bronze_root", required=True, help="Bronze root folder (e.g. D:\\PV_DATA\\bronze)")
    ap.add_argument("--mappings_root", default=None, 
                    help="Versioned mappings folder (default: parent(data_root)/mappings)")
    ap.add_argument("--mapping_override", default=None,
                    help="Optional: override current.txt and use specific mapping file (filename only)")
    ap.add_argument("--dataset_name", default="scada_1d_signal", help="Bronze dataset name folder")
    ap.add_argument("--timezone", default="Europe/Athens", help="Local timezone for daily boundary")
    ap.add_argument("--midnight_is_start", action="store_true",
                    help="If set, midnight timestamp belongs to SAME day (do NOT subtract 1 day)")
    ap.add_argument("--compression", default="zstd", help="Parquet compression: zstd or snappy")
    ap.add_argument("--min_age_seconds", type=int, default=90, help="Only ingest files older than this")
    ap.add_argument("--stable_check_seconds", type=int, default=20, help="Check file size stable across this interval")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    ap.add_argument("--sep", default=None, help="CSV delimiter (optional)")
    ap.add_argument("--encoding", default=None, help="CSV encoding (optional)")
    ap.add_argument("--allow_duplicates", action="store_true",
                    help="Allow duplicate source hashes to be ingested again during this run")
    ap.add_argument("--reset_ops", action="store_true",
                    help="Delete ingest registry, run logs, and lock file, then exit")
    ap.add_argument("--reset_dataset", action="store_true",
                    help="Delete Bronze dataset folder plus ops metadata, then exit")
    ap.add_argument("--i_understand", action="store_true",
                    help="Required with --reset_dataset to avoid accidental wipes")
    args = ap.parse_args()

    cfg = build_cfg(args)

    if args.reset_dataset:
        if not args.i_understand:
            print("Refusing to reset dataset without --i_understand flag.")
            return
        reset_dataset(cfg)
        print(f"Reset dataset under {cfg.bronze_root / cfg.dataset_name} and ops metadata.")
        return

    if args.reset_ops:
        reset_ops(cfg)
        print(f"Reset ops metadata under {cfg.bronze_root / cfg.ops_dirname} and lock file.")
        return

    ingest_folder(cfg, mapping_filename=args.mapping_override)


if __name__ == "__main__":
    main()
