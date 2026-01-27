from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def get_runlog_logger(name: str = "runlog_loader", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for runlog loading.

    Adds a StreamHandler with basic formatting if no handlers are attached.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass
class LoaderStats:
    runlogs_considered: int = 0
    runlogs_matched: int = 0
    files_listed: int = 0
    files_deduped: int = 0
    files_read_ok: int = 0
    files_read_failed: int = 0
    newest_run_id: str = "00000000T000000Z"


class FileLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._acquired = False

    def acquire(self, timeout_sec: int = 30, poll_interval: float = 0.25) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                # Create exclusively; fail if exists
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._acquired = True
                return True
            except FileExistsError:
                time.sleep(poll_interval)
        return False

    def release(self) -> None:
        if self._acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink(missing_ok=True)
            finally:
                self._acquired = False

    def __enter__(self):
        ok = self.acquire()
        if not ok:
            raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        # do not suppress exceptions
        return False


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _is_valid_run_id(run_id: str) -> bool:
    # Very light validation: must be 16 chars like YYYYMMDDThhmmssZ
    return (
        isinstance(run_id, str)
        and len(run_id) == 16
        and run_id[8] == "T"
        and run_id.endswith("Z")
        and run_id[:8].isdigit()
        and run_id[9:15].isdigit()
    )


def _normalize_paths(paths: Iterable[str | Path]) -> List[Path]:
    result: List[Path] = []
    for p in paths:
        pp = Path(p)
        result.append(pp)
    # de-duplicate while preserving order
    seen = set()
    deduped: List[Path] = []
    for p in result:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def load_new_bronze_parts_from_runlogs(
    bronze_root: Path,
    state_path: Path,
    dataset_name: str = "scada_1d_signal",
    lock_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Hardened loader: scans bronze run logs for new runs (by run_id watermark),
    validates payloads, reads available parquet parts, and atomically advances the watermark
    only after at least one file is loaded successfully.

    Parameters
    ----------
    bronze_root : Path
        Root directory containing the bronze dataset and _ops/run_logs.
    state_path : Path
        Path to a file storing last processed run_id watermark.
    dataset_name : str
        Dataset filter for run logs; defaults to 'scada_1d_signal'.
    lock_root : Optional[Path]
        Directory in which to place a watermark lock file; defaults to sibling of state_path.
    logger : Optional[logging.Logger]
        Logger for observability; if None, uses a module-level logger.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame of all successfully read parquet parts; empty if none.
    """
    log = logger or logging.getLogger(__name__)

    runlog_dir = bronze_root / "_ops" / "run_logs"
    if not runlog_dir.exists():
        log.info("Run log directory missing: %s", runlog_dir)
        return pd.DataFrame()

    # Read last processed run_id (watermark)
    last_run_id = "00000000T000000Z"
    if state_path.exists():
        try:
            last_run_id = state_path.read_text(encoding="utf-8").strip() or last_run_id
        except Exception as e:
            log.warning("Failed to read watermark %s: %s", state_path, e)

    stats = LoaderStats(newest_run_id=last_run_id)

    runlog_files = sorted(runlog_dir.glob("run_*.json"))
    stats.runlogs_considered = len(runlog_files)
    new_runlogs: List[Path] = [p for p in runlog_files if p.stem.replace("run_", "") > last_run_id]

    if not new_runlogs:
        log.info("No new run logs after watermark %s", last_run_id)
        return pd.DataFrame()

    parquet_files: List[Path] = []
    for p in new_runlogs:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Skipping unreadable run log %s: %s", p, e)
            continue
        run_id = payload.get("run_id")
        dataset = payload.get("dataset")
        files_written = payload.get("files_written")

        if dataset != dataset_name:
            continue
        stats.runlogs_matched += 1

        if not _is_valid_run_id(run_id or ""):
            log.warning("Skipping run log with invalid run_id %s: %s", run_id, p)
            continue

        if run_id > stats.newest_run_id:
            stats.newest_run_id = run_id

        if not isinstance(files_written, list):
            log.warning("Run log missing files_written list: %s", p)
            continue

        parquet_files.extend(files_written)

    if not parquet_files:
        log.info("No parquet files listed in matched run logs after %s", last_run_id)
        return pd.DataFrame()

    files = _normalize_paths(parquet_files)
    stats.files_listed = len(parquet_files)
    stats.files_deduped = len(files)

    # Verify existence; keep only existing files
    existing_files: List[Path] = []
    for f in files:
        if not Path(f).exists():
            log.warning("Listed parquet file not found: %s", f)
            continue
        existing_files.append(Path(f))

    if not existing_files:
        log.info("No existing parquet files to read after %s", last_run_id)
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for f in existing_files:
        try:
            df_part = pd.read_parquet(f)
            dfs.append(df_part)
            stats.files_read_ok += 1
        except Exception as e:
            stats.files_read_failed += 1
            log.error("Failed reading parquet %s: %s", f, e)

    if not dfs:
        log.error("All parquet reads failed; watermark unchanged: %s", last_run_id)
        return pd.DataFrame()

    # Concatenate; sort columns to tolerate minor schema drift
    df_new = pd.concat(dfs, ignore_index=True, sort=True)

    # Advance watermark atomically with a simple lock
    lock_path = (
        (lock_root / (state_path.name + ".lock")) if lock_root else state_path.with_suffix(state_path.suffix + ".lock")
    )

    try:
        with FileLock(lock_path):
            _atomic_write_text(state_path, stats.newest_run_id)
            log.info(
                "Watermark advanced from %s to %s (ok=%d, failed=%d)",
                last_run_id,
                stats.newest_run_id,
                stats.files_read_ok,
                stats.files_read_failed,
            )
    except Exception as e:
        log.error("Failed to advance watermark %s: %s", state_path, e)

    return df_new
