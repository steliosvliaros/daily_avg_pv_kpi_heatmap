#!/usr/bin/env python3
"""
Silver build: transform Bronze parquet into a curated Silver dataset.

Flow:
- Read Bronze ingest registry to find new run_id values
- Load Bronze parquet parts for each run
- Normalize types, drop invalid rows, de-duplicate by ingest_key
- Write Silver parquet partitioned by year/month
- Track progress in a Silver registry and last_run_id.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class Config:
    bronze_root: Path
    silver_root: Path
    dataset_name: str = "scada_1d_signal"
    ops_dirname: str = "_ops"
    bronze_registry_filename: str = "ingest_registry_files.csv"
    silver_registry_filename: str = "silver_registry_runs.csv"
    runlog_dirname: str = "run_logs"
    last_run_id_filename: str = "last_run_id.txt"
    parquet_compression: str = "zstd"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_id_utc() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


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


def bronze_registry_path(cfg: Config) -> Path:
    return cfg.bronze_root / cfg.ops_dirname / cfg.bronze_registry_filename


def silver_registry_path(cfg: Config) -> Path:
    return cfg.silver_root / cfg.ops_dirname / cfg.silver_registry_filename


def last_run_id_path(cfg: Config) -> Path:
    return cfg.silver_root / cfg.ops_dirname / cfg.last_run_id_filename


def load_last_run_id(cfg: Config) -> str:
    p = last_run_id_path(cfg)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return "00000000T000000Z"


def write_last_run_id(cfg: Config, run_id: str) -> None:
    p = last_run_id_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run_id, encoding="utf-8")


def load_bronze_registry(cfg: Config) -> pd.DataFrame:
    p = bronze_registry_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"Bronze registry not found: {p}")
    return pd.read_csv(p, dtype=str)


def get_new_bronze_runs(cfg: Config, last_run_id: str) -> List[str]:
    df = load_bronze_registry(cfg)
    df = df[(df["dataset"] == cfg.dataset_name) & (df["status"] == "ingested")]
    runs = sorted(df[df["run_id"] > last_run_id]["run_id"].unique())
    return runs


def find_bronze_parts(cfg: Config, bronze_run_id: str) -> List[Path]:
    base = cfg.bronze_root / cfg.dataset_name
    pattern = f"year=*{Path.sep}month=*{Path.sep}part-run={bronze_run_id}-hash=*.parquet"
    return sorted(base.glob(pattern))


def normalize_bronze_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_local"] = pd.to_datetime(df["ts_local"], errors="coerce")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df["interval_start_date"] = pd.to_datetime(df["interval_start_date"], errors="coerce").dt.date

    df["park_capacity_kwp"] = pd.to_numeric(df["park_capacity_kwp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["ts_local", "ts_utc", "interval_start_date", "park_id", "signal_name", "value"])

    # Re-derive partition columns from interval_start_date to avoid drift.
    interval_dt = pd.to_datetime(df["interval_start_date"])
    df["year"] = interval_dt.dt.year.astype(int)
    df["month"] = interval_dt.dt.month.astype(int)

    # De-duplicate by deterministic ingest_key if present.
    if "ingest_key" in df.columns:
        df = df.drop_duplicates(subset=["ingest_key"], keep="last")

    keep = [
        "ts_local",
        "ts_utc",
        "interval_start_date",
        "year",
        "month",
        "park_id",
        "park_capacity_kwp",
        "signal_name",
        "unit",
        "value",
        "source_file_hash",
        "run_id",
        "ingested_at_utc",
        "ingest_key",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing]


def write_silver_monthly(df: pd.DataFrame, cfg: Config, bronze_run_id: str, silver_run_id: str) -> List[str]:
    ensure_parquet_engine()
    base_dir = cfg.silver_root / cfg.dataset_name
    written: List[str] = []

    for (y, m), g in df.groupby(["year", "month"], dropna=False):
        out_dir = base_dir / f"year={int(y):04d}" / f"month={int(m):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"part-bronze={bronze_run_id}-silver={silver_run_id}.parquet"
        if out_file.exists():
            raise FileExistsError(f"Refusing to overwrite Silver file: {out_file}")
        g.to_parquet(out_file, index=False, compression=cfg.parquet_compression)
        written.append(str(out_file))

    return written


def append_silver_registry_row(cfg: Config, row: dict) -> None:
    p = silver_registry_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if p.exists():
        df.to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, mode="w", header=True, index=False)


def write_runlog(cfg: Config, silver_run_id: str, payload: dict) -> None:
    d = cfg.silver_root / cfg.ops_dirname / cfg.runlog_dirname
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run_{silver_run_id}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_silver_for_run(cfg: Config, bronze_run_id: str, silver_run_id: str) -> None:
    parts = find_bronze_parts(cfg, bronze_run_id)
    if not parts:
        append_silver_registry_row(cfg, {
            "bronze_run_id": bronze_run_id,
            "silver_run_id": silver_run_id,
            "status": "missing_parts",
            "rows_in": "0",
            "rows_out": "0",
            "files_written": "0",
            "built_at_utc": utc_now().isoformat(),
            "message": "no parquet parts found for run_id",
        })
        return

    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    rows_in = len(df)
    df = normalize_bronze_rows(df)
    rows_out = len(df)

    written = write_silver_monthly(df, cfg, bronze_run_id=bronze_run_id, silver_run_id=silver_run_id)

    append_silver_registry_row(cfg, {
        "bronze_run_id": bronze_run_id,
        "silver_run_id": silver_run_id,
        "status": "built",
        "rows_in": str(rows_in),
        "rows_out": str(rows_out),
        "files_written": str(len(written)),
        "built_at_utc": utc_now().isoformat(),
        "message": "ok",
    })

    write_runlog(cfg, silver_run_id, {
        "silver_run_id": silver_run_id,
        "bronze_run_id": bronze_run_id,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "files_written": written,
        "built_at_utc": utc_now().isoformat(),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Silver dataset from Bronze runs")
    ap.add_argument("--bronze_root", required=True, help="Bronze root folder")
    ap.add_argument("--silver_root", required=True, help="Silver root folder")
    ap.add_argument("--dataset_name", default="scada_1d_signal", help="Dataset name")
    ap.add_argument("--last_run_id", default=None, help="Override last_run_id.txt")
    ap.add_argument("--compression", default="zstd", help="Parquet compression: zstd or snappy")
    args = ap.parse_args()

    cfg = Config(
        bronze_root=Path(args.bronze_root).resolve(),
        silver_root=Path(args.silver_root).resolve(),
        dataset_name=args.dataset_name,
        parquet_compression=args.compression,
    )

    last_run_id = args.last_run_id or load_last_run_id(cfg)
    new_runs = get_new_bronze_runs(cfg, last_run_id)
    if not new_runs:
        print("No new Bronze runs to process.")
        return

    for bronze_run_id in new_runs:
        silver_run_id = run_id_utc()
        build_silver_for_run(cfg, bronze_run_id, silver_run_id)
        write_last_run_id(cfg, bronze_run_id)
        print(f"Built Silver for Bronze run {bronze_run_id} -> Silver run {silver_run_id}")


if __name__ == "__main__":
    main()
