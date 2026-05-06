from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_daily_energy_long_csv(
    workspace_root: Path,
    output_csv: Path,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    status_effective: str | None = "all",
    interval_minutes: int = 15,
    production_only: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Export long-format daily energy data from silver with park metadata."""
    silver_root = workspace_root / "silver"
    metadata_csv = workspace_root / "mappings" / "park_metadata.csv"
    if not silver_root.exists():
        raise ValueError(f"Silver root not found: {silver_root}")
    if not metadata_csv.exists():
        raise ValueError(f"Metadata file not found: {metadata_csv}")

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    metadata_full = pd.read_csv(metadata_csv)
    metadata_full["park_id"] = metadata_full["park_id"].astype("string").str.strip().str.lower()
    metadata_full = metadata_full.drop_duplicates(subset=["park_id"], keep="last")

    allowed_parks = None
    if status_effective is not None and str(status_effective).strip().lower() != "all":
        status_values = {str(status_effective).strip().lower()}
        status_series = metadata_full["status_effective"].astype("string").str.strip().str.lower()
        allowed_parks = set(metadata_full.loc[status_series.isin(status_values), "park_id"].dropna().astype(str))

    part_files = sorted(silver_root.glob("year=*/month=*/part-*.parquet"))
    if not part_files:
        raise ValueError("No silver parquet partition files found.")

    interval_hours = float(interval_minutes) / 60.0
    base_cols = ["ts_local", "ts_utc", "park_id", "signal_name", "unit", "value"]
    flag_cols = ["flag_missing_required", "flag_invalid_value", "flag_invalid_unit_range", "flag_duplicate"]
    use_cols = base_cols + flag_cols

    daily_chunks: list[pd.DataFrame] = []
    for idx, pf in enumerate(part_files, start=1):
        if debug and (idx == 1 or idx % 24 == 0 or idx == len(part_files)):
            print(f"[{idx}/{len(part_files)}] {pf.name}")

        df = pd.read_parquet(pf, columns=use_cols)
        if df.empty:
            continue

        df["park_id"] = df["park_id"].astype("string").str.strip().str.lower()
        df["signal_name"] = df["signal_name"].astype("string").str.strip().str.lower()

        if production_only:
            sig = df["signal_name"].astype("string")
            include_mask = sig.str.contains("power|energy", regex=True, na=False)
            exclude_mask = sig.str.contains("temperature|irradiance|wind|weather", regex=True, na=False)
            df = df[include_mask & ~exclude_mask]
            if df.empty:
                continue

        if allowed_parks is not None:
            df = df[df["park_id"].isin(allowed_parks)]
            if df.empty:
                continue

        ts_local = pd.to_datetime(df["ts_local"], errors="coerce") if "ts_local" in df.columns else pd.Series(pd.NaT, index=df.index)
        if ts_local.notna().any():
            ts = ts_local
        else:
            ts = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)

        # Normalize to timezone-naive for stable date filtering regardless of source tz.
        if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)

        df = df[ts.notna()].copy()
        ts = ts.loc[df.index]
        if df.empty:
            continue

        if start_ts is not None:
            df = df[ts >= start_ts]
            ts = ts.loc[df.index]
        if end_ts is not None:
            df = df[ts <= end_ts]
            ts = ts.loc[df.index]
        if df.empty:
            continue

        df = df[df["value"].notna()].copy()
        if df.empty:
            continue

        available_flags = [c for c in flag_cols if c in df.columns]
        if available_flags:
            df = df[~df[available_flags].fillna(False).any(axis=1)]
            if df.empty:
                continue

        unit_series = df["unit"].astype("string").str.strip().str.lower().fillna("")
        value = pd.to_numeric(df["value"], errors="coerce")

        is_kwh = unit_series.str.contains("kwh", na=False)
        is_wh = unit_series.str.contains("wh", na=False) & ~is_kwh
        is_kw = unit_series.str.contains("kw", na=False) & ~is_kwh
        is_w = unit_series.str.contains("w", na=False) & ~is_kw & ~is_kwh & ~is_wh

        energy_value = value.copy()
        energy_value.loc[is_w] = (energy_value.loc[is_w] / 1000.0) * interval_hours
        energy_value.loc[is_kw] = energy_value.loc[is_kw] * interval_hours
        energy_value.loc[is_wh] = energy_value.loc[is_wh] / 1000.0
        energy_value.loc[is_kwh] = energy_value.loc[is_kwh]
        unknown_mask = ~(is_w | is_kw | is_wh | is_kwh)
        energy_value.loc[unknown_mask] = energy_value.loc[unknown_mask] * interval_hours

        chunk = pd.DataFrame(
            {
                "date": ts.dt.normalize(),
                "park_id": df["park_id"],
                "sensor_name": df["signal_name"],
                "value": energy_value,
            }
        )
        chunk = chunk.dropna(subset=["date", "park_id", "sensor_name", "value"])
        if chunk.empty:
            continue

        chunk_daily = chunk.groupby(["date", "park_id", "sensor_name"], as_index=False)["value"].sum()
        daily_chunks.append(chunk_daily)

    if not daily_chunks:
        raise ValueError("No daily energy data found in silver for the requested period.")

    long_df = pd.concat(daily_chunks, ignore_index=True)
    long_df = long_df.groupby(["date", "park_id", "sensor_name"], as_index=False)["value"].sum()

    metadata_cols = ["park_id", "park_name", "park_iso_name"]
    metadata_df = pd.read_csv(metadata_csv, usecols=lambda c: c in metadata_cols)
    metadata_df["park_id"] = metadata_df["park_id"].astype("string").str.strip().str.lower()
    metadata_df = metadata_df.drop_duplicates(subset=["park_id"], keep="last")

    long_df = long_df.merge(metadata_df, on="park_id", how="left")

    long_df["park_name"] = long_df["park_name"].fillna(long_df["park_id"])
    long_df["park_iso_name"] = long_df["park_iso_name"].fillna("")

    long_df = long_df[long_df["value"].notna()].copy()
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce").dt.date
    long_df = long_df[long_df["date"].notna()].copy()

    output_df = long_df[["park_name", "sensor_name", "park_iso_name", "park_id", "date", "value"]]
    output_df = output_df.sort_values(["date", "park_id", "sensor_name"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    return output_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export silver daily energy (kWh) to long-format CSV with park metadata."
    )
    parser.add_argument("--workspace-root", default=".", help="Workspace root path")
    parser.add_argument("--output", default="outputs/silver_daily_energy_2015_to_date.csv", help="Output CSV path")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (inclusive)")
    parser.add_argument("--end-date", default=None, help="End date (inclusive), defaults to latest available")
    parser.add_argument("--status-effective", default="all", help="Status filter from metadata, e.g. true/false/all")
    parser.add_argument("--interval-minutes", type=int, default=15, help="Sampling interval in minutes")
    parser.add_argument("--all-sensors", action="store_true", help="Include all sensors instead of only production sensors")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    output_csv = Path(args.output)
    if not output_csv.is_absolute():
        output_csv = workspace_root / output_csv

    df = build_daily_energy_long_csv(
        workspace_root=workspace_root,
        output_csv=output_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        status_effective=args.status_effective,
        interval_minutes=args.interval_minutes,
        production_only=not args.all_sensors,
        debug=args.debug,
    )

    print(f"Exported {len(df):,} rows to {output_csv}")
    if not df.empty:
        print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
        print(f"Parks: {df['park_id'].nunique():,}")
        print(f"Sensors: {df['sensor_name'].nunique():,}")


if __name__ == "__main__":
    main()