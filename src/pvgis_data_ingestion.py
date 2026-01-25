from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.pvgis_pi_heatmap import (
    expected_daily_kwh,
    fetch_pvgis_hourly_cached,
    parse_kwp_from_header,
)
from src.silver_pre_ingestion_eda import EdaConfig, run_silver_pre_ingestion_eda
from src.silver_prepair import load_park_metadata
from src.utils import ensure_dir, sanitize_filename


DEFAULT_SIGNAL_NAME = "pvgis_expected_daily_kwh"
DEFAULT_UNIT = "kwh"
PVGIS_MIN_YEAR = 2005
PVGIS_MAX_YEAR = 2023


@dataclass
class TypicalYearConfig:
    workspace_root: Optional[Path] = None
    metadata_path: Optional[Path] = None
    cache_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    eda_output_dir: Optional[Path] = None
    use_cache: bool = True
    save_cache: bool = False
    save_output: bool = False
    start_year: int = 2015
    end_year: int = 2023
    loss_pct: float = 18.0
    default_capacity_kwp: float = 100.0
    default_timezone: str = "Europe/Athens"
    pvgis_url: str = "https://re.jrc.ec.europa.eu/api/"
    drop_feb29: bool = True
    reference_year: int = 2001
    run_eda_on_new: bool = True
    save_eda_plots: bool = False
    save_eda_stats: bool = False
    show_eda_plots: bool = False
    force_download: bool = False


def _hash_payload(payload: Dict[str, object]) -> str:
    blob = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _clamp_years(start_year: int, end_year: int) -> Tuple[int, int]:
    start = max(PVGIS_MIN_YEAR, int(start_year))
    end = min(PVGIS_MAX_YEAR, int(end_year))
    if start > end:
        raise ValueError(f"Invalid PVGIS year range after clamping: {start} to {end}")
    if start != start_year or end != end_year:
        print(f"Warning: PVGIS years clamped to {start}-{end}")
    return start, end


def _resolve_paths(config: TypicalYearConfig) -> Tuple[Path, Path, Path, Path, Path, Path]:
    root = Path(config.workspace_root) if config.workspace_root else Path.cwd()
    metadata_path = config.metadata_path or (root / "mappings" / "park_metadata.csv")
    cache_root = config.cache_root or (root / "pvgis_cache")
    output_dir = config.output_dir or (root / "outputs" / "pvgis_typical_year")
    eda_output_dir = config.eda_output_dir or (root / "plots" / "pvgis_typical_year_eda")
    daily_cache_dir = cache_root / "typical_daily"
    return root, metadata_path, cache_root, daily_cache_dir, output_dir, eda_output_dir


def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def build_park_meta(
    metadata_path: Path,
    default_capacity_kwp: float,
    default_timezone: str,
) -> pd.DataFrame:
    df = load_park_metadata(metadata_path)
    if df is None or df.empty:
        raise FileNotFoundError(f"Missing or empty metadata file: {metadata_path}")

    df = df.copy()
    if "park_name" not in df.columns:
        df["park_name"] = df["park_id"]

    for col in [
        "latitude",
        "longitude",
        "nearest_meteo_station_lat",
        "nearest_meteo_station_lon",
        "capacity_kwp",
    ]:
        _coerce_numeric(df, col)

    lat = df["latitude"] if "latitude" in df.columns else pd.Series(pd.NA, index=df.index)
    lon = df["longitude"] if "longitude" in df.columns else pd.Series(pd.NA, index=df.index)
    if "nearest_meteo_station_lat" in df.columns:
        lat = lat.fillna(df["nearest_meteo_station_lat"])
    if "nearest_meteo_station_lon" in df.columns:
        lon = lon.fillna(df["nearest_meteo_station_lon"])
    df["latitude"] = lat
    df["longitude"] = lon

    def _capacity_from_row(row: pd.Series) -> float:
        cap = row.get("capacity_kwp")
        if pd.notna(cap) and float(cap) > 0:
            return float(cap)
        label = str(row.get("park_name") or row.get("park_id") or "")
        return float(parse_kwp_from_header(label, default=default_capacity_kwp))

    df["capacity_kwp"] = df.apply(_capacity_from_row, axis=1)

    if "timezone" in df.columns:
        df["timezone"] = df["timezone"].fillna(default_timezone)
    else:
        df["timezone"] = default_timezone

    missing = df["latitude"].isna() | df["longitude"].isna() | df["capacity_kwp"].isna()
    if missing.any():
        missing_count = int(missing.sum())
        print(f"Warning: Skipping {missing_count} parks missing coordinates or capacity")
        df = df[~missing].copy()

    return df[["park_id", "park_name", "latitude", "longitude", "capacity_kwp", "timezone"]]


def typical_daily_from_hourly(
    hourly: pd.DataFrame,
    timezone_name: str,
    *,
    reference_year: int,
    drop_feb29: bool,
) -> pd.Series:
    daily = expected_daily_kwh(hourly, tz=timezone_name)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily = daily[~daily.index.duplicated(keep="first")]
    if drop_feb29:
        daily = daily[~((daily.index.month == 2) & (daily.index.day == 29))]
    grouped = daily.groupby([daily.index.month, daily.index.day]).mean()
    idx = [datetime(reference_year, int(m), int(d)) for m, d in grouped.index]
    return pd.Series(grouped.values, index=pd.to_datetime(idx), name="expected_kwh").sort_index()


def _build_daily_frame(
    series: pd.Series,
    *,
    park_id: str,
    park_name: str,
    latitude: float,
    longitude: float,
    capacity_kwp: float,
    timezone_name: str,
    start_year: int,
    end_year: int,
    loss_pct: float,
    reference_year: int,
) -> pd.DataFrame:
    df = series.rename("value").reset_index().rename(columns={"index": "interval_start_date"})
    df["interval_start_date"] = pd.to_datetime(df["interval_start_date"], errors="coerce").dt.date
    df["ts_utc"] = pd.to_datetime(df["interval_start_date"], errors="coerce", utc=True)
    df["park_id"] = str(park_id).strip().lower()
    df["park_name"] = str(park_name)
    df["signal_name"] = DEFAULT_SIGNAL_NAME
    df["unit"] = DEFAULT_UNIT
    df["park_capacity_kwp"] = float(capacity_kwp)
    df["latitude"] = float(latitude)
    df["longitude"] = float(longitude)
    df["timezone"] = str(timezone_name)
    df["pvgis_start_year"] = int(start_year)
    df["pvgis_end_year"] = int(end_year)
    df["pvgis_loss_pct"] = float(loss_pct)
    df["typical_year_reference"] = int(reference_year)
    df["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return df


def _daily_cache_path(
    daily_cache_dir: Path,
    park_id: str,
    cache_key: str,
) -> Path:
    safe = sanitize_filename(park_id)
    return daily_cache_dir / f"pvgis_typical_daily_{safe}_{cache_key}.parquet"


def fetch_typical_daily_for_park(
    row: pd.Series,
    *,
    start_year: int,
    end_year: int,
    loss_pct: float,
    reference_year: int,
    drop_feb29: bool,
    hourly_cache_dir: Path,
    daily_cache_dir: Path,
    pvgis_url: str,
    force_download: bool,
    use_cache: bool,
    save_cache: bool,
) -> Tuple[pd.DataFrame, bool, Path]:
    payload = {
        "park_id": row["park_id"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "capacity_kwp": float(row["capacity_kwp"]),
        "loss_pct": float(loss_pct),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "reference_year": int(reference_year),
        "drop_feb29": bool(drop_feb29),
        "pvgis_url": str(pvgis_url),
    }
    cache_key = _hash_payload(payload)
    cache_path = _daily_cache_path(daily_cache_dir, str(row["park_id"]), cache_key)

    if use_cache and cache_path.exists() and not force_download:
        return pd.read_parquet(cache_path), False, cache_path

    hourly = fetch_pvgis_hourly_cached(
        lat=float(row["latitude"]),
        lon=float(row["longitude"]),
        start_year=int(start_year),
        end_year=int(end_year),
        peakpower_kw=float(row["capacity_kwp"]),
        loss_pct=float(loss_pct),
        tilt_deg=None,
        azimuth_deg=None,
        cache_dir=hourly_cache_dir,
        url=pvgis_url,
        use_cache=use_cache,
        save_cache=save_cache,
    )

    series = typical_daily_from_hourly(
        hourly,
        str(row["timezone"]),
        reference_year=reference_year,
        drop_feb29=drop_feb29,
    )

    df = _build_daily_frame(
        series,
        park_id=str(row["park_id"]),
        park_name=str(row["park_name"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        capacity_kwp=float(row["capacity_kwp"]),
        timezone_name=str(row["timezone"]),
        start_year=int(start_year),
        end_year=int(end_year),
        loss_pct=float(loss_pct),
        reference_year=int(reference_year),
    )

    if save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df, True, cache_path


def build_pvgis_typical_year_dataset(
    config: TypicalYearConfig,
    park_ids: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    _, metadata_path, cache_root, daily_cache_dir, output_dir, eda_output_dir = _resolve_paths(config)
    if config.save_cache:
        ensure_dir(cache_root)
        ensure_dir(daily_cache_dir)
    if config.save_output:
        ensure_dir(output_dir)

    start_year, end_year = _clamp_years(config.start_year, config.end_year)

    meta = build_park_meta(
        metadata_path=metadata_path,
        default_capacity_kwp=config.default_capacity_kwp,
        default_timezone=config.default_timezone,
    )

    if park_ids:
        wanted = {str(p).strip().lower() for p in park_ids}
        meta = meta[meta["park_id"].isin(wanted)].copy()
        if meta.empty:
            raise ValueError("No matching park_ids found in metadata.")

    results: List[pd.DataFrame] = []
    newly_downloaded: List[pd.DataFrame] = []
    cache_paths: List[Path] = []

    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="PVGIS typical year"):
        row_series = pd.Series(row._asdict())
        df, is_new, cache_path = fetch_typical_daily_for_park(
            row_series,
            start_year=start_year,
            end_year=end_year,
            loss_pct=config.loss_pct,
            reference_year=config.reference_year,
            drop_feb29=config.drop_feb29,
            hourly_cache_dir=cache_root,
            daily_cache_dir=daily_cache_dir,
            pvgis_url=config.pvgis_url,
            force_download=config.force_download,
            use_cache=config.use_cache,
            save_cache=config.save_cache,
        )
        results.append(df)
        cache_paths.append(cache_path)
        if is_new:
            newly_downloaded.append(df)

    if not results:
        return {"message": "No data produced", "parks": 0}

    all_df = pd.concat(results, ignore_index=True)
    output_path = None
    if config.save_output:
        output_path = output_dir / "pvgis_typical_daily.parquet"
        all_df.to_parquet(output_path, index=False)

    eda_outputs = None
    if newly_downloaded and config.run_eda_on_new:
        new_df = pd.concat(newly_downloaded, ignore_index=True)
        if config.save_eda_plots or config.save_eda_stats:
            ensure_dir(eda_output_dir)
        eda_cfg = EdaConfig(
            output_dir=eda_output_dir,
            max_days=None,
            plot_kinds=["timeseries", "hist", "box", "coverage"],
            save_plots=config.save_eda_plots,
            save_stats=config.save_eda_stats,
            close_plots=not config.show_eda_plots,
        )
        eda_outputs = run_silver_pre_ingestion_eda(new_df, eda_cfg)
        print("EDA overview:")
        print(eda_outputs["overview"])
        if eda_cfg.save_plots or eda_cfg.save_stats:
            print("EDA saved outputs:")
            for k in ["overview_path", "unit_stats_path", "signal_stats_path", "coverage_path", "plot_paths"]:
                print(f"  {k}: {eda_outputs.get(k)}")

    return {
        "output_path": output_path,
        "cache_paths": cache_paths,
        "parks_total": len(meta),
        "parks_downloaded": len(newly_downloaded),
        "parks_cached": len(meta) - len(newly_downloaded),
        "eda_outputs": eda_outputs,
        "dataframe": all_df,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Fetch PVGIS typical-year daily production per park and cache results."
    )
    ap.add_argument("--metadata_path", default=None, help="Path to park_metadata.csv")
    ap.add_argument("--cache_root", default=None, help="Cache root directory (default: pvgis_cache)")
    ap.add_argument("--output_dir", default=None, help="Output directory for combined dataset")
    ap.add_argument("--eda_output_dir", default=None, help="EDA output directory")
    ap.add_argument("--start_year", type=int, default=2015, help="PVGIS start year (>=2005)")
    ap.add_argument("--end_year", type=int, default=2023, help="PVGIS end year (<=2023)")
    ap.add_argument("--loss_pct", type=float, default=18.0, help="PVGIS system loss percentage")
    ap.add_argument("--default_capacity_kwp", type=float, default=100.0, help="Default kWp if missing")
    ap.add_argument("--default_timezone", default="Europe/Athens", help="Default timezone")
    ap.add_argument("--reference_year", type=int, default=2001, help="Typical year reference")
    ap.add_argument("--keep_feb29", action="store_true", help="Keep Feb 29 in typical year")
    ap.add_argument("--force_download", action="store_true", help="Re-download even if cached")
    ap.add_argument("--no_eda", action="store_true", help="Skip EDA on newly downloaded data")
    ap.add_argument("--save_output", action="store_true", help="Save combined dataset to output_dir")
    ap.add_argument("--save_cache", action="store_true", help="Save daily/hourly PVGIS cache files")
    ap.add_argument("--no_cache", action="store_true", help="Disable reading cache files")
    ap.add_argument("--save_eda_plots", action="store_true", help="Save EDA plots")
    ap.add_argument("--save_eda_stats", action="store_true", help="Save EDA stats tables")
    ap.add_argument("--show_eda_plots", action="store_true", help="Keep EDA plots open for display")
    ap.add_argument("--park_ids", default=None, help="Comma-separated park_id list to filter")
    args = ap.parse_args()

    park_ids = None
    if args.park_ids:
        park_ids = [p.strip().lower() for p in args.park_ids.split(",") if p.strip()]

    cfg = TypicalYearConfig(
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
        cache_root=Path(args.cache_root) if args.cache_root else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        eda_output_dir=Path(args.eda_output_dir) if args.eda_output_dir else None,
        use_cache=not args.no_cache,
        save_cache=args.save_cache,
        save_output=args.save_output,
        start_year=args.start_year,
        end_year=args.end_year,
        loss_pct=args.loss_pct,
        default_capacity_kwp=args.default_capacity_kwp,
        default_timezone=args.default_timezone,
        reference_year=args.reference_year,
        drop_feb29=not args.keep_feb29,
        force_download=args.force_download,
        run_eda_on_new=not args.no_eda,
        save_eda_plots=args.save_eda_plots,
        save_eda_stats=args.save_eda_stats,
        show_eda_plots=args.show_eda_plots,
    )

    outputs = build_pvgis_typical_year_dataset(cfg, park_ids=park_ids)
    print("PVGIS typical year summary:")
    print({k: v for k, v in outputs.items() if k not in {"eda_outputs", "dataframe"}})


__all__ = ["TypicalYearConfig", "build_park_meta", "build_pvgis_typical_year_dataset"]


if __name__ == "__main__":
    main()
