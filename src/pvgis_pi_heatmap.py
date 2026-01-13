from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm


# ----------------------------
# Header parsing: extract kWp from your park column header
# Supports: "4866kWp", "176 KWp", "0,45MW", "993 Kwp", "993 KWP", etc.
# ----------------------------
_KWP_RE = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>mw|kw(?:p)?)\b", re.IGNORECASE)

def parse_kwp_from_header(col: str, default: float = 100.0) -> float:
    """
    Extract kWp/kW/MW from column header.
    If parsing fails, returns default value (100.0 kWp).
    """
    bracket = re.search(r"\[(.*?)\]", col)
    txt = bracket.group(1) if bracket else col

    m = _KWP_RE.search(txt)
    if not m:
        # Return default instead of raising error
        print(f"Warning: Could not parse kWp/MW from header '{col}', using default {default} kWp")
        return default

    num = float(m.group("num").replace(",", "."))
    unit = m.group("unit").lower()

    if unit == "mw":
        return num * 1000.0
    return num  # treat kW/kWp as kWp numerically


def short_label(col: str) -> str:
    bracket = re.search(r"\[(.*?)\]", col)
    return bracket.group(1) if bracket else col


# ----------------------------
# ASSUMPTION: random Greece coordinates (placeholder only)
# Bounding box (rough): lat 34.8–41.8, lon 19.0–28.3
# ----------------------------
def make_random_greece_meta(
    park_columns: list[str],
    seed: int = 42,
    loss_pct: float = 18.0,     # ASSUMPTION: older parks, total loss %
    timezone: str = "Europe/Athens",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lats = rng.uniform(34.8, 41.8, size=len(park_columns))
    lons = rng.uniform(19.0, 28.3, size=len(park_columns))

    kwp = [parse_kwp_from_header(c) for c in park_columns]

    meta = pd.DataFrame({
        "column_header": park_columns,
        "park_label": [short_label(c) for c in park_columns],
        "kwp": kwp,
        "latitude": lats,
        "longitude": lons,
        "loss_pct": loss_pct,
        "timezone": timezone,
        # No tilt/azimuth provided -> we will use PVGIS optimalangles=True (explicit choice)
        "tilt_deg": np.nan,
        "azimuth_deg": np.nan,
    }).set_index("column_header")

    return meta


# ----------------------------
# PVGIS hourly download + caching
# pvlib docs: peakpower required and loss required if pvcalculation=True;
# P returned only when pvcalculation=True. :contentReference[oaicite:5]{index=5}
# surface_azimuth convention is clockwise from north. :contentReference[oaicite:6]{index=6}
# ----------------------------
def _hash_dict(d: Dict) -> str:
    blob = repr(sorted(d.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def fetch_pvgis_hourly_cached(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    peakpower_kw: float,
    loss_pct: float,
    tilt_deg: Optional[float],
    azimuth_deg: Optional[float],
    cache_dir: Path,
    url: str = "https://re.jrc.ec.europa.eu/api/",
    timeout: int = 60,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)

    optimalangles = (tilt_deg is None) or (azimuth_deg is None)

    key = _hash_dict({
        "lat": lat, "lon": lon,
        "start": start_year, "end": end_year,
        "peakpower_kw": peakpower_kw,
        "loss_pct": loss_pct,
        "optimalangles": optimalangles,
        "tilt_deg": tilt_deg,
        "azimuth_deg": azimuth_deg,
        "url": url,
    })
    cache_path = cache_dir / f"pvgis_{start_year}-{end_year}_{key}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        data, meta = pvlib.iotools.get_pvgis_hourly(
            latitude=lat,
            longitude=lon,
            start=start_year,
            end=end_year,
            pvcalculation=True,
            peakpower=float(peakpower_kw),   # required if pvcalculation=True
            loss=float(loss_pct),            # required if pvcalculation=True
            optimalangles=bool(optimalangles),
            surface_tilt=0.0 if optimalangles else float(tilt_deg),
            surface_azimuth=180.0 if optimalangles else float(azimuth_deg),
            url=url,
            map_variables=True,
            timeout=timeout,
            components=False,
            usehorizon=True,
        )
    except Exception as e:
        print(f"Warning: PVGIS failed for lat={lat}, lon={lon}: {e}")
        # Return empty DataFrame with proper DatetimeIndex and P column (will be all NaN)
        date_index = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='h', tz='UTC')
        return pd.DataFrame({"P": np.nan}, index=date_index)

    if "P" not in data.columns:
        raise RuntimeError("PVGIS output missing 'P'. (P is returned only when pvcalculation=True.)")

    data.to_parquet(cache_path)
    return data


def expected_daily_kwh(hourly: pd.DataFrame, tz: str) -> pd.Series:
    # hourly index from pvlib is typically tz-aware; if not, assume UTC then convert
    if hourly.index.tz is None:
        hourly = hourly.tz_localize("UTC")

    hourly = hourly.tz_convert(tz)

    # PVGIS 'P' is PV system power in W. :contentReference[oaicite:10]{index=10}
    # For hourly steps: daily kWh = sum(P/1000) over the day
    daily = (hourly["P"] / 1000.0).resample("D").sum()
    daily.name = "expected_kwh"
    return daily


def robust_z_mad(x: pd.Series, window: int = 31) -> pd.Series:
    med = x.rolling(window, min_periods=max(7, window // 3)).median()
    mad = (x - med).abs().rolling(window, min_periods=max(7, window // 3)).median()
    scale = 1.4826 * mad.replace(0, np.nan)
    return (x - med) / scale


def compute_pi_anomaly(
    daily_df: pd.DataFrame,
    meta: pd.DataFrame,
    cache_dir: Path,
    pvgis_url: str = "https://re.jrc.ec.europa.eu/api/",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    daily_df: index=date, columns=park energy kWh/day
    meta: index=column_header with lat/lon/kwp/loss_pct/timezone and optional tilt/azimuth
    returns: pi (date x park), score (date x park), flag (date x park)
    """
    date_min = pd.to_datetime(daily_df.index.min()).normalize()
    date_max = pd.to_datetime(daily_df.index.max()).normalize()
    start_year, end_year = int(date_min.year), int(date_max.year)
    
    # PVGIS API only supports years 2005-2023
    # If data is outside this range, map to the nearest supported year
    year_offset = 0
    if end_year > 2023:
        # Data is in future/recent years, map to 2023
        year_offset = start_year - 2023
        start_year, end_year = 2023, 2023
    elif start_year < 2005:
        # Data is too old, map to 2005
        year_offset = start_year - 2005
        start_year, end_year = 2005, 2005
    
    # Adjust date_min and date_max for PVGIS queries (make them UTC aware for slicing)
    pvgis_date_min = pd.to_datetime(date_min - pd.DateOffset(years=year_offset)).tz_localize('UTC')
    pvgis_date_max = pd.to_datetime(date_max - pd.DateOffset(years=year_offset)).tz_localize('UTC')

    expected = {}

    for col in tqdm(daily_df.columns, desc="PVGIS per park"):
        m = meta.loc[col]
        hourly = fetch_pvgis_hourly_cached(
            lat=float(m["latitude"]),
            lon=float(m["longitude"]),
            start_year=start_year,
            end_year=end_year,
            peakpower_kw=float(m["kwp"]),
            loss_pct=float(m["loss_pct"]),
            tilt_deg=None if pd.isna(m.get("tilt_deg", np.nan)) else float(m["tilt_deg"]),
            azimuth_deg=None if pd.isna(m.get("azimuth_deg", np.nan)) else float(m["azimuth_deg"]),
            cache_dir=cache_dir,
            url=pvgis_url,
        )
        # Extract expected daily energy, but adjust dates back to original year
        exp_raw = expected_daily_kwh(hourly, tz=str(m["timezone"])).loc[pvgis_date_min:pvgis_date_max]
        # Shift dates back to original year range
        exp_raw.index = exp_raw.index + pd.DateOffset(years=year_offset)
        # Convert to naive datetime for slicing
        exp_raw.index = exp_raw.index.tz_localize(None) if exp_raw.index.tz is not None else exp_raw.index
        exp = exp_raw.loc[date_min:date_max]
        expected[col] = exp

    exp_df = pd.DataFrame(expected).reindex(daily_df.index)

    pi = daily_df / exp_df
    pi = pi.replace([np.inf, -np.inf], np.nan)

    score = pi.apply(robust_z_mad, axis=0)

    # simple flags for triage (adjust thresholds)
    flag = pd.DataFrame(0, index=pi.index, columns=pi.columns)
    flag[pi < 0.80] = -1
    flag[pi > 1.20] = 1

    return pi, score, flag
