from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

DEFAULT_DATASET_NAME = "scada_1d_signal"
UNIT_TOKEN_RE = re.compile(r"__u_([a-z0-9_]+)$")
RULE_VERSION = "v2"


@dataclass
class SanityConfig:
    bronze_root: Path
    mappings_root: Path
    output_dir: Path
    dataset_name: str = DEFAULT_DATASET_NAME
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)
    max_files: Optional[int] = None


@dataclass(frozen=True)
class UnitRule:
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    note: str = ""


def default_unit_rules() -> Dict[str, UnitRule]:
    return {
        "u_pct": UnitRule(min_value=0.0, max_value=1.0, note="fraction range"),
        "u_w_m_2": UnitRule(min_value=0.0, max_value=1500.0, note="solar irradiance"),
        "u_kw": UnitRule(min_value=0.0, note="active power non-negative"),
        "u_kva": UnitRule(min_value=0.0, note="apparent power non-negative"),
        "u_kvar": UnitRule(min_value=0.0, note="reactive power non-negative"),
        "u_kwh": UnitRule(min_value=0.0, note="energy non-negative"),
        "u_kvah": UnitRule(min_value=0.0, note="energy non-negative"),
        "u_kvarh": UnitRule(min_value=0.0, note="energy non-negative"),
        "u_a": UnitRule(min_value=0.0, note="current magnitude non-negative"),
        "u_vac": UnitRule(min_value=0.0, note="voltage non-negative"),
        "u_hz": UnitRule(min_value=0.0, note="frequency non-negative"),
        "u_m_s_1": UnitRule(min_value=0.0, max_value=60.0, note="wind speed (assumed max 60 m/s)"),
        "u_num": UnitRule(min_value=0.0, note="non-negative numeric"),
    }


def get_current_mapping_path(mappings_root: Path) -> Path:
    current_file = mappings_root / "current.txt"
    if not current_file.exists():
        raise FileNotFoundError(f"Mapping pointer file not found: {current_file}")
    mapping_name = current_file.read_text(encoding="utf-8").strip()
    mapping_path = mappings_root / mapping_name
    if not mapping_path.exists():
        raise FileNotFoundError(f"Active mapping file not found: {mapping_path}")
    return mapping_path


def compute_mapping_hash(mapping_path: Path) -> str:
    sha256 = hashlib.sha256()
    with mapping_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def read_units_from_mapping(mapping_path: Path) -> List[str]:
    units = set()
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) >= 2 and header[0].lower() == "original" and header[1].lower() == "sanitized":
            pass
        elif header and len(header) >= 2:
            units.update(_units_from_sanitized(header[1]))

        for row in reader:
            if len(row) < 2:
                continue
            units.update(_units_from_sanitized(row[1]))

    return sorted(u for u in units if u)


def _units_from_sanitized(sanitized: str) -> List[str]:
    sanitized = sanitized.strip()
    if sanitized == "datetime":
        return []
    m = UNIT_TOKEN_RE.search(sanitized)
    if not m:
        return ["u_unknown"]
    return [f"u_{m.group(1)}"]


def list_bronze_parquet_files(
    bronze_root: Path,
    dataset_name: str,
    max_files: Optional[int] = None,
) -> List[Path]:
    base = bronze_root / dataset_name
    files = sorted(base.glob("**/*.parquet"))
    if max_files:
        files = files[:max_files]
    return files


def load_bronze_values(files: Iterable[Path]) -> pd.DataFrame:
    dfs = []
    for p in files:
        df = pd.read_parquet(p, columns=["unit", "value"])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["unit", "value"])
    return pd.concat(dfs, ignore_index=True)


def _quantile_label(q: float) -> str:
    pct = int(round(q * 100))
    return f"p{pct:02d}"


def compute_unit_benchmarks(
    df: pd.DataFrame,
    units: Iterable[str],
    quantiles: Iterable[float],
    unit_rules: Optional[Dict[str, UnitRule]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["unit"])

    df = df.copy()
    df["unit"] = df["unit"].astype("string").str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    unit_set = None
    if units:
        unit_set = {u.strip().lower() for u in units}
        df = df[df["unit"].isin(unit_set)]

    if df.empty:
        return pd.DataFrame(columns=["unit"])

    group = df.groupby("unit")["value"]
    stats = group.agg(count="size", min="min", max="max", mean="mean", std="std")

    q_list = list(quantiles)
    q_labels = [_quantile_label(q) for q in q_list]
    q_df = group.quantile(q_list).unstack()
    q_df.columns = q_labels

    out = stats.join(q_df, how="left")

    rules = unit_rules or {}
    invalid_rows: List[Dict[str, object]] = []
    for unit, series in group:
        rule = rules.get(unit)
        if rule:
            low_mask = False
            high_mask = False
            if rule.min_value is not None:
                low_mask = series < rule.min_value
            if rule.max_value is not None:
                high_mask = series > rule.max_value
            invalid_mask = low_mask | high_mask
            invalid_low = int(low_mask.sum()) if isinstance(low_mask, pd.Series) else 0
            invalid_high = int(high_mask.sum()) if isinstance(high_mask, pd.Series) else 0
            invalid_total = int(invalid_mask.sum()) if isinstance(invalid_mask, pd.Series) else 0
            invalid_note = rule.note or "rule"
            min_allowed = rule.min_value
            max_allowed = rule.max_value
        else:
            invalid_low = 0
            invalid_high = 0
            invalid_total = 0
            invalid_note = "no_rule"
            min_allowed = None
            max_allowed = None

        invalid_rows.append(
            {
                "unit": unit,
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
                "invalid_low_count": invalid_low,
                "invalid_high_count": invalid_high,
                "invalid_count": invalid_total,
                "invalid_fraction": invalid_total / len(series) if len(series) else 0.0,
                "rule_note": invalid_note,
            }
        )

    invalid_df = pd.DataFrame(invalid_rows).set_index("unit")
    out = out.join(invalid_df, how="left")
    out = out.reset_index()

    if unit_set:
        all_units = sorted(unit_set)
        out = out.set_index("unit").reindex(all_units)
        missing_mask = out["count"].isna() if "count" in out.columns else pd.Series(True, index=out.index)

        if rules:
            min_map = {u: r.min_value for u, r in rules.items()}
            max_map = {u: r.max_value for u, r in rules.items()}
            note_map = {u: (r.note or "rule") for u, r in rules.items()}
            if "min_allowed" in out.columns:
                min_series = pd.Series(min_map, dtype="float64")
                out["min_allowed"] = pd.to_numeric(out["min_allowed"], errors="coerce")
                out["min_allowed"] = out["min_allowed"].combine_first(min_series)
            if "max_allowed" in out.columns:
                max_series = pd.Series(max_map, dtype="float64")
                out["max_allowed"] = pd.to_numeric(out["max_allowed"], errors="coerce")
                out["max_allowed"] = out["max_allowed"].combine_first(max_series)
            if "rule_note" in out.columns:
                note_series = pd.Series(note_map, dtype="string")
                out["rule_note"] = out["rule_note"].astype("string").combine_first(note_series)

        if "rule_note" in out.columns:
            out["rule_note"] = out["rule_note"].where(out["rule_note"].notna(), "no_rule")

        for col in ["invalid_low_count", "invalid_high_count", "invalid_count", "invalid_fraction"]:
            if col in out.columns:
                out.loc[missing_mask, col] = 0

        if "count" in out.columns:
            out.loc[missing_mask, "count"] = 0

        out = out.reset_index()

    return out


def write_benchmarks(
    df: pd.DataFrame,
    output_dir: Path,
    metadata: Dict[str, str],
    filename: str = "unit_sanity_benchmarks.csv",
) -> Tuple[Optional[Path], Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    json_path = output_dir / f"{csv_path.stem}.meta.json"

    if df.empty:
        return None, None

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, json_path


def run_unit_sanity_check(
    *,
    bronze_root: Path,
    mappings_root: Path,
    output_dir: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    quantiles: Iterable[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    max_files: Optional[int] = None,
    unit_rules: Optional[Dict[str, UnitRule]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Optional[Path], Optional[Path]]:
    mapping_path = get_current_mapping_path(mappings_root)
    mapping_hash = compute_mapping_hash(mapping_path)
    units = read_units_from_mapping(mapping_path)

    files = list_bronze_parquet_files(bronze_root, dataset_name, max_files=max_files)
    df = load_bronze_values(files)
    rules = unit_rules or default_unit_rules()
    bench = compute_unit_benchmarks(df, units, quantiles, unit_rules=rules)

    metadata = {
        "dataset_name": dataset_name,
        "bronze_root": str(bronze_root),
        "mapping_filename": mapping_path.name,
        "mapping_file_hash": mapping_hash,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_read": str(len(files)),
        "rows_used": str(len(df)),
        "rule_version": RULE_VERSION,
        "rules_count": str(len(rules)),
    }

    csv_path, json_path = write_benchmarks(bench, output_dir, metadata)
    return bench, metadata, csv_path, json_path


def build_cfg(args) -> SanityConfig:
    return SanityConfig(
        bronze_root=Path(args.bronze_root).resolve(),
        mappings_root=Path(args.mappings_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        dataset_name=args.dataset_name,
        max_files=args.max_files,
        quantiles=tuple(float(q) for q in args.quantiles.split(",")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute unit sanity benchmarks from bronze data.")
    ap.add_argument("--bronze_root", required=True, help="Bronze root folder")
    ap.add_argument("--mappings_root", required=True, help="Mappings folder containing current.txt")
    ap.add_argument("--output_dir", required=True, help="Output directory for benchmarks")
    ap.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME, help="Bronze dataset name folder")
    ap.add_argument("--quantiles", default="0.01,0.05,0.5,0.95,0.99", help="Comma-separated quantiles")
    ap.add_argument("--max_files", type=int, default=None, help="Optional limit on parquet files read")
    args = ap.parse_args()

    cfg = build_cfg(args)
    bench, meta, csv_path, json_path = run_unit_sanity_check(
        bronze_root=cfg.bronze_root,
        mappings_root=cfg.mappings_root,
        output_dir=cfg.output_dir,
        dataset_name=cfg.dataset_name,
        quantiles=cfg.quantiles,
        max_files=cfg.max_files,
    )

    if bench.empty:
        print("No data found. Benchmarks not written.")
        return

    print(f"Wrote benchmarks: {csv_path}")
    print(f"Wrote metadata: {json_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_DATASET_NAME",
    "RULE_VERSION",
    "SanityConfig",
    "UnitRule",
    "default_unit_rules",
    "get_current_mapping_path",
    "compute_mapping_hash",
    "read_units_from_mapping",
    "list_bronze_parquet_files",
    "load_bronze_values",
    "compute_unit_benchmarks",
    "run_unit_sanity_check",
]
