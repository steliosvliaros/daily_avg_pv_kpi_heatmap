from __future__ import annotations

import csv
import importlib
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".csv", ".txt"}


def _load_park_capacity_map(workspace_root: Path) -> Dict[str, float]:
    """Load park capacity (kWp) per park_id from park_metadata.csv if available."""
    path = workspace_root / "mappings" / "park_metadata.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "park_id" not in df.columns or "capacity_kwp" not in df.columns:
        return {}

    cap_map: Dict[str, float] = {}
    for _, row in df.iterrows():
        park_id = str(row.get("park_id", "")).strip()
        if not park_id or park_id.lower() == "nan":
            continue
        capacity = row.get("capacity_kwp")
        try:
            cap_map[park_id] = float(capacity) if pd.notna(capacity) else 1.0
        except (TypeError, ValueError):
            cap_map[park_id] = 1.0
    return cap_map


def _build_sanitizer(workspace_root: Path, default_capacity_kwp: float, prompt_missing_capacity: bool):
    """Create a ScadaColumnSanitizer with capacity cache preloaded from park metadata."""
    if str(workspace_root / "src") not in sys.path:
        sys.path.append(str(workspace_root / "src"))

    from src import scada_column_sanitizer as scs

    importlib.reload(scs)
    
    # Pass park metadata path to config
    park_metadata_path = workspace_root / "mappings" / "park_metadata.csv"
    
    cfg = scs.SanitizeConfig(
        prompt_missing_capacity=prompt_missing_capacity,
        default_capacity_kwp=default_capacity_kwp,
        park_metadata_path=park_metadata_path if park_metadata_path.exists() else None,
    )
    sanitizer = scs.ScadaColumnSanitizer(config=cfg)
    
    # Note: capacity cache is now loaded automatically from metadata in __init__
    return sanitizer


def _collect_inbox_columns(inbox_dir: Path, workspace_root: Path) -> Iterable[str]:
    """Read only column names from inbox files (first sheet for Excel)."""
    columns = []
    if not inbox_dir.exists():
        return columns

    if str(workspace_root / "src") not in sys.path:
        sys.path.append(str(workspace_root / "src"))

    from src import scada_column_sanitizer as scs

    for p in inbox_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        cols = scs.read_columns_only(str(p), sheet_name=0)
        columns.extend(cols)
    return columns


def recompute_sanitized_columns(
    workspace_root: Path | str | None = None,
    resanitize_current: bool = False,
    prompt_missing_capacity: bool = False,
    default_capacity_kwp: float = 1.0,
) -> Path:
    """
    Recompute sanitized columns from inbox files and write a new versioned mapping.

    - Uses park_metadata.csv to prefill capacity cache (no user prompts)
    - Supports resanitizing from scratch or incrementally extending existing mapping
    """
    root = Path(workspace_root) if workspace_root else Path.cwd().parent.resolve()
    inbox_dir = root / "data" / "inbox"
    mappings_dir = root / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    current_txt = mappings_dir / "current.txt"

    columns = list(dict.fromkeys(_collect_inbox_columns(inbox_dir, workspace_root=root)))
    if not columns and not resanitize_current:
        raise FileNotFoundError(f"No files found in inbox: {inbox_dir}")

    sanitizer = _build_sanitizer(
        workspace_root=root,
        default_capacity_kwp=default_capacity_kwp,
        prompt_missing_capacity=prompt_missing_capacity,
    )

    from src import scada_column_sanitizer as scs

    base_mapping: Dict[str, str] = {}
    mapping_files = sorted(mappings_dir.glob("park_power_mapping_v*.csv"))
    current_mapping_path = None
    if current_txt.exists():
        current_mapping_file = current_txt.read_text(encoding="utf-8").strip()
        current_mapping_path = mappings_dir / current_mapping_file

    if mapping_files:
        for f in mapping_files:
            base_mapping.update(sanitizer.load_mapping_csv(f))
        print(f"Loaded {len(mapping_files)} mapping versions, {len(base_mapping)} total entries.")
    elif current_mapping_path and current_mapping_path.exists():
        base_mapping = sanitizer.load_mapping_csv(current_mapping_path)
        print(f"Loaded existing mapping: {current_mapping_path.name}")
    else:
        print("No existing mapping found (first run)")

    originals = list(dict.fromkeys(list(base_mapping.keys()) + columns))
    if not originals:
        raise RuntimeError("No columns available to sanitize. Check inbox or mapping files.")

    if resanitize_current:
        sanitized, mapping = sanitizer.sanitize_columns(originals, existing_mapping={})
        full_mapping = mapping
    else:
        sanitized, mapping = sanitizer.sanitize_columns(originals, existing_mapping=base_mapping)
        full_mapping = dict(base_mapping)
        full_mapping.update(mapping)

    print("\nRecomputed sanitized columns. Showing first 10:")
    for c in sanitized[:10]:
        print(" -", c)
    print(f"Total mappings: {len(full_mapping)}")

    if base_mapping and full_mapping == base_mapping:
        print("No new columns detected; skipping new mapping version.")
        if current_mapping_path and current_mapping_path.exists():
            return current_mapping_path
        if mapping_files:
            return mapping_files[-1]
        raise RuntimeError("No existing mapping file found to return.")

    version_pattern = re.compile(r"park_power_mapping_v(\d+)\.csv")
    versions = []
    for f in mappings_dir.glob("park_power_mapping_v*.csv"):
        m = version_pattern.match(f.name)
        if m:
            versions.append(int(m.group(1)))
    next_version = max(versions, default=0) + 1

    new_mapping_filename = f"park_power_mapping_v{next_version:03d}.csv"
    new_mapping_path = mappings_dir / new_mapping_filename

    with open(new_mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "sanitized"])
        for orig in sorted(full_mapping):
            writer.writerow([orig, full_mapping[orig]])

    current_txt.write_text(new_mapping_filename, encoding="utf-8")
    print(f"\nSaved new mapping version: {new_mapping_filename}")
    print(f"Updated mappings/current.txt -> {new_mapping_filename}")

    return new_mapping_path


__all__ = [
    "recompute_sanitized_columns",
]
