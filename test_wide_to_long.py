#!/usr/bin/env python
"""Direct test of wide_to_long with park_metadata."""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bronze_ingest import wide_to_long
from config import WorkspaceConfig

# Load the processed Excel file (sanitized)
processed_xlsx = list(Path(__file__).parent.glob("data/processing/*.xlsx"))
if not processed_xlsx:
    print("No processed Excel files found")
    # Check inbox
    inbox_xlsx = list(Path(__file__).parent.glob("data/inbox/*.xlsx"))
    if not inbox_xlsx:
        print("No Excel files in inbox either")
        sys.exit(1)
    print(f"Using inbox file: {inbox_xlsx[0]}")
    import openpyxl
    import shutil
    # Copy to processing
    proc_dir = Path(__file__).parent / "data" / "processing"
    proc_dir.mkdir(exist_ok=True)
    processed_file = shutil.copy(inbox_xlsx[0], proc_dir)
    print(f"Copied to processing: {processed_file}")
    processed_xlsx = [processed_file]

excel_path = processed_xlsx[0]
print(f"Loading: {excel_path}")

df = pd.read_excel(excel_path)
print(f"Shape: {df.shape}")

# Load metadata
meta_path = Path(__file__).parent / "mappings" / "park_metadata.csv"
metadata = pd.read_csv(meta_path)
metadata["park_id"] = metadata["park_id"].astype("string").str.strip().str.lower()

print(f"Metadata: {len(metadata)} parks")

# Get config
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestConfig:
    data_root: Path
    inbox: Path
    processing: Path
    archived: Path
    rejected: Path
    bronze_root: Path
    dataset_name: str = "scada_1d_signal"
    mappings_root: Path = None
    timezone_local: str = "Europe/Athens"
    daily_interval_end_is_midnight: bool = True
    allowed_parks: list = None
    ops_dirname: str = "_ops"
    registry_filename: str = "ingest_registry_files.csv"
    runlog_dirname: str = "run_logs"
    sheet_name: int = 0
    csv_sep: str = ","
    csv_encoding: str = "utf-8"

workspace_root = Path(__file__).parent
config = TestConfig(
    data_root=workspace_root / "data",
    inbox=workspace_root / "data" / "inbox",
    processing=workspace_root / "data" / "processing",
    archived=workspace_root / "data" / "archived",
    rejected=workspace_root / "data" / "rejected",
    bronze_root=workspace_root / "bronze",
    mappings_root=workspace_root / "mappings",
    allowed_parks=['4e_energeiaki_176_kwp_likovouni', '4e_energeiaki_4472_kwp_lexaina', '4e_energeiaki_805_kwp_darali', 'hliatoras_474kwp_andravida', 'kalamata_asproxoma_2131kwp', 'ntarali_bonitas_0_45mw', 'ntarali_concept_296kw', 'ntarali_concept_320kw', 'ntarali_concept_592kw', 'ntarali_konenergy_590kw', 'nycontec_993_kwp_giannopouleika', 'palaionaziro_iraio', 'solar_concept_276_kwp_likovouni', 'solar_concept_3721_kwp_lexaina', 'solar_datum_2910_kwp_lexaina', 'solar_datum_864_kwp_darali', 'solar_factory_494kwp_andravida', 'spes_solaris_1527_kwp_darali', 'spes_solaris_1986_kwp_lexaina', 'spes_solaris_201_kwp_konizos', 'spes_solaris_500_kwp_kavasila', 'spes_solaris_805_kwp_konizos']
)

# Call wide_to_long
print("\nCalling wide_to_long...")
try:
    result = wide_to_long(
        df_wide=df,
        cfg=config,
        source_file="test.xlsx",
        source_file_hash="test_hash",
        run_id="test_run",
        park_metadata=metadata
    )
    print(f"Result shape: {result.shape}")
    print(f"Unique parks: {result['park_id'].nunique()}")
    print(f"Parks: {sorted(result['park_id'].unique())}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
