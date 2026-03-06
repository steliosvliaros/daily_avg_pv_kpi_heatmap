# Gold Layer Usage Example

This demonstrates the new gold layer ingestion pattern with automatic deduplication and datetime integrity.

## Example 1: Ingest Daily Energy to Gold

```python
from src.config import WorkspaceConfig
from src.silver_loader import load_and_ingest_daily_energy_gold, load_gold_daily_energy

# Initialize config
config = WorkspaceConfig()
config.setup_workspace()

# Ingest new data from silver to gold (incremental append)
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    debug=True
)

print(f"Ingestion Status: {result['status']}")
print(f"Rows Written: {result['rows_written']}")
print(f"Last Date: {result['last_date']}")
print(f"Message: {result['message']}")

# Load the gold data
daily_energy = load_gold_daily_energy(gold_root=config.GOLD_DIR, debug=True)
print(f"\nGold daily_energy shape: {daily_energy.shape}")
print(daily_energy.head())
```

## Example 2: Force Full Replace

```python
# Use this when you need to reprocess all data (e.g., after schema changes)
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    force_full_replace=True,  # Ignore watermark and replace all data
    debug=True
)
print(f"Full Replace Result: {result}")
```

## Example 3: Ingest PVGIS Reference to Gold

```python
from src.silver_loader import load_and_ingest_pvgis_reference_gold, load_gold_pvgis_reference

# Ingest PVGIS reference (hash-based, only writes if data changed)
result = load_and_ingest_pvgis_reference_gold(
    pvgis_path=config.PVGIS_OUTPUT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    debug=True
)

print(f"PVGIS Ingestion Status: {result['status']}")
print(f"Rows Written: {result['rows_written']}")
print(f"Data Hash: {result['data_hash']}")
print(f"Message: {result['message']}")

# Load the gold PVGIS data
pvgis_ref = load_gold_pvgis_reference(gold_root=config.GOLD_DIR, debug=True)
print(f"\nGold pvgis_reference shape: {pvgis_ref.shape}")
print(pvgis_ref.head())
```

## Example 4: Date Range Filtering

```python
# Only process specific date range
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    start_date="2024-01-01",
    end_date="2024-12-31",
    debug=True
)
print(f"Date-filtered ingestion: {result}")
```

## Example 5: Park and Signal Filtering

```python
# Only process specific parks/signals
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    park_ids=["energeiaki", "amarynthos"],  # Substring matching
    signals=["active_power", "ptotal"],     # Substring matching
    debug=True
)
print(f"Filtered ingestion: {result}")
```

## Safety Guarantees

1. **No Duplicates**: Watermark tracking ensures each date is processed only once
2. **Datetime Integrity**: Index is always DatetimeIndex, sorted, and deduplicated
3. **Incremental Updates**: Default behavior only processes new data beyond watermark
4. **Idempotent**: Running multiple times with same data won't create duplicates
5. **Observable**: Result dict provides full visibility into what happened

## Migration from Legacy Pattern

### Old Pattern (in notebook Cell 35)
```python
# OLD - Direct write with no deduplication
daily_energy = load_daily_energy_wide(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    persist_final_path=config.GOLD_DIR / "daily_energy.parquet",
    cleanup_chunks=True,
    debug=True
)
```

### New Pattern (recommended)
```python
# NEW - Safe ingestion with watermark tracking
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    debug=True
)

# If you need the DataFrame for immediate use:
daily_energy = load_gold_daily_energy(gold_root=config.GOLD_DIR)
```

## Monitoring Gold Layer State

```python
from pathlib import Path

# Check watermark
watermark_file = config.GOLD_DAILY_ENERGY_OPS / "last_gold_committed.txt"
if watermark_file.exists():
    last_date = watermark_file.read_text().strip()
    print(f"Last committed date: {last_date}")
else:
    print("No watermark found (first run)")

# Check PVGIS hash
hash_file = config.GOLD_PVGIS_REFERENCE_OPS / "pvgis_reference_data_hash.txt"
if hash_file.exists():
    data_hash = hash_file.read_text().strip()
    print(f"PVGIS data hash: {data_hash}")
else:
    print("No PVGIS hash found (first run)")

# Check metadata
import json
metadata_file = config.GOLD_DAILY_ENERGY_OPS / "metadata.json"
if metadata_file.exists():
    metadata = json.loads(metadata_file.read_text())
    print(f"Table: {metadata['table_name']}")
    print(f"Version: {metadata['version']}")
    print(f"Refresh strategy: {metadata['refresh_strategy']['type']}")
```
