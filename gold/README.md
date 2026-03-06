# Gold Layer - Business-Ready Aggregated Data

The gold layer contains curated, aggregated data ready for analysis and reporting. This layer follows the same safety patterns as bronze and silver layers to prevent data duplication and ensure datetime integrity.

## Directory Structure

```
gold/
├── daily_energy/                    # Daily aggregated energy per park
│   ├── _ops/                        # Operational metadata
│   │   ├── metadata.json            # Table schema and refresh logic
│   │   └── last_gold_committed.txt  # Watermark for incremental updates
│   └── daily_energy.parquet         # Wide format: date × park_id
│
├── pvgis_reference/                 # PVGIS typical year reference
│   ├── _ops/                        # Operational metadata
│   │   ├── metadata.json            # Table schema and refresh logic
│   │   └── pvgis_reference_data_hash.txt  # Hash for change detection
│   └── pvgis_reference.parquet      # Wide format: date × park_id
```

## Tables

### 1. daily_energy

**Description**: Daily aggregated energy (kWh) per park in wide format

**Schema**:
- Index: `date` (DatetimeIndex, local timezone Europe/Athens)
- Columns: `park_id` (each column = one park)
- Values: Daily energy in kWh

**Source**: Silver layer (`silver/year=*/month=*/part-*.parquet`)

**Transformation**: `load_daily_energy_wide()` aggregates 15-minute power readings to daily energy with unit-aware conversion:
- W → kW → kWh (with interval scaling)
- kW → kWh (with interval scaling)  
- Wh → kWh (no interval scaling)
- kWh → kWh (no scaling)

**Refresh Strategy**: Partition-based incremental updates
- Tracks last processed date via watermark file
- Only appends new dates beyond the watermark
- Prevents duplicate aggregations
- Respects datetime boundaries

**Usage Example**:
```python
from src.config import WorkspaceConfig
from src.silver_loader import load_and_ingest_daily_energy_gold, load_gold_daily_energy

config = WorkspaceConfig()

# Ingest new data from silver to gold
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    debug=True
)
print(result)  # {'status': 'appended', 'rows_written': 150, 'last_date': ..., 'message': '...'}

# Load existing gold data
daily_energy = load_gold_daily_energy(gold_root=config.GOLD_DIR, debug=True)
print(daily_energy.shape)
print(daily_energy.head())
```

### 2. pvgis_reference

**Description**: PVGIS typical meteorological year reference data for each park location

**Schema**:
- Index: `date` (DatetimeIndex within typical year)
- Columns: `park_id` (each column = one park)
- Values: PVGIS reference metrics (power, irradiance, temperature)

**Source**: External PVGIS cache (`pvgis/pvgis_typical_year/*.parquet`)

**Transformation**: `load_pvgis_filtered_wide()` loads and filters cached PVGIS data per park location

**Refresh Strategy**: Hash-based full replacement
- Computes hash of source data to detect changes
- Only writes if data has changed (hash mismatch)
- Full replace when PVGIS data is updated or new parks added

**Usage Example**:
```python
from src.config import WorkspaceConfig
from src.silver_loader import load_and_ingest_pvgis_reference_gold, load_gold_pvgis_reference

config = WorkspaceConfig()

# Ingest PVGIS data to gold
result = load_and_ingest_pvgis_reference_gold(
    pvgis_path=config.PVGIS_OUTPUT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    debug=True
)
print(result)  # {'status': 'replaced', 'rows_written': 365, 'data_hash': '...', 'message': '...'}

# Load existing gold data
pvgis_ref = load_gold_pvgis_reference(gold_root=config.GOLD_DIR, debug=True)
print(pvgis_ref.shape)
print(pvgis_ref.head())
```

## Safety Features

### 1. Watermark-Based Deduplication (daily_energy)

The `daily_energy` table uses watermark tracking to prevent duplicate aggregations:

1. **First Run**: No watermark exists
   - Aggregates all available silver data
   - Writes result to `daily_energy.parquet`
   - Records last date to `last_gold_committed.txt`

2. **Subsequent Runs**: Watermark exists
   - Reads watermark: e.g., `2024-12-31`
   - Filters silver data to dates > watermark
   - Aggregates only new data
   - Merges with existing gold data
   - Updates watermark to new last date

3. **Force Full Replace**: `force_full_replace=True`
   - Ignores watermark
   - Replaces entire dataset
   - Updates watermark

### 2. Hash-Based Change Detection (pvgis_reference)

The `pvgis_reference` table uses hash-based change detection:

1. **First Run**: No hash exists
   - Loads PVGIS data
   - Computes hash of DataFrame content
   - Writes to `pvgis_reference.parquet`
   - Records hash to `pvgis_reference_data_hash.txt`

2. **Subsequent Runs**: Hash exists
   - Loads PVGIS data
   - Computes new hash
   - Compares with stored hash
   - **If match**: Skip write (data unchanged)
   - **If mismatch**: Replace data and update hash

3. **Force Replace**: `force_replace=True`
   - Ignores hash comparison
   - Replaces data unconditionally

### 3. Datetime Integrity

Both tables ensure datetime integrity:
- Index must be `DatetimeIndex`
- Sorted by index before writing
- Duplicates removed (keep last occurrence)
- Timezone-aware timestamps preserved

## Configuration

Gold layer paths are defined in `src/config.py`:

```python
class WorkspaceConfig:
    def __init__(self, workspace_root=None):
        # ...
        self.GOLD_DIR = self.WORKSPACE_ROOT / "gold"
        self.GOLD_DAILY_ENERGY = self.GOLD_DIR / "daily_energy"
        self.GOLD_DAILY_ENERGY_OPS = self.GOLD_DAILY_ENERGY / "_ops"
        self.GOLD_PVGIS_REFERENCE = self.GOLD_DIR / "pvgis_reference"
        self.GOLD_PVGIS_REFERENCE_OPS = self.GOLD_PVGIS_REFERENCE / "_ops"
```

All directories are auto-created by `config.setup_workspace()`.

## Implementation Modules

- **`src/gold_ingest.py`**: Core ingestion logic with watermark and hash tracking
- **`src/silver_loader.py`**: High-level functions for loading and ingesting gold data
  - `load_and_ingest_daily_energy_gold()`: Load from silver and ingest to gold
  - `load_gold_daily_energy()`: Load existing gold data
  - `load_and_ingest_pvgis_reference_gold()`: Load PVGIS and ingest to gold
  - `load_gold_pvgis_reference()`: Load existing PVGIS gold data

## Best Practices

1. **Use the High-Level Functions**: Prefer `load_and_ingest_*_gold()` over manual ingestion
2. **Enable Debug Mode**: Use `debug=True` during development to see ingestion details
3. **Check Return Values**: Inspect the result dict for status, rows_written, etc.
4. **Incremental Updates**: Default behavior appends only new data (efficient)
5. **Full Refresh**: Use `force_full_replace=True` sparingly (after schema changes, data corrections)
6. **Monitor Watermarks**: Check `_ops/*.txt` files to verify ingestion state

## Migration from Legacy persist_final_path

If you were using `load_daily_energy_wide()` with `persist_final_path`, migrate to the new gold ingestion pattern:

**Before**:
```python
daily_energy = load_daily_energy_wide(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    persist_final_path=config.GOLD_DIR / "daily_energy.parquet",
    cleanup_chunks=True,
    debug=True
)
```

**After**:
```python
result = load_and_ingest_daily_energy_gold(
    silver_root=config.SILVER_ROOT,
    metadata_path=config.PARK_METADATA_CSV,
    gold_root=config.GOLD_DIR,
    debug=True
)
```

**Benefits**:
- Automatic deduplication via watermark tracking
- Incremental updates (only process new data)
- Consistent with bronze/silver ingestion patterns
- Better observability via result dict

## Extending the Gold Layer

To add new gold tables:

1. **Create subdirectory**: `gold/new_table/_ops/`
2. **Add metadata.json**: Document schema, source, refresh strategy
3. **Update config.py**: Add paths to `WorkspaceConfig`
4. **Implement ingestion logic**: Add functions to `src/gold_ingest.py`
5. **Add high-level API**: Add convenience functions to `src/silver_loader.py`
6. **Update this README**: Document the new table

## Troubleshooting

### Issue: "No new data beyond watermark"

**Cause**: All silver data has already been processed

**Solution**: 
- Check watermark: `cat gold/daily_energy/_ops/last_gold_committed.txt`
- Verify silver data range: inspect `silver/year=*/month=*` partitions
- To reprocess all data: use `force_full_replace=True`

### Issue: "Data unchanged (hash match)"

**Cause**: PVGIS reference data hasn't changed since last ingestion

**Solution**:
- Normal behavior, no action needed
- To force rewrite: use `force_replace=True`

### Issue: AttributeError: 'WorkspaceConfig' object has no attribute 'GOLD_DAILY_ENERGY'

**Cause**: Config was initialized before the gold paths were added

**Solution**: Rerun notebook setup cell to reinitialize config with new attributes

## Version History

- **v1.0.0** (2026-02-15): Initial gold layer implementation
  - daily_energy table with watermark-based ingestion
  - pvgis_reference table with hash-based change detection
  - Metadata/manifest files for each table
  - Integration with WorkspaceConfig
