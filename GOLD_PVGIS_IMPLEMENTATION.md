# Gold Layer PVGIS Reference Implementation

## Overview

Extended the gold layer ingestion pattern to include PVGIS reference data, following the same architecture as daily energy ingestion.

## Implementation Details

### 1. PVGIS Workflow Integration (`src/pvgis_workflow.py`)

**Added Configuration Options:**
```python
# Gold layer options
ingest_to_gold: bool = True
force_gold_replace: bool = False
```

**Added Result Tracking:**
```python
# Gold ingestion results
gold_ingestion_result: Optional[Dict[str, Any]] = None
```

**Gold Ingestion Logic:**
After PVGIS data download/cache hit, the workflow now:
1. Checks if `ingest_to_gold=True`
2. Calls `load_and_ingest_pvgis_reference_gold()` 
3. Uses hash-based change detection (no unnecessary rewrites)
4. Stores result in `PVGISPipelineResult.gold_ingestion_result`

### 2. Existing Gold Functions (`src/silver_loader.py`)

The following functions were already implemented (from previous work):

**Ingestion Function:**
```python
load_and_ingest_pvgis_reference_gold(
    pvgis_path, metadata_path, gold_root,
    park_ids=None, signals=None, status_effective="true",
    force_replace=False, debug=False
) -> dict
```
- Loads PVGIS data using `load_pvgis_filtered_wide()`
- Calls `ingest_pvgis_reference_safe()` with hash-based detection
- Returns: `{status, rows_written, data_hash, message}`

**Loader Function:**
```python
load_gold_pvgis_reference(gold_root, debug=False) -> pd.DataFrame
```
- Loads existing PVGIS reference data from gold layer
- Returns DataFrame with date index and park columns

### 3. Core Gold Ingestion (`src/gold_ingest.py`)

**Hash-Based Change Detection:**
```python
ingest_pvgis_reference_safe(pvgis_df, config, force_replace=False, debug=False)
```
- Computes SHA256 hash of DataFrame content
- Compares with last committed hash
- Only writes if data has changed or `force_replace=True`
- Stores hash in `_ops/last_gold_committed_hash.txt`

### 4. Notebook Integration (Cell 36)

**Updated PVGIS Cell:**
```python
# Configure PVGIS pipeline with gold ingestion
pvgis_config = pw.PVGISPipelineConfig(
    workspace_config=config,
    # ... existing config ...
    ingest_to_gold=True,        # ← NEW: Enable gold ingestion
    force_gold_replace=False,   # ← NEW: Use hash-based detection
)

result = pw.run_pvgis_pipeline(pvgis_config)

# Display gold ingestion results
if result.gold_ingestion_result:
    print(f"Status: {result.gold_ingestion_result.get('status')}")
    print(f"Rows Written: {result.gold_ingestion_result.get('rows_written')}")
    print(f"Message: {result.gold_ingestion_result.get('message')}")

# Load PVGIS reference from gold
pvgis_reference = load_gold_pvgis_reference(gold_root=config.GOLD_DIR, debug=True)
```

## Usage Pattern

### First Run (Initial Ingestion)
```python
result = pw.run_pvgis_pipeline(pvgis_config)
# Output: Status: appended, Rows Written: 8395, Message: Initial write
```

### Subsequent Runs (No Changes)
```python
result = pw.run_pvgis_pipeline(pvgis_config)
# Output: Status: unchanged, Rows Written: 0, Message: Data unchanged (hash match)
```

### Force Replace
```python
pvgis_config.force_gold_replace = True
result = pw.run_pvgis_pipeline(pvgis_config)
# Output: Status: replaced, Rows Written: 8395, Message: Force replace
```

## Data Flow

```
PVGIS API/Cache
    ↓
pvgis_workflow.run_pvgis_pipeline()
    ↓
load_and_ingest_pvgis_reference_gold()
    ├─ load_pvgis_filtered_wide() → DataFrame
    └─ ingest_pvgis_reference_safe()
        ├─ compute_dataframe_hash() → SHA256
        ├─ compare with previous hash
        └─ write if changed → gold/pvgis_reference/*.parquet
    ↓
load_gold_pvgis_reference() → DataFrame (for downstream use)
```

## Gold Layer Structure

```
gold/
├── daily_energy/              # Daily energy aggregations
│   ├── *.parquet              # Partitioned data
│   └── _ops/
│       ├── metadata.json
│       └── last_gold_committed.txt  # Watermark (timestamp)
│
└── pvgis_reference/           # PVGIS reference data
    ├── *.parquet              # Full dataset (365 days x N parks)
    └── _ops/
        ├── metadata.json
        └── last_gold_committed_hash.txt  # Hash tracking
```

## Key Differences: Daily Energy vs PVGIS

| Aspect | Daily Energy | PVGIS Reference |
|--------|--------------|-----------------|
| **Update Strategy** | Incremental (watermark-based) | Full replace (hash-based) |
| **Tracking** | Last date committed | Data content hash |
| **Typical Rows** | 4,000+ (growing daily) | 8,395 (365 days × 23 parks) |
| **Change Frequency** | Daily | Rarely (only when parks added/removed) |
| **Ingestion Logic** | Append new dates only | Replace if hash differs |

## Benefits

1. **Consistent Pattern**: Both gold tables use same infrastructure
2. **Efficient**: Hash-based detection avoids unnecessary rewrites
3. **Safe**: No data duplication or corruption
4. **Integrated**: Single workflow handles download + gold ingestion
5. **Debuggable**: Comprehensive logging at each step

## Testing

Run Cell 36 to test:
1. First run: Should ingest ~8,395 rows (365 days × 23 parks)
2. Second run: Should show "Data unchanged (hash match)"
3. Check gold folder: `gold/pvgis_reference/*.parquet` should exist
4. Verify hash file: `gold/pvgis_reference/_ops/last_gold_committed_hash.txt`

## Next Steps

- ✅ Daily energy gold ingestion (watermark-based)
- ✅ PVGIS reference gold ingestion (hash-based)
- 🔄 Use gold tables in power_ratio_workflow
- 🔄 Update downstream analysis to load from gold layer
