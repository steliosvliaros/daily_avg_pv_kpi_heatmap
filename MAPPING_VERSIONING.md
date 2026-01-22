# Mapping Versioning & Governance

## Overview

This system implements immutable, versioned column mappings with full traceability. No mapping file is ever overwritten—instead, new versions are created with incremental numbering.

## Folder Structure

```
workspace_root/
├── mappings/
│   ├── park_power_mapping_v001.csv    # Immutable version 001
│   ├── park_power_mapping_v002.csv    # Immutable version 002
│   ├── park_power_mapping_v003.csv    # Immutable version 003
│   └── current.txt                     # Active mapping pointer (single line)
└── data/
    └── inbox/
```

## How It Works

### 1. **current.txt** - The Single Source of Truth

`mappings/current.txt` contains exactly one line: the filename of the active mapping.

**Example:**
```
park_power_mapping_v002.csv
```

When ingestion runs, it **always** reads this file to determine which mapping to use.

### 2. **Immutable Versioned Files**

Each mapping file is numbered sequentially:
- `park_power_mapping_v001.csv` - Created initially
- `park_power_mapping_v002.csv` - Created when columns change
- `park_power_mapping_v003.csv` - And so on...

**Rules:**
- Never edit a versioned file
- Never delete old versions
- Old files stay forever for audit and rollback

### 3. **Changing the Active Mapping**

To use a different mapping version:

1. Generate a new mapping file (e.g., `park_power_mapping_v003.csv`)
2. Update `mappings/current.txt` to reference it:
   ```bash
   echo "park_power_mapping_v003.csv" > mappings/current.txt
   ```
3. The next ingestion run automatically uses `v003`

**No code changes required.** No Task Scheduler restart needed.

### 4. **Rolling Back**

If a mapping causes issues:

```bash
echo "park_power_mapping_v001.csv" > mappings/current.txt
```

Next ingestion run uses the old mapping. Instant rollback.

## Traceability & Metadata

Every ingestion run records mapping identity:

### Bronze Registry
`bronze/_ops/ingest_registry_files.csv` now includes:
```
dataset, source_file_hash, source_file, run_id, status, ..., 
mapping_filename, mapping_file_hash, ingested_at_utc, message
```

**Example row:**
```
scada_1d_signal | abc123... | archived/file.xlsx | 20260122T103427Z | ingested | 500 | 5 | 
park_power_mapping_v002.csv | def456... | 2026-01-22T10:34:41+00:00 | ok
```

### Run Logs
`bronze/_ops/run_logs/run_<run_id>.json`:
```json
{
  "run_id": "20260122T103427Z",
  "dataset": "scada_1d_signal",
  "mapping_filename": "park_power_mapping_v002.csv",
  "mapping_file_hash": "def456abc789...",
  "rows_long": 15000,
  "files_written": 5,
  "ingested_at_utc": "2026-01-22T10:34:41+00:00"
}
```

### Bronze Data (Future)
Ideally, every row in Bronze includes:
```
mapping_filename, mapping_file_hash, ...
```

This allows querying: "Which rows were processed with which mapping version?"

## CLI Usage

### Old way (no longer supported):
```bash
python src/bronze_ingest.py --data_root ... --mapping /path/to/mapping.csv
```

### New way (current.txt driven):
```bash
python src/bronze_ingest.py --data_root D:\data --bronze_root D:\bronze
```

No `--mapping` argument needed. Mapping is read from `mappings/current.txt`.

### Optional override (for testing):
```bash
python src/bronze_ingest.py --data_root D:\data --bronze_root D:\bronze \
  --mapping_override park_power_mapping_v001.csv
```

## Notebook Usage

Cell 3 (Bronze Ingestion) automatically:
1. Ensures `mappings/` folder exists
2. Calls `bi.ingest_folder(cfg)` which reads `current.txt`
3. Records mapping metadata in logs and registry

No changes to the notebook when you switch mapping versions—just edit `mappings/current.txt`.

## Updating the Mapping

When columns change in your Excel exports:

1. Run Cell 2 (Sanitize Columns) to regenerate the mapping
2. Move the newly generated `outputs/park_power_mapping.csv` to `mappings/park_power_mapping_v003.csv`
3. Update `mappings/current.txt`:
   ```bash
   echo "park_power_mapping_v003.csv" > mappings/current.txt
   ```
4. Run Cell 3 (Bronze Ingestion) - it now uses `v003`

## File Permissions (Server)

On shared storage:

```bash
# Make mappings folder read-only for everyone except admins
chmod 555 /path/to/mappings/
chmod 444 /path/to/mappings/*.csv
chmod 644 /path/to/mappings/current.txt  # Slightly more permissive for rotation

# Only admin can write:
sudo chown root:admins /path/to/mappings/
```

Or via Windows:
- Right-click `mappings/` → Properties → Security → Advanced
- Grant "Modify" only to admin group
- Everyone else gets "Read" only

## Benefits

✅ **No Overwrite Risk** – Old mappings never lost  
✅ **Easy Rollback** – One-line change to `current.txt`  
✅ **Full Audit Trail** – Every row traceable to its mapping version  
✅ **No Code Changes** – Switch mappings without touching code  
✅ **Safe Server Operations** – Task Scheduler never needs restart  
✅ **Data Governance** – Immutable record of what mapping was used when  

## Example Workflow

```
Day 1: Create initial mapping
  → mappings/park_power_mapping_v001.csv
  → mappings/current.txt points to v001

Day 10: Columns added/changed in Excel
  → Run Cell 2 to regenerate
  → mv outputs/park_power_mapping.csv mappings/park_power_mapping_v002.csv
  → echo "park_power_mapping_v002.csv" > mappings/current.txt
  → Run Cell 3 (uses v002 automatically)

Day 15: Discover v002 has a typo
  → echo "park_power_mapping_v001.csv" > mappings/current.txt
  → Next ingest run uses v001 again (rollback)

Day 16: Create fixed version
  → mappings/park_power_mapping_v003.csv
  → echo "park_power_mapping_v003.csv" > mappings/current.txt
```

## Future Enhancements

- Store mapping versions in git (with full history)
- Automated schema drift detection
- Per-park mapping overrides
- Mapping validation rules (e.g., "unit token always present")
