# Bronze Ingest Reset & Recovery Guide

## Overview
`bronze_ingest.py` now includes operational controls for managing ingestion state, supporting both normal operations and recovery scenarios (corrupted parquets, mapping changes, testing).

## New Command-Line Flags

### `--allow_duplicates`
**Purpose**: Allow re-ingestion of files with the same hash in the current run.

**Use case**: 
- Recovering from corrupted parquet files without deleting the entire dataset
- Testing changes without clearing historical data
- Re-applying a file with different mapping logic

**Behavior**:
- File is ingested and new parquet files are written even if the same hash was already processed
- Registry and run logs record that duplicate was allowed
- Status message shows `(allow_duplicates)` to indicate override was used

**Example**:
```bash
python src/bronze_ingest.py --data_root data --bronze_root bronze --allow_duplicates
```

---

### `--reset_ops`
**Purpose**: Clear operational metadata without touching Bronze data.

**Deletes**:
- `bronze/_ops/ingest_registry_files.csv` (hash registry)
- `bronze/_ops/run_logs/` (all run logs)
- `data/_locks/bronze_ingest.lock` (lock file if present)

**Use case**:
- Accepting files that were previously rejected as duplicates
- Recovering from registry corruption
- Restarting a stuck ingestion after lock cleanup

**Behavior**:
- Only metadata is deleted; no parquet files are modified
- Next ingestion will treat all files as new (no hash-skip)
- Safe to use; no confirmation required

**Example**:
```bash
python src/bronze_ingest.py --data_root data --bronze_root bronze --reset_ops
```

---

### `--reset_dataset` + `--i_understand`
**Purpose**: Complete reset of Bronze dataset and operations (destructive).

**Deletes**:
- `bronze/scada_1d_signal/` (all year/month partitions and parquet files)
- `bronze/_ops/ingest_registry_files.csv`
- `bronze/_ops/run_logs/`
- `data/_locks/bronze_ingest.lock`

**Use case**:
- Corrupted parquet files across entire dataset
- Restarting ingestion from scratch after major schema changes
- Full cleanup during testing/development

**Behavior**:
- **REQUIRES** `--i_understand` flag to prevent accidental wipes
- All Bronze data and metadata are permanently deleted
- Next ingestion rebuilds dataset from inbox

**Safety guards**:
- Path validation ensures deletions only occur under expected roots
- Refuses to run without explicit `--i_understand` flag
- Prints what will be deleted before acting

**Example**:
```bash
# CAREFUL: This deletes all Bronze data!
python src/bronze_ingest.py \
  --data_root data \
  --bronze_root bronze \
  --reset_dataset \
  --i_understand
```

---

## Recovery Workflows

### Scenario 1: Single Month of Corrupted Parquets
**Problem**: Parquets for Jan 2025 are corrupted but rest are good.

**Solution**:
1. Delete corrupted parquets manually:
   ```bash
   rm -r bronze/scada_1d_signal/year=2025/month=01/
   ```
2. Delete the corresponding source file entry from registry (edit `bronze/_ops/ingest_registry_files.csv`)
3. Place the source file back in `data/inbox/`
4. Run normal ingestion:
   ```bash
   python src/bronze_ingest.py --data_root data --bronze_root bronze
   ```

### Scenario 2: File Rejection Due to Duplicate Hash (but you need to re-process)
**Problem**: A file was marked as duplicate and archived, but you want to re-ingest with different settings.

**Solution**:
1. Copy the file from `data/archived/duplicates/` back to `data/inbox/`
2. Run with `--allow_duplicates`:
   ```bash
   python src/bronze_ingest.py --data_root data --bronze_root bronze --allow_duplicates
   ```

### Scenario 3: Clean Slate After Major Mapping Change
**Problem**: You changed the mapping logic and want to reprocess everything.

**Solution**:
1. Reset everything:
   ```bash
   python src/bronze_ingest.py \
     --data_root data \
     --bronze_root bronze \
     --reset_dataset \
     --i_understand
   ```
2. Place all source files in `data/inbox/`
3. Run normal ingestion:
   ```bash
   python src/bronze_ingest.py --data_root data --bronze_root bronze
   ```

### Scenario 4: Stuck Lock File
**Problem**: Previous run crashed and left a lock file; can't ingest new files.

**Solution**:
```bash
python src/bronze_ingest.py --data_root data --bronze_root bronze --reset_ops
```
This removes the lock file and allows ingestion to proceed.

---

## Registry and Run Log Tracking

### Registry Entries with Duplicates
When `--allow_duplicates` is used, the registry records:
- **status**: `"ingested"` (not skipped_duplicate)
- **message**: `"duplicate hash re-ingested (allow_duplicates)"`

### Run Log Metadata
Run logs now include:
- `duplicate_seen`: whether the hash was already in registry
- `allow_duplicates`: whether the flag was used
- `status_note`: human-readable explanation

Example snippet:
```json
{
  "run_id": "20260124T061511Z",
  "duplicate_seen": true,
  "allow_duplicates": true,
  "status_note": "duplicate hash re-ingested (allow_duplicates)"
}
```

---

## Path Safety
All reset operations validate that:
- Registry file is under `bronze/`
- Dataset folder is under `bronze/`
- Lock file is under `data/`

Attempts to delete paths outside expected roots are blocked with an error.

---

## Audit Trail
Always check:
1. **Registry** (`bronze/_ops/ingest_registry_files.csv`): confirms what was processed
2. **Run logs** (`bronze/_ops/run_logs/run_<timestamp>.json`): detailed metadata including duplicate override
3. **Parquet files** (`bronze/scada_1d_signal/year=YYYY/month=MM/`): actual data partitions
