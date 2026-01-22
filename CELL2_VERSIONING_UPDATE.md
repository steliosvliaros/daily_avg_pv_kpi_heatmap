# Cell 2 & scada_column_sanitizer - Versioning Integration

**Updated:** January 22, 2026

## Changes Made

### Cell 2 (Mapping Regeneration)

**Before:**
- Saved mapping to `outputs/park_power_mapping.csv` (would overwrite on each run)
- No version tracking

**After:**
- Loads existing mapping from `mappings/current.txt` (active version)
- Regenerates columns preserving unchanged mappings
- Automatically determines next version number (v001 → v002 → v003)
- Saves to `mappings/park_power_mapping_vNNN.csv` (immutable)
- Updates `mappings/current.txt` to point to new version
- Displays clear feedback about versioning status

### Key Features

1. **Automatic Version Detection**
   ```python
   def get_next_version() -> int:
       # Finds highest existing version from mappings/*.csv
       # Returns next sequential number
   ```

2. **Mapping Preservation**
   - Loads existing mapping from current active version
   - Passes it to sanitizer via `existing_mapping=...`
   - Unchanged columns are not re-prompted/re-processed

3. **Atomic Version Switch**
   - New mapping saved to versioned filename
   - `current.txt` updated to point to new version
   - One step to activate new mapping (no manual file copying needed)

4. **User Feedback**
   ```
   ✅ Saved new mapping version: park_power_mapping_v002.csv
   ✅ Updated mappings/current.txt → park_power_mapping_v002.csv
   
   📌 Mapping versioning system:
      Location: mappings/park_power_mapping_v002.csv
      Active: Yes (current.txt points to this version)
      To rollback: edit mappings/current.txt
   ```

## Integration with Bronze Ingestion

Cell 2 and Cell 3 now work together seamlessly:

1. **Cell 2 (Mapping Update)**
   - Generates new mapping version
   - Updates `current.txt`

2. **Cell 3 (Bronze Ingestion)**
   - Reads `mappings/current.txt`
   - Uses whatever mapping Cell 2 just activated
   - Records mapping metadata in registry and logs
   - No manual mapping path needed

## Workflow Example

### Scenario 1: Column Changes in Excel

```
1. Run Cell 2
   → Detects new columns
   → Prompts for park capacities (if missing)
   → Saves as park_power_mapping_v002.csv
   → current.txt → v002

2. Run Cell 3
   → Reads current.txt
   → Uses v002
   → Ingests data with v002 metadata
   
✅ Automatic, no manual steps
```

### Scenario 2: Need to Rollback

```
1. Edit mappings/current.txt manually:
   park_power_mapping_v001.csv

2. Run Cell 3
   → Uses v001 instead
   → Next ingestion uses old mapping
   
✅ One-line rollback, no code changes
```

### Scenario 3: Multiple Excel Exports with Different Schemas

```
Run 1: Excel has columns A, B, C
  → Cell 2 generates v001
  → Cell 3 ingests with v001

(Excel schema changes: adds column D)

Run 2: Excel has columns A, B, C, D
  → Cell 2 generates v002 (preserves A,B,C; prompts for D)
  → Cell 3 ingests with v002

(Need to re-process old data with v001)

Run 3: Change current.txt back to v001
  → Cell 3 ingests old files with v001
  → Registry shows mapping_filename=v001 for those rows

✅ Different schemas handled gracefully with full audit trail
```

## scada_column_sanitizer.py

No changes needed to scada_column_sanitizer.py itself. The module already:
- Supports `existing_mapping` parameter
- Handles preservation of known columns
- Can save mappings to any path

Cell 2 now:
- Loads mapping from versioned location
- Passes to sanitizer
- Saves result to versioned location

## Backward Compatibility

- Old `outputs/park_power_mapping.csv` is still created by Cell 1 (sanitization) but ignored by versioning system
- Cell 2 reads from `mappings/current.txt` (versioned)
- Cell 3 reads from `mappings/current.txt` (versioned)
- Both ignore `outputs/park_power_mapping.csv`

You can safely delete `outputs/park_power_mapping.csv` if desired—all active mappings are in `mappings/`.

## What Happens When You Run Cell 2

1. Checks if `mappings/` folder exists (creates if not)
2. Reads `mappings/current.txt` to get current active mapping
3. Loads that mapping using scada_column_sanitizer
4. Reads Excel columns from `DATA_XLSX_TOTAL_FULL`
5. Regenerates sanitized names (only new columns need interaction)
6. Calculates next version number: `max(existing) + 1`
7. Saves to `mappings/park_power_mapping_v00X.csv`
8. Updates `mappings/current.txt` to new version
9. Prints summary

## Testing

Run Cell 2 multiple times:
```
Run 1: current.txt → v001, Cell 2 creates v002
Run 2: current.txt → v002, Cell 2 creates v003
Run 3: current.txt → v003, Cell 2 creates v004
```

Check `mappings/` folder:
```
park_power_mapping_v001.csv (original)
park_power_mapping_v002.csv (from run 1)
park_power_mapping_v003.csv (from run 2)
park_power_mapping_v004.csv (from run 3)
current.txt → v004
```

## Summary

✅ Cell 2 now integrated with versioning system  
✅ Automatic version numbering  
✅ One-line rollback via current.txt  
✅ Full audit trail in registry  
✅ No code changes to Cell 3 needed  
✅ Backward compatible with existing workflows
