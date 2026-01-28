#!/usr/bin/env python
"""Debug park_id matching to understand why join is failing."""

import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import WorkspaceConfig

# Load park_metadata directly
metadata_path = Path(__file__).parent / "mappings" / "park_metadata.csv"
workspace_root = Path(__file__).parent
park_metadata = pd.read_csv(metadata_path)

print("=" * 80)
print("PARK METADATA")
print("=" * 80)
print(f"Total parks in metadata: {len(park_metadata)}")
print(f"Columns: {list(park_metadata.columns)}")
print("\nActive parks (status_effective=true):")
active_parks = park_metadata[park_metadata['status_effective'] == True]
print(f"Count: {len(active_parks)}")
print(active_parks[['park_id', 'park_name']].to_string())

print("\n" + "=" * 80)
print("NORMALIZATION TEST")
print("=" * 80)

def normalize_park_name_for_matching(name):
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    # Remove p_ prefix if present (from sanitized columns)
    s = re.sub(r'^p_', '', s)
    # Remove capacity patterns: "176 kwp" or "176kwp" but preserve word boundaries
    s = re.sub(r'\s+\d+\s*(kwp|mw)\b', '', s)  # "176 KWp" with space
    s = re.sub(r'\d+\s*(kwp|mw)\b', '', s)     # "176KWp" without space  
    s = re.sub(r'\s*\d+\s*mw\b', '', s)        # "0.45MW" pattern
    # Normalize whitespace and underscores to single underscore
    s = re.sub(r'[\s\.\-]+', '_', s)
    s = re.sub(r'_+', '_', s)
    # Remove other punctuation but preserve underscores
    s = re.sub(r'[,;:\(\)]', '', s)
    return s.strip('_')

# Test normalization of metadata park_names
print("\nMetadata park_names normalized:")
for i, row in active_parks.iterrows():
    original = row['park_name']
    normalized = normalize_park_name_for_matching(original)
    print(f"  '{original:50}' -> '{normalized}'")

# Now load the Excel file and see what columns are there
print("\n" + "=" * 80)
print("EXCEL FILE COLUMNS")
print("=" * 80)

excel_path = workspace_root / "data" / "inbox" / "Exported Data-20150101T000000__hash=9baf4873632c.xlsx"
if not excel_path.exists():
    print(f"Excel file not found: {excel_path}")
    # Try to find it
    inbox_dir = workspace_root / "data" / "inbox"
    files = list(inbox_dir.glob("*.xlsx"))
    if files:
        excel_path = files[0]
        print(f"Using: {excel_path}")
    else:
        print("No Excel files in inbox")
        sys.exit(1)

df = pd.read_excel(excel_path)
print(f"Excel shape: {df.shape}")
print(f"\nColumns ({len(df.columns)} total):")
print(list(df.columns)[:15])  # Print first 15 columns

# Look for park-related columns
park_cols = [c for c in df.columns if '__' in c]
print(f"\nMeasurement columns (with __): {len(park_cols)}")

# Extract park_id from first few columns
print("\nFirst 15 sanitized measurement columns:")
COL_RE = re.compile(
    r"^p_(?P<park_id>[a-z0-9_]+?)__"
    r"(?P<capacity>[a-z0-9_]+?)__"
    r"(?P<meas>[a-z0-9_]+?)__"
    r"(?P<unit>[a-z0-9_]+)$"
)

for col in park_cols[:15]:
    match = COL_RE.match(col)
    if match:
        park_id = match.group('park_id')
        capacity = match.group('capacity')
        meas = match.group('meas')
        print(f"  {col}")
        print(f"    -> park_id='{park_id}', capacity='{capacity}', meas='{meas}'")
    else:
        print(f"  {col} - NO MATCH!")

print(f"\nTotal unique extracted park_ids: {len(set(c.split('__')[0] for c in park_cols if '__' in c))}")
all_extracted = set()
for col in park_cols:
    match = COL_RE.match(col)
    if match:
        park_id = match.group('park_id')
        all_extracted.add(park_id)

print(f"All extracted park_ids:\n{sorted(all_extracted)}")

# Now test normalization of extracted park_ids
print("\n" + "=" * 80)
print("MATCHING TEST")
print("=" * 80)

# Create lookup from metadata
meta_lookup = park_metadata[['park_id', 'park_name']].copy()
meta_lookup['lookup_name'] = meta_lookup['park_name'].apply(normalize_park_name_for_matching)
meta_lookup = meta_lookup[['lookup_name', 'park_id']].drop_duplicates()

print(f"Metadata lookup has {len(meta_lookup)} entries:")
print(meta_lookup.to_string())

# Now check which extracted park names match
print("\n" + "=" * 80)
print("JOIN RESULTS")
print("=" * 80)

print(f"\nChecking {len(all_extracted)} extracted park_ids against {len(meta_lookup)} metadata entries:")
matches = 0
no_matches = 0
for extracted_name in sorted(all_extracted):
    # The extracted_name is already in the "p_xxx" format, need to normalize it
    normalized = normalize_park_name_for_matching(extracted_name)
    
    # Look for this in metadata lookup
    matching_meta = meta_lookup[meta_lookup['lookup_name'] == normalized]
    
    if len(matching_meta) > 0:
        park_id = matching_meta.iloc[0]['park_id']
        print(f"✓ '{extracted_name}' -> normalized='{normalized}' -> park_id='{park_id}'")
        matches += 1
    else:
        print(f"✗ '{extracted_name}' -> normalized='{normalized}' -> NO MATCH")
        # Show similar entries in metadata
        similar = meta_lookup[meta_lookup['lookup_name'].str.contains(normalized[:5], na=False, case=False)]
        if len(similar) > 0:
            print(f"   Similar in metadata: {similar['lookup_name'].tolist()}")
        no_matches += 1

print(f"\nSummary: {matches} matches, {no_matches} no matches")
