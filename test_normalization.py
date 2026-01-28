import re
import pandas as pd

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

# Test with sanitized column names (what we extract from Excel)
print("Test with sanitized column park_ids (from Excel):")
sanitized_parks = [
    "p_4e_energeiaki_likovouni",
    "p_4e_energeiaki_lexaina",
    "p_fragiatoula_utilitas",
    "p_hliatoras_andravida",
    "p_ntarali_bonitas",
    "p_palaionaziro_iraio",
]
for park in sanitized_parks:
    normalized = normalize_park_name_for_matching(park)
    print(f"  '{park}'")
    print(f"    -> '{normalized}'")

# Test with metadata park_names
print("\n\nTest with metadata park_names:")
meta = pd.read_csv('mappings/park_metadata.csv')
active = meta[meta['status_effective'].astype(str).str.strip().str.lower() == 'true']
for idx, row in active.iterrows():
    normalized = normalize_park_name_for_matching(row['park_name'])
    print(f"  '{row['park_name']}'")
    print(f"    -> '{normalized}'")
    if idx >= 8:
        break

