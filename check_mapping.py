import pandas as pd
import re

# Load the mapping
mapping = pd.read_csv("mappings/park_power_mapping_v004.csv")

COL_RE = re.compile(r"^(?P<park>.+?)__(?P<cap>\d+(?:\.\d+)?kwp)__(?P<meas>.+?)(?:__(?P<unit>u_[a-z0-9_]+))?$")

# Check which columns match or don't match
sanitized_cols = mapping['sanitized'].tolist()

print("Checking all sanitized columns against regex...")
print(f"Total sanitized columns: {len(sanitized_cols)}\n")

matched = []
not_matched = []

for col in sanitized_cols:
    if col == 'datetime' or col.lower() == 'ts':
        continue  # Skip timestamp column
    
    if "__" in col:
        if COL_RE.match(col):
            matched.append(col)
        else:
            not_matched.append(col)

print(f"✅ Matched: {len(matched)}")
print(f"❌ Not matched: {len(not_matched)}\n")

if not_matched:
    print("Columns that DON'T match:")
    for col in not_matched[:10]:  # Show first 10
        print(f"  - {col}")
    if len(not_matched) > 10:
        print(f"  ... and {len(not_matched) - 10} more")
