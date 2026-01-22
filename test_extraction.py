import pandas as pd
import re

COL_RE = re.compile(r"^(?P<park>.+?)__(?P<cap>\d+(?:\.\d+)?kwp)__(?P<meas>.+?)(?:__(?P<unit>u_[a-z0-9_]+))?$")

# Simulate the melted dataframe
test_cols = [
    "p_4e_ener_liko__176kwp__pcc_curr_thd_r_of_neut__u_pct",
    "p_4e_ener_lexa__4472kwp__pcc_acti_ener_expo__u_kwh",
    "frag_util__4866kwp__pcc_acti_ener_expo__u_kwh",
]

df = pd.DataFrame({"col": test_cols, "value": [1.5, 2.3, 3.4]})

print("Original columns:")
print(df)

# Extract
extracted = df["col"].str.extract(COL_RE)
print("\n\nExtracted groups:")
print(extracted)

# Apply transformations
df["park_id"] = extracted["park"]
df["signal_name"] = extracted["meas"]
df["unit"] = extracted["unit"]

# Capacity
cap_str = extracted["cap"].str.replace(r"kwp$", "", regex=True)
df["park_capacity_kwp"] = pd.to_numeric(cap_str, errors="coerce")

print("\n\nFinal dataframe:")
print(df[['park_id', 'park_capacity_kwp', 'signal_name', 'unit']])
