import pandas as pd
from pathlib import Path

# Read a sample parquet file
bronze_path = Path("bronze/scada_1d_signal")
parquet_files = list(bronze_path.glob("**/*.parquet"))

if parquet_files:
    sample_file = parquet_files[0]
    print(f"Reading: {sample_file}\n")
    df = pd.read_parquet(sample_file)
    
    # Show sample rows
    print("Sample rows:")
    print(df[['park_id', 'park_capacity_kwp', 'signal_name', 'unit']].head(10))
    
    # Show unique signal names (first 10)
    print("\n\nUnique signal names (first 10):")
    for sig in df['signal_name'].unique()[:10]:
        print(f"  {sig}")
else:
    print("No parquet files found")
