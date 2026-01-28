import pandas as pd
from pathlib import Path

# Read one parquet file to see structure
bronze_path = Path('bronze/scada_1d_signal')
parquet_files = list(bronze_path.rglob('*.parquet'))
print(f'Total parquet files: {len(parquet_files)}')

if parquet_files:
    # Read first file
    df = pd.read_parquet(parquet_files[0])
    print(f'\nShape of first file: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'\nFirst few rows:')
    print(df.head())
    
    # Count total rows across ALL files
    total_rows = 0
    all_parks = set()
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        total_rows += len(df)
        all_parks.update(df['park_id'].unique())
    
    print(f'\nTotal rows across ALL files: {total_rows:,}')
    print(f'Total unique parks: {len(all_parks)}')
    print(f'Parks: {sorted(all_parks)}')
