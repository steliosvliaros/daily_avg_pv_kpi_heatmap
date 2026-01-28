import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')
from src.config import setup_workspace

config = setup_workspace(Path.cwd(), False)
df = pd.read_parquet(str(config.BRONZE_ROOT / 'scada_1d_signal' / 'year=2015' / 'month=01'))

print(f'Unique parks: {df["park_id"].nunique()}')
print('\nParks in bronze:')
for p in sorted(df['park_id'].unique()):
    count = (df['park_id'] == p).sum()
    print(f'  {p}: {count} rows')
