import sys
import importlib
from pathlib import Path

# Setup paths
workspace_root = Path.cwd()
sys.path.insert(0, str(workspace_root))

from src import bronze_workflow as bw
from src.config import setup_workspace

importlib.reload(bw)

# Setup config
config = setup_workspace(workspace_root=workspace_root, verbose=False)

# Configure bronze pipeline with reset
bronze_config = bw.get_bronze_pipeline_config_from_workspace_config(
    workspace_config=config,
    sanitize_files=True,
    run_ingestion=True,
    run_inspection=True,
    run_sanity_checks=True,
)

# Reset before re-ingesting
bronze_config.reset_before_ingestion = True
bronze_config.reset_and_remove_logs = False

result = bw.run_bronze_pipeline(bronze_config)

if result.success:
    print("\n[OK] Bronze pipeline complete")
    # Check parks in bronze
    import pandas as pd
    df = pd.read_parquet(config.BRONZE_ROOT / 'scada_1d_signal' / 'year=2015' / 'month=01')
    parks = df['park_id'].value_counts()
    print(f"\nUnique parks in bronze: {parks.index.nunique()}")
    print("\nPark row counts:")
    print(parks)
else:
    print("\n[FAIL] Bronze pipeline failed:")
    for err in result.errors:
        print(f"  {err}")
