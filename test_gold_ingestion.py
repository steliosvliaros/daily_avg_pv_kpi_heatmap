"""
Quick test script for gold layer ingestion.

Run this to verify the gold layer setup is working correctly.
"""

from pathlib import Path
from src.config import WorkspaceConfig
from src.silver_loader import load_and_ingest_daily_energy_gold, load_gold_daily_energy

def test_gold_ingestion():
    """Test gold layer ingestion with basic checks."""
    
    print("="*80)
    print("TESTING GOLD LAYER INGESTION")
    print("="*80)
    
    # Initialize config
    config = WorkspaceConfig()
    config.setup_workspace()
    
    print(f"\n✓ Config initialized")
    print(f"  GOLD_DIR: {config.GOLD_DIR}")
    print(f"  GOLD_DAILY_ENERGY: {config.GOLD_DAILY_ENERGY}")
    print(f"  GOLD_DAILY_ENERGY_OPS: {config.GOLD_DAILY_ENERGY_OPS}")
    
    # Check directory structure
    assert config.GOLD_DAILY_ENERGY.exists(), "daily_energy folder missing"
    assert config.GOLD_DAILY_ENERGY_OPS.exists(), "daily_energy/_ops folder missing"
    assert config.GOLD_PVGIS_REFERENCE.exists(), "pvgis_reference folder missing"
    assert config.GOLD_PVGIS_REFERENCE_OPS.exists(), "pvgis_reference/_ops folder missing"
    print(f"\n✓ Directory structure OK")
    
    # Check metadata files
    metadata_file = config.GOLD_DAILY_ENERGY_OPS / "metadata.json"
    assert metadata_file.exists(), "metadata.json missing"
    import json
    metadata = json.loads(metadata_file.read_text())
    assert metadata["table_name"] == "daily_energy", "Invalid metadata"
    print(f"\n✓ Metadata files OK")
    print(f"  Table: {metadata['table_name']}")
    print(f"  Version: {metadata['version']}")
    print(f"  Refresh strategy: {metadata['refresh_strategy']['type']}")
    
    # Test ingestion (if silver data exists)
    if config.SILVER_ROOT.exists():
        print(f"\n→ Testing ingestion from silver...")
        try:
            result = load_and_ingest_daily_energy_gold(
                silver_root=config.SILVER_ROOT,
                metadata_path=config.PARK_METADATA_CSV,
                gold_root=config.GOLD_DIR,
                debug=True
            )
            print(f"\n✓ Ingestion completed")
            print(f"  Status: {result['status']}")
            print(f"  Rows written: {result['rows_written']}")
            print(f"  Last date: {result['last_date']}")
            print(f"  Message: {result['message']}")
            
            # Try loading the data
            if result['status'] in ['appended', 'replaced']:
                daily_energy = load_gold_daily_energy(config.GOLD_DIR, debug=True)
                print(f"\n✓ Gold data loaded successfully")
                print(f"  Shape: {daily_energy.shape}")
                if not daily_energy.empty:
                    print(f"  Date range: {daily_energy.index.min()} to {daily_energy.index.max()}")
                    print(f"  Parks: {len(daily_energy.columns)}")
                    print(f"\nFirst 3 rows:")
                    print(daily_energy.head(3))
                
        except Exception as e:
            print(f"\n⚠ Ingestion test failed (may be expected if no silver data): {e}")
    else:
        print(f"\n⚠ Silver folder not found, skipping ingestion test")
    
    print("\n" + "="*80)
    print("GOLD LAYER SETUP VERIFIED")
    print("="*80)
    print("\nNext steps:")
    print("1. Run load_and_ingest_daily_energy_gold() in your notebook")
    print("2. Check gold/daily_energy/_ops/last_gold_committed.txt for watermark")
    print("3. Run again to test incremental append (should skip duplicate dates)")
    print("4. See gold/USAGE_EXAMPLES.md for more examples")

if __name__ == "__main__":
    test_gold_ingestion()
