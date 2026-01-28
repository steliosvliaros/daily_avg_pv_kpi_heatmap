#!/usr/bin/env python3
"""Run bronze pipeline with corrected park normalization."""

import sys
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

from src.config import get_config
from src.bronze_workflow import run_bronze_pipeline

if __name__ == "__main__":
    cfg = get_config()
    print(f"Running bronze pipeline with data_root={cfg.DATA_DIR}")
    print(f"Bronze root: {cfg.BRONZE_ROOT}")
    
    # Run the pipeline
    result = run_bronze_pipeline(config=cfg)
    
    print(f"\nPipeline complete!")
    print(f"Result: {result}")
