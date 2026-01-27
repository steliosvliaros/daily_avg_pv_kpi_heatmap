"""
Silver data processing workflow

Consolidates the full silver pipeline:
1. Load new bronze data using watermark pattern
2. Clean and validate with quality flags
3. Stage valid data with retention policy
4. Run exploratory data analysis (optional)
5. Enrich with park metadata (optional)
6. Ingest to persistent silver layer with quality gates
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src import silver_prepair as sp
from src import silver_pre_ingestion_eda as spie


@dataclass
class SilverPipelineConfig:
    """Configuration for silver processing pipeline"""
    bronze_root: Path = None
    silver_root: Path = None
    silver_watermark_path: Path = None
    unit_benchmarks_path: Optional[Path] = None
    park_metadata_path: Optional[Path] = None
    
    # Quality gates
    max_invalid_pct: float = 20.0
    
    # Staging options
    stage_retention: str = "last_n"
    stage_retain_n: int = 3
    archive_invalid: bool = True
    
    # EDA options
    run_eda: bool = False
    eda_show_plots: bool = False
    eda_max_days: Optional[int] = None
    eda_max_parks: Optional[int] = 10
    eda_max_signals: Optional[int] = 5
    
    # Enrichment options
    enrich_with_metadata: bool = False
    
    # Ingestion options
    ingest_to_persistent: bool = True
    parquet_compression: str = "zstd"
    
    dataset_name: str = "scada_1d_signal"
    
    def __init__(self, workspace_config=None, **kwargs):
        """Initialize config from workspace_config or individual paths.
        
        Parameters
        ----------
        workspace_config : WorkspaceConfig, optional
            Workspace configuration object. If provided, paths are derived from it.
        **kwargs : dict
            Override any config attributes
        """
        if workspace_config is not None:
            self.bronze_root = workspace_config.BRONZE_ROOT
            self.silver_root = workspace_config.SILVER_ROOT
            self.silver_watermark_path = workspace_config.SILVER_OPS / "last_silver_committed.txt"
            self.unit_benchmarks_path = workspace_config.UNIT_BENCHMARKS_CSV
            self.park_metadata_path = workspace_config.PARK_METADATA_CSV
        
        # Apply any overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Validate required paths
        if self.bronze_root is None or self.silver_root is None:
            raise ValueError("bronze_root and silver_root must be provided either via workspace_config or directly")


@dataclass
class SilverPipelineResult:
    """Results from silver pipeline execution"""
    success: bool
    rows_loaded: int = 0
    rows_valid: int = 0
    rows_invalid: int = 0
    rows_ingested: int = 0
    run_ids_committed: List[str] = None
    stage_path: Optional[Path] = None
    eda_outputs: Optional[Dict] = None
    enriched_df: Optional[pd.DataFrame] = None
    ingest_result: Optional[Dict] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.run_ids_committed is None:
            self.run_ids_committed = []
        if self.errors is None:
            self.errors = []


def run_silver_pipeline(config: SilverPipelineConfig) -> SilverPipelineResult:
    """
    Execute the complete silver data processing pipeline.
    
    Steps:
    1. Load new bronze data using watermark pattern
    2. Clean and validate with quality flags
    3. Archive invalid rows (optional)
    4. Stage valid data with retention policy
    5. Commit watermark after successful staging
    6. Run EDA on staged data (optional)
    7. Enrich with park metadata (optional)
    8. Ingest to persistent silver layer (optional)
    
    Args:
        config: SilverPipelineConfig with all pipeline settings
        
    Returns:
        SilverPipelineResult with execution details and outputs
    """
    result = SilverPipelineResult(success=False)
    
    try:
        # Step 1: Load new bronze data
        print("="*80)
        print("SILVER PIPELINE: Loading new bronze data")
        print("="*80)
        
        df_new, loaded_run_ids = sp.load_new_bronze_parts_from_runlogs(
            bronze_root=config.bronze_root,
            silver_watermark_path=config.silver_watermark_path,
            dataset_name=config.dataset_name,
        )

        result.rows_loaded = len(df_new)

        if df_new.empty:
            print("No new bronze rows found.")
            result.success = True
            return result

        print(f"✓ Loaded {len(df_new):,} new bronze rows from {len(loaded_run_ids)} run(s)")

        # Filter to parks with status_effective == true
        meta = sp.load_park_metadata(config.park_metadata_path)
        if meta is None:
            raise ValueError("park_metadata not found; required for status_effective filtering")
        if "status_effective" not in meta.columns:
            raise ValueError("park_metadata must contain status_effective column")

        status_series = meta["status_effective"].astype("string").str.strip().str.lower()
        allowed_parks = set(meta.loc[status_series == "true", "park_id"].astype(str))
        print(f"[silver_pipeline] status_effective=true parks: {len(allowed_parks)}")

        if not allowed_parks:
            print("No parks with status_effective=true; skipping silver pipeline")
            result.success = True
            return result

        df_new["park_id"] = df_new["park_id"].astype("string").str.strip().str.lower()
        before_filter = len(df_new)
        df_new = df_new[df_new["park_id"].isin(allowed_parks)]
        after_filter = len(df_new)
        print(f"[silver_pipeline] rows after status_effective filter: {before_filter:,} -> {after_filter:,}")

        if df_new.empty:
            print("No rows remain after status_effective filter; skipping silver pipeline")
            result.success = True
            return result
        
        # Step 2: Clean and validate
        print("\n" + "="*80)
        print("SILVER PIPELINE: Cleaning and validating data")
        print("="*80)
        
        df_silver, prep_stats = sp.clean_bronze_for_silver(
            df_new,
            keep_invalid=True,
            unit_benchmarks_path=config.unit_benchmarks_path
        )
        
        print(prep_stats)
        print("\nFlag counts:")
        flag_counts = df_silver.filter(regex="^flag_").sum()
        print(flag_counts)
        
        # Split valid/invalid
        invalid_mask = df_silver.filter(regex="^flag_").any(axis=1)
        df_invalid = df_silver[invalid_mask].copy()
        df_valid = df_silver[~invalid_mask].copy()
        
        result.rows_valid = len(df_valid)
        result.rows_invalid = len(df_invalid)
        
        print(f"\n✓ Valid rows: {len(df_valid):,} | Invalid rows: {len(df_invalid):,}")
        
        # Step 3: Archive invalid rows
        silver_stage_dir = config.silver_root / "_stage"
        
        if config.archive_invalid and not df_invalid.empty:
            print("\n" + "="*80)
            print("SILVER PIPELINE: Archiving invalid rows")
            print("="*80)
            
            invalid_dir = silver_stage_dir / "invalid"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            run_id_for_invalid = loaded_run_ids[-1] if loaded_run_ids else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            invalid_path = invalid_dir / f"invalid_{run_id_for_invalid}.parquet"
            df_invalid.to_parquet(invalid_path, index=False)
            print(f"✓ Archived {len(df_invalid):,} invalid rows -> {invalid_path.name}")
        
        if df_valid.empty:
            print("\n⚠ No valid rows to stage; skipping remaining pipeline steps.")
            result.success = True
            return result
        
        # Step 4: Stage valid data
        print("\n" + "="*80)
        print("SILVER PIPELINE: Staging valid data")
        print("="*80)
        
        stage_path = sp.write_silver_stage(
            df_valid,
            silver_stage_dir,
            retention=config.stage_retention,
            retain_n=config.stage_retain_n,
        )
        result.stage_path = stage_path
        print(f"✓ Staged {len(df_valid):,} rows -> {stage_path.name}")
        
        # Step 5: Commit watermark
        sp.commit_silver_watermark(config.silver_watermark_path, loaded_run_ids)
        result.run_ids_committed = loaded_run_ids
        print(f"✓ Committed {len(loaded_run_ids)} run(s) to silver watermark")
        
        # Step 6: Run EDA (optional)
        if config.run_eda:
            print("\n" + "="*80)
            print("SILVER PIPELINE: Running exploratory data analysis")
            print("="*80)
            
            eda_cfg = spie.EdaConfig(
                max_days=config.eda_max_days,
                max_parks=config.eda_max_parks,
                max_signals=config.eda_max_signals,
                focus_signal=None,
                focus_signals=None,
            )
            
            eda_outputs = spie.run_silver_eda(df_valid, config=eda_cfg)
            result.eda_outputs = eda_outputs
            
            print(f"✓ Generated {len(eda_outputs['plots'])} EDA plots")
            
            if config.eda_show_plots:
                from IPython.display import display
                import matplotlib.pyplot as plt
                
                print("\nSignal statistics:")
                display(eda_outputs["signal_stats"].head())
                print("\nCoverage summary:")
                display(eda_outputs["coverage"].head())
                
                for fig in eda_outputs["plots"]:
                    display(fig)
                    plt.close(fig)
        
        # Step 7: Enrich with metadata (optional)
        if config.enrich_with_metadata and config.park_metadata_path:
            print("\n" + "="*80)
            print("SILVER PIPELINE: Enriching with park metadata")
            print("="*80)
            
            park_meta = sp.load_park_metadata(config.park_metadata_path)
            
            if park_meta is not None:
                enriched = df_valid.merge(
                    park_meta,
                    on="park_id",
                    how="left",
                    validate="m:1",
                )
                result.enriched_df = enriched
                print(f"✓ Enriched with metadata: {enriched.shape}")
                print(f"  Metadata columns: {list(park_meta.columns)}")
            else:
                print(f"⚠ Park metadata not found at {config.park_metadata_path}")
        
        # Step 8: Ingest to persistent layer (optional)
        if config.ingest_to_persistent:
            print("\n" + "="*80)
            print("SILVER PIPELINE: Ingesting to persistent silver layer")
            print("="*80)
            
            ingest_result = sp.ingest_silver_stage(
                stage_path,
                config.silver_root,
                max_invalid_pct=config.max_invalid_pct,
                compression=config.parquet_compression,
            )
            
            result.ingest_result = ingest_result
            
            if ingest_result["success"]:
                result.rows_ingested = ingest_result["rows_ingested"]
                print(f"✓ Ingested {result.rows_ingested:,} rows to persistent silver layer")
                print(f"  Partitioned by year/month")
                print(f"  Quality gate: {ingest_result['quality_gate_passed']}")
            else:
                errors = ingest_result.get("errors", [])
                result.errors.extend(errors)
                print(f"✗ Ingestion failed:")
                for err in errors:
                    print(f"  {err}")
                return result
        
        result.success = True
        print("\n" + "="*80)
        print("SILVER PIPELINE: Complete")
        print("="*80)
        print(f"Summary:")
        print(f"  Loaded: {result.rows_loaded:,} rows")
        print(f"  Valid: {result.rows_valid:,} rows")
        print(f"  Invalid: {result.rows_invalid:,} rows")
        if config.ingest_to_persistent:
            print(f"  Ingested: {result.rows_ingested:,} rows")
        print(f"  Run IDs committed: {len(result.run_ids_committed)}")
        
        return result
        
    except Exception as e:
        result.errors.append(str(e))
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return result


def get_latest_staged_file(silver_root: Path) -> Optional[Path]:
    """
    Get the most recently modified staged parquet file.
    
    Args:
        silver_root: Root silver directory
        
    Returns:
        Path to latest staged file or None if none found
    """
    stage_dir = silver_root / "_stage"
    if not stage_dir.exists():
        return None
    
    candidates = sorted(
        stage_dir.glob("silver_stage_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    return candidates[0] if candidates else None
