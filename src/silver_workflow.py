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

import gc
import shutil
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
    
    # Memory optimization
    filter_chunk_size: int = 5_000_000  # Chunk size for filtering large DataFrames to avoid memory spikes
    
    # Reset options (full reload scenarios)
    reset_before_processing: bool = False  # Set to True to clear silver and reset watermark (full reload)
    reset_keep_backups: bool = True        # If True, archive deleted data before reset
    
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
        # Step 0: Optional reset for full reload
        if config.reset_before_processing:
            print("="*80)
            print("SILVER PIPELINE: Resetting for full reload")
            print("="*80)
            
            # Backup silver data if requested
            if config.reset_keep_backups and config.silver_root.exists():
                backup_dir = config.silver_root.parent / f"silver_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"Backing up silver data to: {backup_dir}")
                if config.silver_root.exists():
                    shutil.copytree(config.silver_root, backup_dir)
            
            # Reset silver persistent data (but keep _ops for logs)
            silver_data_dir = config.silver_root / "year=*"
            import glob
            for year_dir in glob.glob(str(silver_data_dir)):
                shutil.rmtree(year_dir)
                print(f"Deleted: {year_dir}")
            
            # Reset watermark to process all bronze data
            config.silver_watermark_path.parent.mkdir(parents=True, exist_ok=True)
            config.silver_watermark_path.write_text("")  # Empty watermark = reprocess everything
            print("Watermark reset - will reprocess all bronze data on next run")
        
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
            if config.ingest_to_persistent:
                stage_path = get_latest_staged_file(config.silver_root)
                if stage_path is not None:
                    print("\n" + "="*80)
                    print("SILVER PIPELINE: Resuming ingestion from last stage")
                    print("="*80)
                    print(f"Stage file: {stage_path}")
                    last_ingested = sp.read_last_ingested_stage(config.silver_root)
                    if last_ingested and stage_path.name == last_ingested:
                        print("Stage file already ingested; skipping re-ingest.")
                        result.stage_path = stage_path
                        result.rows_ingested = 0
                        result.success = True
                        return result
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
                        result.success = True
                        return result
                    errors = ingest_result.get("errors", [])
                    result.errors.extend(errors)
                    print("✗ Ingestion failed:")
                    for err in errors:
                        print(f"  {err}")
                    return result
                print("No staged file found; skipping ingestion.")
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
        
        # Filter in chunks to avoid memory allocation for entire boolean mask
        # Split into ~500K row chunks and filter each separately
        chunk_filter_size = 500_000
        filtered_chunks = []
        
        for i in range(0, len(df_new), chunk_filter_size):
            chunk = df_new.iloc[i : i + chunk_filter_size]
            mask = chunk["park_id"].isin(allowed_parks)
            filtered_chunk = chunk[mask].copy()
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
            del chunk, mask, filtered_chunk
            gc.collect()
        
        if filtered_chunks:
            df_new = pd.concat(filtered_chunks, ignore_index=True)
            del filtered_chunks
            gc.collect()
        else:
            df_new = pd.DataFrame()
        
        after_filter = len(df_new)
        print(f"[silver_pipeline] rows after status_effective filter: {before_filter:,} -> {after_filter:,}")

        if df_new.empty:
            print("No rows remain after status_effective filter; skipping silver pipeline")
            result.success = True
            return result
        
        # Step 2: Clean, validate and partition in chunks to manage memory
        print("\n" + "="*80)
        print("SILVER PIPELINE: Cleaning and validating data (chunked)")
        print("="*80)
        
        # Use temporary parquet files to avoid memory accumulation
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tempfile import NamedTemporaryFile
        
        temp_valid_file = NamedTemporaryFile(delete=False, suffix=".parquet")
        temp_invalid_file = NamedTemporaryFile(delete=False, suffix=".parquet")
        temp_valid_path = Path(temp_valid_file.name)
        temp_invalid_path = Path(temp_invalid_file.name)
        temp_valid_file.close()
        temp_invalid_file.close()
        
        valid_writer = None
        invalid_writer = None
        total_valid = 0
        total_invalid = 0
        chunk_size = config.filter_chunk_size
        
        try:
            for i in range(0, len(df_new), chunk_size):
                chunk = None
                df_chunk_clean = None
                invalid_mask = None
                
                chunk = df_new.iloc[i : i + chunk_size].copy()
                chunk_num = i // chunk_size + 1
                progress = min(i + chunk_size, len(df_new))
                print(f"\n  Chunk {chunk_num}: rows {i:,} to {progress:,}")
                
                try:
                    # Clean this chunk
                    df_chunk_clean, prep_stats = sp.clean_bronze_for_silver(
                        chunk,
                        keep_invalid=True,
                        unit_benchmarks_path=config.unit_benchmarks_path,
                        inplace=True
                    )
                    
                    # Split valid/invalid
                    invalid_mask = df_chunk_clean.filter(regex="^flag_").any(axis=1)
                    df_chunk_valid = df_chunk_clean.loc[~invalid_mask]
                    df_chunk_invalid = df_chunk_clean.loc[invalid_mask]
                    
                    chunk_valid = len(df_chunk_valid)
                    chunk_invalid = len(df_chunk_invalid)
                    total_valid += chunk_valid
                    total_invalid += chunk_invalid
                    
                    print(f"    Valid: {chunk_valid:,} | Invalid: {chunk_invalid:,}")
                    
                    # Write chunks directly to parquet files using PyArrow
                    if chunk_valid > 0:
                        table_valid = pa.Table.from_pandas(df_chunk_valid, preserve_index=False)
                        if valid_writer is None:
                            valid_writer = pq.ParquetWriter(temp_valid_path, table_valid.schema)
                        valid_writer.write_table(table_valid)
                        del table_valid
                    
                    if chunk_invalid > 0:
                        table_invalid = pa.Table.from_pandas(df_chunk_invalid, preserve_index=False)
                        if invalid_writer is None:
                            invalid_writer = pq.ParquetWriter(temp_invalid_path, table_invalid.schema)
                        invalid_writer.write_table(table_invalid)
                        del table_invalid
                    
                    # Free chunk data immediately
                    del df_chunk_valid, df_chunk_invalid
                    
                except MemoryError as e:
                    print(f"    ⚠️ Memory error on chunk {chunk_num}: {e}")
                    print(f"    Attempting recovery with smaller batch...")
                    gc.collect()
                    continue
                finally:
                    # Always free temporary references (only if they exist)
                    if chunk is not None:
                        del chunk
                    if df_chunk_clean is not None:
                        del df_chunk_clean
                    if invalid_mask is not None:
                        del invalid_mask
                    gc.collect()  # Force collection after each chunk
            
            # Close parquet writers
            if valid_writer is not None:
                valid_writer.close()
            if invalid_writer is not None:
                invalid_writer.close()
            
            # Instead of reading back, move temp files directly to final locations
            # This avoids loading 38M+ rows into memory
            
        finally:
            pass  # Don't cleanup yet - we need the temp files
        
        gc.collect()  # Force garbage collection after combining chunks
        
        result.rows_valid = total_valid
        result.rows_invalid = total_invalid
        
        print(f"\n✓ Total valid rows: {total_valid:,} | Total invalid rows: {total_invalid:,}")
        
        # Step 3: Archive invalid rows (move temp file directly)
        silver_stage_dir = config.silver_root / "_stage"
        
        if config.archive_invalid and total_invalid > 0 and temp_invalid_path.exists():
            print("\n" + "="*80)
            print("SILVER PIPELINE: Archiving invalid rows")
            print("="*80)
            
            invalid_dir = silver_stage_dir / "invalid"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            run_id_for_invalid = loaded_run_ids[-1] if loaded_run_ids else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            invalid_path = invalid_dir / f"invalid_{run_id_for_invalid}.parquet"
            
            # Move temp file directly (avoid reading into memory)
            shutil.move(str(temp_invalid_path), str(invalid_path))
            print(f"✓ Archived {total_invalid:,} invalid rows -> {invalid_path.name}")
        else:
            # Cleanup temp invalid file if not archiving
            try:
                if temp_invalid_path.exists():
                    temp_invalid_path.unlink()
            except Exception:
                pass
        
        if total_valid == 0:
            print("\n⚠ No valid rows to stage; skipping remaining pipeline steps.")
            result.success = True
            # Cleanup temp valid file
            try:
                if temp_valid_path.exists():
                    temp_valid_path.unlink()
            except Exception:
                pass
            return result
        
        # Step 4: Stage valid data (move temp file directly)
        print("\n" + "="*80)
        print("SILVER PIPELINE: Staging valid data")
        print("="*80)
        
        # Generate stage filename using same logic as write_silver_stage
        run_id = loaded_run_ids[-1] if loaded_run_ids else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stage_filename = f"silver_stage_{run_id}.parquet"
        stage_path = silver_stage_dir / stage_filename
        silver_stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Move temp file directly (avoid reading into memory)
        shutil.move(str(temp_valid_path), str(stage_path))
        
        # Apply retention policy manually (since we're not using write_silver_stage)
        retention_mode = (config.stage_retention or "keep").strip().lower()
        if retention_mode == "last_n":
            stage_files = sorted(
                silver_stage_dir.glob("silver_stage_*.parquet"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            retain_n = max(1, int(config.stage_retain_n))
            for old_file in stage_files[retain_n:]:
                try:
                    old_file.unlink(missing_ok=True)
                except Exception:
                    pass
        
        result.stage_path = stage_path
        print(f"✓ Staged {total_valid:,} rows -> {stage_path.name}")
        
        # Step 5: Commit watermark
        sp.commit_silver_watermark(config.silver_watermark_path, loaded_run_ids)
        result.run_ids_committed = loaded_run_ids
        print(f"✓ Committed {len(loaded_run_ids)} run(s) to silver watermark")
        
        # Step 6: Run EDA (optional) - SKIP for large datasets
        if config.run_eda:
            # For datasets > 10M rows, skip EDA to avoid memory issues
            if total_valid > 10_000_000:
                print("\n" + "="*80)
                print("SILVER PIPELINE: Skipping EDA (dataset too large)")
                print("="*80)
                print(f"⚠ EDA skipped for {total_valid:,} rows (> 10M threshold)")
                print(f"  To run EDA, load data manually from: {stage_path}")
            else:
                print("\n" + "="*80)
                print("SILVER PIPELINE: Running exploratory data analysis")
                print("="*80)
                
                # Load data for EDA
                df_valid = pd.read_parquet(stage_path)
                
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
                
                # Free memory
                del df_valid
                gc.collect()
        
        # Step 7: Enrich with metadata (optional) - SKIP for large datasets
        if config.enrich_with_metadata and config.park_metadata_path:
            if total_valid > 10_000_000:
                print("\n" + "="*80)
                print("SILVER PIPELINE: Skipping enrichment (dataset too large)")
                print("="*80)
                print(f"⚠ Enrichment skipped for {total_valid:,} rows (> 10M threshold)")
            else:
                print("\n" + "="*80)
                print("SILVER PIPELINE: Enriching with park metadata")
                print("="*80)
                
                # Load data for enrichment
                df_valid = pd.read_parquet(stage_path)
                
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
                
                # Free memory
                del df_valid
                gc.collect()
        
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
