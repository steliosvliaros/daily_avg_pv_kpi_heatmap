"""
Bronze data processing workflow

Consolidates the complete bronze ingestion pipeline:
1. File sanitization (restore rejected files, clean columns)
2. Configure bronze ingestion (build Config from workspace)
3. Execute ingestion (process inbox files into bronze layer)
4. Inspect bronze data (sample and summarize ingested data)
5. Run sanity checks (validate units, identify anomalies)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src import bronze_ingest as bi
from src import bronze_inspection as bi_inspect
from src import file_sanitizer as fs
from src import unit_sanity_check as usc


@dataclass
class BronzePipelineConfig:
    """Configuration for bronze processing pipeline"""
    workspace_root: Path
    data_root: Path
    bronze_root: Path
    mappings_root: Path
    outputs_root: Path
    
    # Sanitization options
    sanitize_files: bool = True
    restore_rejected: bool = True
    dry_run_sanitize: bool = False
    
    # Ingestion options
    run_ingestion: bool = True
    timezone_local: str = "Europe/Athens"
    min_age_seconds: int = 0  # For notebook: ingest immediately
    stable_check_seconds: int = 0
    allow_duplicates: bool = False
    parquet_compression: str = "zstd"
    
    # Inspection options
    run_inspection: bool = True
    inspection_year: Optional[int] = 2026
    inspection_n_files: int = 10
    inspection_partition_limit: int = 10
    inspection_show_partitions: bool = True
    inspection_show_schema: bool = True
    inspection_show_missing: bool = True
    inspection_show_summary: bool = True
    
    # Sanity check options
    run_sanity_checks: bool = True


@dataclass
class BronzePipelineResult:
    """Results from bronze pipeline execution"""
    success: bool = False
    
    # Sanitization results
    sanitization_success: bool = False
    files_restored: int = 0
    files_sanitized: int = 0
    sanitization_errors: List[tuple] = field(default_factory=list)
    
    # Ingestion results
    ingestion_success: bool = False
    run_logs: List[str] = field(default_factory=list)
    files_ingested: int = 0
    
    # Inspection results
    inspection_data: Optional[Dict] = None
    inspection_rows_sampled: int = 0
    
    # Sanity check results
    sanity_check_success: bool = False
    benchmark_df: Optional[pd.DataFrame] = None
    benchmark_csv_path: Optional[Path] = None
    
    # Overall
    errors: List[str] = field(default_factory=list)


def run_bronze_pipeline(config: BronzePipelineConfig) -> BronzePipelineResult:
    """
    Execute the complete bronze data processing pipeline.
    
    Steps:
    1. File sanitization (restore rejected, clean columns)
    2. Build ingestion config from workspace config
    3. Reset dataset (optional) and execute ingestion
    4. Inspect bronze data (sample and summarize)
    5. Run unit sanity checks (validate values)
    
    Args:
        config: BronzePipelineConfig with all pipeline settings
        
    Returns:
        BronzePipelineResult with execution details and outputs
    """
    result = BronzePipelineResult(success=False)
    
    try:
        # Step 1: File Sanitization
        if config.sanitize_files:
            print("="*80)
            print("BRONZE PIPELINE: File Sanitization")
            print("="*80)
            
            # Restore rejected files
            if config.restore_rejected:
                restored = fs.restore_rejected_files(config.data_root, verbose=True)
                result.files_restored = restored
                print(f"✓ Restored {restored} files from rejected")
            
            # Read current mapping
            current_mapping_path = config.mappings_root / "current.txt"
            if not current_mapping_path.exists():
                result.errors.append(f"Current mapping file not found: {current_mapping_path}")
                return result
            
            current_mapping = current_mapping_path.read_text().strip()
            mapping_path = config.mappings_root / current_mapping
            
            if not mapping_path.exists():
                result.errors.append(f"Mapping file not found: {mapping_path}")
                return result
            
            print(f"\nUsing mapping: {current_mapping}")
            
            # Sanitize files
            sanitize_results = fs.sanitize_inbox_files(
                inbox_dir=config.data_root / "inbox",
                output_dir=config.data_root / "inbox",  # In-place
                mapping_path=mapping_path,
                backup_dir=config.data_root / "backups",
                dry_run=config.dry_run_sanitize,
                verbose=True
            )
            
            result.files_sanitized = sanitize_results.success_count
            result.sanitization_errors = sanitize_results.failed
            result.sanitization_success = True
            
            print(f"\n✓ Sanitized {sanitize_results.success_count} files")
            if sanitize_results.failed_count > 0:
                print(f"⚠ {sanitize_results.failed_count} files failed sanitization")
                result.errors.extend([f"Sanitization failed: {f}" for f, e in sanitize_results.failed])
        
        # Step 2: Configure Bronze Ingestion
        print("\n" + "="*80)
        print("BRONZE PIPELINE: Building Ingestion Config")
        print("="*80)
        
        # Build config from workspace settings
        ingest_cfg = bi.Config(
            data_root=config.data_root,
            inbox=config.data_root / "inbox",
            processing=config.data_root / "processing",
            archived=config.data_root / "archived",
            rejected=config.data_root / "rejected",
            bronze_root=config.bronze_root,
            mappings_root=config.mappings_root,
            dataset_name="scada_1d_signal",
            timezone_local=config.timezone_local,
            daily_interval_end_is_midnight=True,
            parquet_compression=config.parquet_compression,
            min_age_seconds=config.min_age_seconds,
            stable_check_seconds=config.stable_check_seconds,
            allow_duplicates=config.allow_duplicates,
        )
        
        ingest_cfg.lockfile = config.data_root / "_locks" / "bronze_ingest.lock"
        
        print(f"✓ Config built for bronze ingestion")
        print(f"  Inbox: {ingest_cfg.inbox}")
        print(f"  Bronze: {ingest_cfg.bronze_root}")
        
        # Step 3: Execute Bronze Ingestion
        if config.run_ingestion:
            print("\n" + "="*80)
            print("BRONZE PIPELINE: Executing Ingestion")
            print("="*80)
            
            # Optional: reset dataset first
            print("Resetting dataset (keeping run logs)...")
            bi.reset_dataset(ingest_cfg, remove_run_logs=False)
            
            print("Ingesting files from inbox...")
            bi.ingest_folder(ingest_cfg)
            
            result.ingestion_success = True
            print(f"✓ Bronze ingestion complete")
        
        # Step 4: Inspect Bronze Data
        if config.run_inspection:
            print("\n" + "="*80)
            print("BRONZE PIPELINE: Inspecting Bronze Data")
            print("="*80)
            
            inspection_result = bi_inspect.run_bronze_inspection(
                bronze_root=config.bronze_root,
                dataset_name="scada_1d_signal",
                year=config.inspection_year,
                n_files=config.inspection_n_files,
                partition_limit=config.inspection_partition_limit,
                show_partitions=config.inspection_show_partitions,
                show_schema=config.inspection_show_schema,
                show_missing=config.inspection_show_missing,
                show_summary=config.inspection_show_summary,
                verbose=True,
            )
            
            result.inspection_data = inspection_result
            if "df" in inspection_result:
                result.inspection_rows_sampled = len(inspection_result["df"])
            
            print(f"✓ Bronze inspection complete: {result.inspection_rows_sampled} rows sampled")
        
        # Step 5: Run Unit Sanity Checks
        if config.run_sanity_checks:
            print("\n" + "="*80)
            print("BRONZE PIPELINE: Running Unit Sanity Checks")
            print("="*80)
            
            bench_df, meta, csv_path, json_path = usc.run_unit_sanity_check(
                bronze_root=config.bronze_root,
                mappings_root=config.mappings_root,
                output_dir=config.outputs_root,
            )
            
            result.benchmark_df = bench_df
            result.benchmark_csv_path = csv_path
            result.sanity_check_success = True
            
            print(meta)
            print(f"\n✓ Sanity checks complete")
            if csv_path is not None:
                print(f"  Benchmarks saved: {csv_path.name}")
            else:
                print(f"  No data available for benchmark computation")
        
        result.success = True
        
        print("\n" + "="*80)
        print("BRONZE PIPELINE: Complete")
        print("="*80)
        print(f"Summary:")
        if config.sanitize_files:
            print(f"  Restored: {result.files_restored} files")
            print(f"  Sanitized: {result.files_sanitized} files")
        if config.run_ingestion:
            print(f"  Ingestion: {'✓ Success' if result.ingestion_success else '✗ Failed'}")
        if config.run_inspection:
            print(f"  Inspection: {result.inspection_rows_sampled} rows sampled")
        if config.run_sanity_checks:
            print(f"  Sanity checks: ✓ Complete")
        
        return result
        
    except Exception as e:
        result.errors.append(str(e))
        print(f"\n✗ Bronze pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return result


def get_bronze_pipeline_config_from_workspace_config(
    workspace_config,
    sanitize_files: bool = True,
    run_ingestion: bool = True,
    run_inspection: bool = True,
    run_sanity_checks: bool = True,
) -> BronzePipelineConfig:
    """
    Create a BronzePipelineConfig from centralized WorkspaceConfig.
    
    Args:
        workspace_config: WorkspaceConfig instance (from src.config)
        sanitize_files: Whether to run file sanitization
        run_ingestion: Whether to run bronze ingestion
        run_inspection: Whether to inspect bronze data
        run_sanity_checks: Whether to run sanity checks
        
    Returns:
        BronzePipelineConfig ready to pass to run_bronze_pipeline()
    """
    return BronzePipelineConfig(
        workspace_root=workspace_config.WORKSPACE_ROOT,
        data_root=workspace_config.DATA_DIR,
        bronze_root=workspace_config.BRONZE_ROOT,
        mappings_root=workspace_config.MAPPINGS_ROOT,
        outputs_root=workspace_config.OUTPUTS_DIR,
        
        sanitize_files=sanitize_files,
        restore_rejected=True,
        dry_run_sanitize=False,
        
        run_ingestion=run_ingestion,
        timezone_local="Europe/Athens",
        min_age_seconds=0,  # Notebook: ingest immediately
        stable_check_seconds=0,
        allow_duplicates=False,
        parquet_compression="zstd",
        
        run_inspection=run_inspection,
        inspection_year=2026,
        inspection_n_files=10,
        inspection_partition_limit=10,
        inspection_show_partitions=True,
        inspection_show_schema=True,
        inspection_show_missing=True,
        inspection_show_summary=True,
        
        run_sanity_checks=run_sanity_checks,
    )
