"""
PVGIS Typical-Year Data Pipeline

Orchestrates the complete PVGIS workflow:
- Configuration management
- Metadata validation and filtering
- Cache detection and management
- Conditional data download
- EDA output handling
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import pandas as pd

from src.config import WorkspaceConfig
from src import pvgis_data_ingestion as pr_est
from src.silver_prepair import load_park_metadata


logger = logging.getLogger(__name__)


@dataclass
class PVGISPipelineConfig:
    """Configuration for PVGIS typical-year pipeline."""
    
    workspace_config: WorkspaceConfig
    
    # Data source configuration
    use_cache: bool = True
    save_cache: bool = True
    save_output: bool = True
    
    # Year range for analysis
    start_year: int = 2015
    end_year: int = 2023
    
    # System parameters
    loss_pct: float = 18.0
    default_capacity_kwp: float = 100.0
    default_timezone: str = "Europe/Athens"
    reference_year: int = 2001
    
    # Data cleaning
    drop_feb29: bool = True
    
    # EDA options
    run_eda_on_new: bool = True
    save_eda_plots: bool = False
    save_eda_stats: bool = False
    show_eda_plots: bool = False
    
    # Gold layer options
    ingest_to_gold: bool = True
    force_gold_replace: bool = False


@dataclass
class PVGISPipelineResult:
    """Result object from PVGIS pipeline execution."""
    
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    eda_outputs: Optional[Dict[str, Any]] = None
    cached_parks: int = 0
    downloaded_parks: int = 0
    total_parks: int = 0
    errors: list = field(default_factory=list)
    
    # Gold ingestion results
    gold_ingestion_result: Optional[Dict[str, Any]] = None
    
    @property
    def cache_hit(self) -> bool:
        """Returns True if all data was from cache."""
        return self.downloaded_parks == 0 and self.cached_parks > 0


def run_pvgis_pipeline(config: PVGISPipelineConfig) -> PVGISPipelineResult:
    """
    Execute complete PVGIS typical-year pipeline.
    
    Args:
        config: PVGISPipelineConfig with all settings
    
    Returns:
        PVGISPipelineResult with dataframe and metadata
    """
    
    result = PVGISPipelineResult(success=False)
    
    try:
        # Create TypicalYearConfig from pipeline config
        typ_cfg = pr_est.TypicalYearConfig(
            workspace_config=config.workspace_config,
            use_cache=config.use_cache,
            save_cache=config.save_cache,
            save_output=config.save_output,
            start_year=config.start_year,
            end_year=config.end_year,
            loss_pct=config.loss_pct,
            default_capacity_kwp=config.default_capacity_kwp,
            default_timezone=config.default_timezone,
            reference_year=config.reference_year,
            drop_feb29=config.drop_feb29,
            run_eda_on_new=config.run_eda_on_new,
            save_eda_plots=config.save_eda_plots,
            save_eda_stats=config.save_eda_stats,
            show_eda_plots=config.show_eda_plots,
        )
        
        # Load and filter metadata to active parks only
        park_meta = load_park_metadata(typ_cfg.metadata_path)
        if park_meta is None:
            raise ValueError("park_metadata not found; required for PVGIS download")
        
        if "status_effective" not in park_meta.columns:
            raise ValueError("park_metadata missing status_effective column")
        
        status_series = park_meta["status_effective"].astype("string").str.strip().str.lower()
        park_meta = park_meta[status_series == "true"]
        
        if park_meta.empty:
            raise ValueError("No parks with status_effective=true in park_metadata")
        
        logger.info("Loaded %d active parks from metadata", len(park_meta))
        
        # Detect cached parks
        daily_cache_dir = config.workspace_config.PVGIS_CACHE_TYPICAL_DAILY
        cached_parks = set()
        
        if daily_cache_dir.exists():
            for cache_file in daily_cache_dir.glob("pvgis_typical_daily_*.parquet"):
                parts = cache_file.stem.split("_")
                if len(parts) >= 4:
                    park_id = "_".join(parts[3:-1])
                    cached_parks.add(park_id)
        
        expected_parks = set(park_meta["park_id"].astype(str).str.lower())
        result.total_parks = len(expected_parks)
        result.cached_parks = len(cached_parks)
        
        # Determine if download is needed
        if cached_parks and cached_parks == expected_parks:
            logger.info("Cache hit: all %d parks already downloaded", len(cached_parks))
            pvgis_outputs = {"message": "Using cached data", "parks": len(cached_parks)}
            result.downloaded_parks = 0
        else:
            missing = expected_parks - cached_parks
            result.downloaded_parks = len(missing)
            logger.info("Cache miss: downloading PVGIS for %d parks (missing %d)", 
                       len(expected_parks), len(missing))
            pvgis_outputs = pr_est.build_pvgis_typical_year_dataset(typ_cfg)
        
        # Extract results
        result.dataframe = pvgis_outputs.get("dataframe")
        result.eda_outputs = pvgis_outputs.get("eda_outputs")
        
        # Ingest to gold layer if requested
        if config.ingest_to_gold and result.dataframe is not None and not result.dataframe.empty:
            logger.info("Ingesting PVGIS data to gold layer...")
            try:
                from src.silver_loader import load_and_ingest_pvgis_reference_gold
                
                # Resolve parquet path for ingestion
                pvgis_temp_path = pvgis_outputs.get("output_path") or config.workspace_config.PVGIS_OUTPUT_TYPICAL_DAILY
                if not Path(pvgis_temp_path).exists():
                    raise FileNotFoundError(f"PVGIS parquet not found for gold ingestion: {pvgis_temp_path}")
                
                gold_result = load_and_ingest_pvgis_reference_gold(
                    pvgis_path=pvgis_temp_path,
                    metadata_path=config.workspace_config.PARK_METADATA_CSV,
                    gold_root=config.workspace_config.GOLD_DIR,
                    force_replace=config.force_gold_replace,
                    debug=True,
                )
                
                result.gold_ingestion_result = gold_result
                logger.info("Gold ingestion result: %s", gold_result.get("status"))
                
            except Exception as e:
                logger.error("Gold ingestion failed: %s", str(e))
                result.errors.append(f"Gold ingestion failed: {str(e)}")
        
        result.success = True
        logger.info("PVGIS pipeline completed successfully")
        
    except Exception as e:
        logger.exception("PVGIS pipeline failed")
        result.success = False
        result.errors.append(str(e))
    
    return result


def build_pvgis_expected_daily_wide(config: PVGISPipelineConfig) -> pd.DataFrame:
    """
    Thin wrapper to build wide expected daily PVGIS production from park metadata.

    Uses the same pipeline config/workspace defaults and returns a wide dataframe
    with one column per ``park_iso_name``.
    """
    return pr_est.build_pvgis_expected_daily_wide_from_metadata(
        metadata_path=config.workspace_config.PARK_METADATA_CSV,
        cache_root=config.workspace_config.PVGIS_CACHE,
        start_year=config.start_year,
        end_year=config.end_year,
        loss_pct=config.loss_pct,
        reference_year=config.reference_year,
        drop_feb29=config.drop_feb29,
        default_timezone=config.default_timezone,
        use_cache=config.use_cache,
        save_cache=config.save_cache,
        force_download=False,
    )
