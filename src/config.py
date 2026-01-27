"""
Centralized configuration for the PV KPI Heatmap project.

This module defines all workspace paths, settings, and provides a setup function
to initialize the directory structure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class WorkspaceConfig:
    """Centralized configuration for workspace paths and settings."""

    def __init__(self, workspace_root: Optional[Path] = None):
        """Initialize workspace configuration.

        Parameters
        ----------
        workspace_root : Path, optional
            Root directory of the workspace. If None, attempts to detect from cwd.
        """
        if workspace_root is None:
            # Auto-detect: if running from notebooks/, go up one level
            cwd = Path.cwd()
            if cwd.name == "notebooks":
                workspace_root = cwd.parent
            else:
                workspace_root = cwd
        
        self.WORKSPACE_ROOT = Path(workspace_root).resolve()

        # Data directories
        self.DATA_DIR = self.WORKSPACE_ROOT / "data"
        self.DATA_INBOX = self.DATA_DIR / "inbox"
        self.DATA_PROCESSING = self.DATA_DIR / "processing"
        self.DATA_REJECTED = self.DATA_DIR / "rejected"
        self.DATA_ARCHIVED = self.DATA_DIR / "archived"
        self.DATA_BACKUPS = self.DATA_DIR / "backups"
        self.DATA_LOCKS = self.DATA_DIR / "_locks"

        # Bronze layer (raw ingested data)
        self.BRONZE_ROOT = self.WORKSPACE_ROOT / "bronze"
        self.BRONZE_OPS = self.BRONZE_ROOT / "_ops"
        self.BRONZE_RUN_LOGS = self.BRONZE_OPS / "run_logs"
        self.BRONZE_DATASET = self.BRONZE_ROOT / "scada_1d_signal"

        # Silver layer (cleaned/validated data)
        self.SILVER_ROOT = self.WORKSPACE_ROOT / "silver"
        self.SILVER_OPS = self.SILVER_ROOT / "_ops"
        self.SILVER_STAGE = self.SILVER_ROOT / "_stage"

        # Mappings and metadata
        self.MAPPINGS_ROOT = self.WORKSPACE_ROOT / "mappings"
        self.PARK_METADATA_CSV = self.MAPPINGS_ROOT / "park_metadata.csv"
        self.PARK_POWER_MAPPING_CSV = self.MAPPINGS_ROOT / "park_power_mapping.csv"
        self.CURRENT_MAPPING_TXT = self.MAPPINGS_ROOT / "current.txt"

        # PVGIS data
        self.PVGIS_ROOT = self.WORKSPACE_ROOT / "pvgis"
        self.PVGIS_CACHE = self.PVGIS_ROOT / "pvgis_cache"
        self.PVGIS_CACHE_TYPICAL_DAILY = self.PVGIS_CACHE / "typical_daily"
        self.PVGIS_OUTPUT = self.PVGIS_ROOT / "pvgis_typical_year"

        # Outputs and artifacts
        self.PLOTS_DIR = self.WORKSPACE_ROOT / "plots"
        self.PLOTS_FINANCIAL = self.PLOTS_DIR / "financial_analysis"
        self.PLOTS_STL = self.PLOTS_DIR / "stl_analysis"
        self.PLOTS_WEEKLY = self.PLOTS_DIR / "weekly_analysis"
        self.PLOTS_PVGIS_EDA = self.PLOTS_DIR / "pvgis_typical_year_eda"
        
        self.DOCS_DIR = self.WORKSPACE_ROOT / "docs"
        self.OUTPUTS_DIR = self.WORKSPACE_ROOT / "outputs"
        self.UNIT_BENCHMARKS_CSV = self.OUTPUTS_DIR / "unit_sanity_benchmarks.csv"

        # Legacy file paths (for backward compatibility)
        self.DATA_XLSX = self.DATA_DIR / "daily_energy.xlsx"
        self.DATA_XLSX_HISTORICAL = self.DATA_DIR / "daily_energy_historical.xlsx"
        self.DATA_XLSX_TOTAL_IRR = self.DATA_DIR / "Exported Data-20150101T000000.xlsx"
        self.DATA_XLSX_TOTAL_FULL = self.DATA_DIR / "column_full_export.xlsx"
        self.CACHE_DIR = self.WORKSPACE_ROOT / "pvgis_cache"  # legacy

    def get_all_directories(self) -> list[Path]:
        """Return list of all directories that should be created."""
        return [
            self.DATA_DIR,
            self.DATA_INBOX,
            self.DATA_PROCESSING,
            self.DATA_REJECTED,
            self.DATA_ARCHIVED,
            self.DATA_BACKUPS,
            self.DATA_LOCKS,
            self.BRONZE_ROOT,
            self.BRONZE_OPS,
            self.BRONZE_RUN_LOGS,
            self.BRONZE_DATASET,
            self.SILVER_ROOT,
            self.SILVER_OPS,
            self.SILVER_STAGE,
            self.MAPPINGS_ROOT,
            self.PVGIS_ROOT,
            self.PVGIS_CACHE,
            self.PVGIS_CACHE_TYPICAL_DAILY,
            self.PVGIS_OUTPUT,
            self.PLOTS_DIR,
            self.PLOTS_FINANCIAL,
            self.PLOTS_STL,
            self.PLOTS_WEEKLY,
            self.PLOTS_PVGIS_EDA,
            self.DOCS_DIR,
            self.OUTPUTS_DIR,
        ]

    def setup_workspace(self, verbose: bool = True) -> int:
        """Create all required directories.

        Parameters
        ----------
        verbose : bool
            Whether to print progress messages

        Returns
        -------
        int
            Number of directories created
        """
        directories = self.get_all_directories()
        created = 0
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created += 1
        
        if verbose:
            print(f"✓ Workspace setup complete: {len(directories)} directories ({created} created)")
        
        return created

    def __repr__(self) -> str:
        return f"WorkspaceConfig(root={self.WORKSPACE_ROOT})"


# Global default instance (can be overridden)
_default_config: Optional[WorkspaceConfig] = None


def get_config(workspace_root: Optional[Path] = None) -> WorkspaceConfig:
    """Get or create the default workspace configuration.

    Parameters
    ----------
    workspace_root : Path, optional
        Root directory. If None, uses cached config or auto-detects.

    Returns
    -------
    WorkspaceConfig
        Workspace configuration instance
    """
    global _default_config
    
    if _default_config is None or workspace_root is not None:
        _default_config = WorkspaceConfig(workspace_root)
    
    return _default_config


def setup_workspace(workspace_root: Optional[Path] = None, verbose: bool = True) -> WorkspaceConfig:
    """Setup workspace directory structure and return config.

    Parameters
    ----------
    workspace_root : Path, optional
        Root directory. If None, auto-detects from cwd.
    verbose : bool
        Whether to print progress

    Returns
    -------
    WorkspaceConfig
        Configured workspace with directories created
    """
    config = get_config(workspace_root)
    config.setup_workspace(verbose=verbose)
    return config


# Project-wide settings
class Settings:
    """Project-wide settings and defaults."""
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    
    # Data processing
    TIMEZONE_LOCAL = "Europe/Athens"
    DAILY_INTERVAL_END_IS_MIDNIGHT = True
    
    # Parquet
    PARQUET_COMPRESSION = "zstd"
    
    # Economic defaults
    DEFAULT_PRICE_PER_KWH = 0.2
    DEFAULT_CURRENCY = "EUR"
    
    # Plot defaults
    DEFAULT_DPI = 150
    DEFAULT_PLOT_FORMAT = "png"
    
    # PVGIS
    PVGIS_API_TIMEOUT = 30
    PVGIS_MAX_RETRIES = 3


__all__ = [
    "WorkspaceConfig",
    "Settings",
    "get_config",
    "setup_workspace",
]
