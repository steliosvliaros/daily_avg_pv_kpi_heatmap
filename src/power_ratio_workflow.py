"""
Power Ratio Calculation Workflow

Orchestrates the complete power ratio and anomaly metrics pipeline:
- Load measured (gold daily_energy) data
- Load reference (gold pvgis_reference, fallback to PVGIS loader)
- Calculate power ratio (measured / reference)
- Calculate anomaly metrics (pi, score, flag)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import re

import pandas as pd

from src.config import WorkspaceConfig
from src import silver_loader
from src import metrics_calculator


logger = logging.getLogger(__name__)


@dataclass
class PowerRatioConfig:
    """Configuration for power ratio calculation workflow."""
    
    workspace_config: WorkspaceConfig
    
    # Measured data (gold daily_energy) parameters
    measured_start_date: str = "2015-01-01"
    measured_end_date: str = "2025-12-31"
    measured_signals: list = field(default_factory=lambda: ["pcc_active_energy_export"])
    measured_signal_contains: list = field(default_factory=lambda: ["active", "energy"])
    measured_units: list = field(default_factory=lambda: ["kW", "kWh"])
    
    # Reference data (PVGIS) parameters
    reference_start_date: str = "2001-01-01"
    reference_end_date: str = "2001-12-31"
    
    # Ratio calculation parameters
    match_by_calendar_day: bool = True
    multiply_by_100: bool = False
    debug: bool = True


@dataclass
class PowerRatioResult:
    """Result object from power ratio and anomaly metrics calculation."""
    
    success: bool
    ratio: Optional[pd.DataFrame] = None
    measured: Optional[pd.DataFrame] = None
    reference: Optional[pd.DataFrame] = None
    active_parks: Optional[pd.DataFrame] = None
    active_park_ids: list = field(default_factory=list)
    # Anomaly metrics
    pi: Optional[pd.DataFrame] = None  # Performance Index (power ratio %)
    score: Optional[pd.DataFrame] = None  # Robust z-score
    flag: Optional[pd.DataFrame] = None  # -1/0/+1 classification
    errors: list = field(default_factory=list)


def calculate_power_ratio(config: PowerRatioConfig) -> PowerRatioResult:
    """
    Calculate power ratio (measured / reference) between gold and PVGIS data.
    
    Loads measured daily energy from gold, reference data from PVGIS (prefer gold table), and computes
    the ratio matched by calendar day (e.g., Jan-15 in any year matched to
    Jan-15 in the typical year).
    
    Args:
        config: PowerRatioConfig with all loading and calculation parameters
    
    Returns:
        PowerRatioResult with ratio dataframe and metadata
    """
    
    result = PowerRatioResult(success=False)
    
    try:
        # Load and validate active parks metadata
        logger.info("Loading park metadata")
        active_parks = pd.read_csv(config.workspace_config.PARK_METADATA_CSV)
        
        if active_parks is None or active_parks.empty:
            raise ValueError("Park metadata not found or empty")
        
        # Filter to active parks only (status_effective == true)
        status_series = active_parks['status_effective'].astype(str).str.strip().str.lower()
        active_parks = active_parks[status_series == 'true']
        
        if active_parks.empty:
            raise ValueError("No active parks found (status_effective != true)")
        
        active_park_ids = active_parks['park_id'].unique().tolist()
        logger.info("Found %d active parks", len(active_park_ids))
        
        result.active_parks = active_parks
        result.active_park_ids = active_park_ids
        
        # Load measured data from gold daily_energy
        logger.info("Loading measured data from gold daily_energy")
        measured_wide = silver_loader.load_gold_daily_energy(
            gold_root=config.workspace_config.GOLD_DIR,
            debug=config.debug,
        )

        # Fallback: if gold is empty, load from silver as before
        if measured_wide is None or measured_wide.empty:
            logger.warning("Gold daily_energy is empty; falling back to silver loader")

            measured_signals = config.measured_signals
            if isinstance(measured_signals, str):
                measured_signals = [measured_signals]
            if measured_signals is not None:
                measured_signals = [str(s).strip() for s in measured_signals if str(s).strip()]
                if not measured_signals or any(s.lower() == "all" for s in measured_signals):
                    measured_signals = None

            signal_name_contains = config.measured_signal_contains
            if isinstance(signal_name_contains, (list, tuple, set)):
                terms = [str(term).strip() for term in signal_name_contains if str(term).strip()]
                signal_name_contains = "|".join(re.escape(term.lower()) for term in terms) if terms else None
            elif signal_name_contains is not None:
                signal_name_contains = str(signal_name_contains).strip()
                if signal_name_contains.lower() in ("", "all"):
                    signal_name_contains = None

            measured_wide = silver_loader.load_silver_filtered_wide(
                silver_root=config.workspace_config.SILVER_ROOT,
                start_date=config.measured_start_date,
                end_date=config.measured_end_date,
                timestamp_col="ts_local",
                signals=measured_signals,
                park_id_contains="all",
                park_capacity_min=None,
                park_capacity_max=None,
                signal_name_contains=signal_name_contains,
                units=config.measured_units,
                flatten_columns=False,
                debug=config.debug,
            )
        
        if measured_wide is None or measured_wide.empty:
            raise ValueError("Failed to load measured data from silver")

        # Ensure datetime index and apply measured date filtering
        if not isinstance(measured_wide.index, pd.DatetimeIndex):
            if "ts_local" in measured_wide.columns:
                measured_wide = measured_wide.set_index("ts_local")
            elif "timestamp" in measured_wide.columns:
                measured_wide = measured_wide.set_index("timestamp")
            elif "date" in measured_wide.columns:
                measured_wide = measured_wide.set_index("date")
        measured_wide.index = pd.to_datetime(measured_wide.index, errors="coerce")
        measured_wide = measured_wide[~measured_wide.index.isna()]

        measured_start = pd.to_datetime(config.measured_start_date)
        measured_end = pd.to_datetime(config.measured_end_date)
        if getattr(measured_wide.index, "tz", None) is not None:
            measured_start = measured_start.tz_localize(measured_wide.index.tz)
            measured_end = measured_end.tz_localize(measured_wide.index.tz)
        measured_wide = measured_wide[(measured_wide.index >= measured_start) & (measured_wide.index <= measured_end)]

        # Keep only active parks where possible
        if isinstance(measured_wide.columns, pd.MultiIndex):
            keep_cols = measured_wide.columns.get_level_values(0).isin(active_park_ids)
            measured_wide = measured_wide.loc[:, keep_cols]
        else:
            keep_cols = [c for c in measured_wide.columns if str(c) in set(active_park_ids)]
            if keep_cols:
                measured_wide = measured_wide[keep_cols]
        
        logger.info("Loaded measured data: %s", measured_wide.shape)
        result.measured = measured_wide
        
        # Load reference data from gold pvgis_reference (fallback to PVGIS loader)
        logger.info("Loading reference data from gold pvgis_reference")
        reference_wide = silver_loader.load_gold_pvgis_reference(
            gold_root=config.workspace_config.GOLD_DIR,
            debug=config.debug,
        )

        if reference_wide is None or reference_wide.empty:
            logger.warning("Gold pvgis_reference is empty; falling back to load_pvgis_filtered_wide")
            reference_wide = silver_loader.load_pvgis_filtered_wide(
                workspace_root=config.workspace_config.WORKSPACE_ROOT,
                start_date=config.reference_start_date,
                end_date=config.reference_end_date,
                park_id_contains="all",
                park_capacity_min=None,
                park_capacity_max=None,
                timestamp_col="ts_local",
                flatten_columns=False,
                debug=config.debug,
            )
        
        if reference_wide is None or reference_wide.empty:
            raise ValueError("Failed to load reference data from PVGIS")

        if not isinstance(reference_wide.index, pd.DatetimeIndex):
            if "ts_local" in reference_wide.columns:
                reference_wide = reference_wide.set_index("ts_local")
            elif "timestamp" in reference_wide.columns:
                reference_wide = reference_wide.set_index("timestamp")
            elif "date" in reference_wide.columns:
                reference_wide = reference_wide.set_index("date")
        reference_wide.index = pd.to_datetime(reference_wide.index, errors="coerce")
        reference_wide = reference_wide[~reference_wide.index.isna()]
        
        logger.info("Loaded reference data: %s", reference_wide.shape)
        result.reference = reference_wide
        
        # Calculate power ratio
        logger.info("Calculating power ratio (measured / reference)")
        ratio = silver_loader.divide_wide_by_reference(
            measured_wide=measured_wide,
            reference_wide=reference_wide,
            match_by_calendar_day=config.match_by_calendar_day,
            multiply_by_100=config.multiply_by_100,
            debug=config.debug,
        )
        
        if ratio is None or ratio.empty:
            raise ValueError("Failed to calculate power ratio")
        
        logger.info("Calculated power ratio: %s", ratio.shape)
        result.ratio = ratio
        
        # Calculate anomaly metrics (pi, score, flag) from power ratio
        logger.info("Calculating anomaly metrics from power ratio")
        anomaly_metrics = metrics_calculator.calculate_anomaly_metrics(
            power_ratio_pct=ratio,
            daily_historical=measured_wide,
        )
        
        if anomaly_metrics is None:
            raise ValueError("Failed to calculate anomaly metrics")
        
        # Extract metrics
        result.pi = anomaly_metrics.get('pi')
        result.score = anomaly_metrics.get('score')
        result.flag = anomaly_metrics.get('flag')
        
        if result.pi is None or result.score is None or result.flag is None:
            raise ValueError("Anomaly metrics missing required keys (pi, score, flag)")
        
        logger.info("Calculated anomaly metrics: pi %s, score %s, flag %s", 
                   result.pi.shape, result.score.shape, result.flag.shape)
        
        result.success = True
        logger.info("Power ratio and anomaly metrics calculation completed successfully")
        
        
    except Exception as e:
        logger.exception("Power ratio calculation failed")
        result.success = False
        result.errors.append(str(e))
    
    return result
