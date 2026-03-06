"""
Daily PV KPI Heatmap - Source package.

Modules for data ingestion, processing, and reporting.
"""

# Notebook initialization utilities
from src.notebook_init import init_workspace, reload_modules

# Pipeline workflows
from src import (
    bronze_workflow,
    silver_workflow,
    pvgis_workflow,
    power_ratio_workflow,
)

# Core modules (auto-imported for convenience)
from src import (
    pvgis_pi_heatmap,
    silver_loader,
    visualizations,
    metrics_calculator,
    report_generator,
    degradation_analysis,
)

__all__ = [
    "init_workspace",
    "reload_modules",
    "bronze_workflow",
    "silver_workflow",
    "pvgis_workflow",
    "power_ratio_workflow",
    "pvgis_pi_heatmap",
    "silver_loader",
    "visualizations",
    "metrics_calculator",
    "report_generator",
    "degradation_analysis",
]
