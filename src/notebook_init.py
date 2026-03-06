"""
Notebook initialization module.

Provides workspace setup, logging configuration, and module reloading utilities
for Jupyter notebooks.
"""

import sys
import logging
from pathlib import Path
import importlib

from src.config import setup_workspace, Settings


def init_workspace(workspace_root: Path = None, verbose: bool = True) -> tuple:
    """
    Initialize workspace and logging configuration.
    
    Args:
        workspace_root: Root directory of the workspace. If None, assumes notebooks are
                       in a 'notebooks' subfolder and workspace root is one level up.
        verbose: Whether to print setup information.
    
    Returns:
        Tuple of (config, logger) where config is the WorkspaceConfig object
        and logger is the notebook logger.
    """
    # Determine workspace root if not provided
    if workspace_root is None:
        workspace_root = Path.cwd().parent.resolve()
    
    # Add workspace root to path (idempotent)
    workspace_root_str = str(workspace_root)
    if workspace_root_str not in sys.path:
        sys.path.insert(0, workspace_root_str)
    
    # Setup workspace and get config
    config = setup_workspace(workspace_root=workspace_root, verbose=verbose)
    
    # Configure notebook-wide logging
    logging.basicConfig(
        level=Settings.LOG_LEVEL,
        format=Settings.LOG_FORMAT,
        force=True,
    )
    logger = logging.getLogger("notebook")
    logger.setLevel(Settings.LOG_LEVEL)
    logger.info("Workspace initialized from config")
    
    return config, logger


def reload_modules(*modules) -> None:
    """
    Reload one or more Python modules.
    
    Useful for picking up code changes during notebook development.
    
    Args:
        *modules: Module objects to reload.
    """
    for m in modules:
        importlib.reload(m)
