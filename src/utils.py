from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename derived from a title or label."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    safe = safe.strip("._")
    return safe or "figure"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if missing and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_versioned_filename(
    base_name: str,
    save_dir: Path,
    fmt: str = "png",
    add_date: bool = True,
) -> str:
    """
    Generate a versioned filename with YYYYMMDD date and version suffix.
    
    If a file with the same base name and date exists, increments version (v001, v002, etc.).
    
    Parameters
    ----------
    base_name : str
        Base name for the file (will be sanitized)
    save_dir : Path
        Directory where file will be saved
    fmt : str
        File format/extension
    add_date : bool
        Whether to add YYYYMMDD date to filename
    
    Returns
    -------
    str
        Versioned filename without extension
    """
    safe_name = sanitize_filename(base_name)
    
    if add_date:
        date_str = datetime.now().strftime("%Y%m%d")
        pattern = f"{safe_name}_{date_str}"
    else:
        pattern = safe_name
        date_str = None
    
    # Find existing files with same pattern
    existing_files = list(save_dir.glob(f"{pattern}*.{fmt}"))
    
    if not existing_files:
        # First version - always add version suffix
        return f"{pattern}_v001"
    
    # Extract version numbers from existing files
    versions = []
    for f in existing_files:
        stem = f.stem
        if "_v" in stem:
            try:
                version_part = stem.split("_v")[-1]
                version_num = int(version_part)
                versions.append(version_num)
            except ValueError:
                continue
    
    if versions:
        next_version = max(versions) + 1
    else:
        next_version = 1
    
    return f"{pattern}_v{next_version:03d}"


def save_figure(
    fig,
    title_prefix: str = "Figure",
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    base_filename: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png",
    auto_version: bool = False,
    add_date: bool = False,
):
    """
    Save a Matplotlib figure with consistent defaults.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    title_prefix : str
        Used to derive a default filename when `base_filename` is not provided
    save : bool
        When False, does nothing and returns None
    save_dir : str | Path | None
        Target directory (created if missing). If None, uses CWD/"plots".
    base_filename : str | None
        Filename without extension. If None, derived from `title_prefix`.
    dpi : int
        Image resolution
    fmt : str
        File format (e.g., "png", "pdf", "svg")
    auto_version : bool
        If True, automatically adds date (YYYYMMDD) and version (v001, v002, etc.)
    add_date : bool
        If True with auto_version, adds YYYYMMDD to filename

    Returns
    -------
    Path | None
        The path to the saved file, or None if `save` is False.
    """
    if not save:
        return None

    if save_dir is None:
        save_dir = Path.cwd() / "plots"
    out_dir = ensure_dir(save_dir)

    if not base_filename:
        base_filename = sanitize_filename(title_prefix) or "figure"
    
    if auto_version:
        base_filename = generate_versioned_filename(
            base_name=base_filename,
            save_dir=out_dir,
            fmt=fmt,
            add_date=add_date,
        )

    out_path = out_dir / f"{base_filename}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=fmt)
    return out_path


__all__ = ["sanitize_filename", "ensure_dir", "save_figure", "generate_versioned_filename"]
