from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


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


def save_figure(
    fig,
    title_prefix: str = "Figure",
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    base_filename: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png",
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

    out_path = out_dir / f"{base_filename}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=fmt)
    return out_path


__all__ = ["sanitize_filename", "ensure_dir", "save_figure"]
