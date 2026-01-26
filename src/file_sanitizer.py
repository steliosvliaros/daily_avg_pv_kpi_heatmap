"""
File sanitization utilities for vendor Excel files.

Handles column name transformation using mapping files and manages
backup/restoration of files in the data pipeline.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Constants
MEASUREMENT_COLUMN_MARKER = "__"  # Sanitized measurement columns contain double underscore
BACKUP_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


class SanitizationResult:
    """Results of a file sanitization operation."""
    
    def __init__(self):
        self.success: List[str] = []
        self.failed: List[Tuple[str, str]] = []  # (filename, error_message)
        self.warnings: List[Tuple[str, str]] = []  # (filename, warning_message)
    
    @property
    def success_count(self) -> int:
        return len(self.success)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)


def restore_rejected_files(
    data_root: Path,
    verbose: bool = True
) -> int:
    """
    Restore rejected files from data/rejected/ back to data/inbox/ for reprocessing.
    
    Args:
        data_root: Root data directory containing rejected/ and inbox/ subdirectories
        verbose: Print progress messages
    
    Returns:
        Number of files restored
    """
    data_root = Path(data_root)
    rejected_dir = data_root / "rejected"
    inbox_dir = data_root / "inbox"
    
    if not rejected_dir.exists():
        if verbose:
            print("No rejected directory found")
        return 0
    
    # Find all rejected Excel files
    rejected_files = list(rejected_dir.glob("year=*/month=*/day=*/*.xlsx"))
    
    if not rejected_files:
        if verbose:
            print("No rejected files found")
        return 0
    
    inbox_dir.mkdir(parents=True, exist_ok=True)
    restored_count = 0
    
    if verbose:
        print(f"Found {len(rejected_files)} rejected files to restore")
    
    for rejected_file in rejected_files:
        try:
            # Extract original filename (before the __reason hash)
            original_name = rejected_file.name.split("__reason")[0] + ".xlsx"
            inbox_file = inbox_dir / original_name
            
            shutil.copy2(rejected_file, inbox_file)
            restored_count += 1
            
            if verbose:
                print(f"  ✓ Restored: {original_name}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to restore {rejected_file.name}: {e}")
    
    return restored_count


def sanitize_inbox_files(
    inbox_dir: Path | str,
    output_dir: Path | str,
    mapping_path: Path | str,
    backup_dir: Path | str | None = None,
    dry_run: bool = False,
    verbose: bool = True
) -> SanitizationResult:
    """
    Sanitize vendor column names in Excel files using a mapping.
    
    Args:
        inbox_dir: Directory containing original vendor Excel files
        output_dir: Directory to write sanitized files (can be same as inbox_dir for in-place)
        mapping_path: Path to park_power_mapping_vXXX.csv file
        backup_dir: Optional directory to store timestamped backups
        dry_run: If True, only validate without modifying files
        verbose: Print progress messages
    
    Returns:
        SanitizationResult with success/failure/warning details
    """
    inbox_dir = Path(inbox_dir)
    output_dir = Path(output_dir)
    mapping_path = Path(mapping_path)
    
    result = SanitizationResult()
    
    # Validate inputs
    if not inbox_dir.exists():
        raise FileNotFoundError(f"Inbox directory not found: {inbox_dir}")
    
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    # Load mapping
    try:
        mapping_df = pd.read_csv(mapping_path)
        col_mapping = dict(zip(mapping_df['original'], mapping_df['sanitized']))
    except Exception as e:
        raise ValueError(f"Failed to load mapping from {mapping_path}: {e}")
    
    if verbose:
        print(f"Loaded mapping with {len(col_mapping)} column transformations")
    
    # Ensure output directory exists
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process Excel files
    excel_files = list(inbox_dir.glob("*.xlsx"))
    
    if verbose:
        print(f"Found {len(excel_files)} Excel files in {inbox_dir.name}")
    
    for excel_file in excel_files:
        try:
            if verbose:
                print(f"\nProcessing: {excel_file.name}")
            
            # Read Excel file
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                error_msg = f"Failed to read Excel file: {e}"
                result.failed.append((excel_file.name, error_msg))
                if verbose:
                    print(f"  ✗ {error_msg}")
                continue
            
            if verbose:
                print(f"  Original columns: {len(df.columns)}")
            
            # Apply column mapping
            rename_dict = {col: col_mapping.get(col, col) for col in df.columns}
            df_sanitized = df.rename(columns=rename_dict)
            
            # Validate transformation
            measurement_cols = [
                c for c in df_sanitized.columns 
                if MEASUREMENT_COLUMN_MARKER in c and not c.startswith('datetime')
            ]
            
            if verbose:
                print(f"  Sanitized columns: {len(df_sanitized.columns)}")
                print(f"  Measurement columns: {len(measurement_cols)}")
            
            if not measurement_cols:
                warning_msg = "No measurement columns found after mapping"
                result.warnings.append((excel_file.name, warning_msg))
                if verbose:
                    print(f"  ⚠️  WARNING: {warning_msg}")
                    print(f"  Sample columns: {list(df_sanitized.columns[:5])}")
                continue
            
            # Skip writing if dry run
            if dry_run:
                result.success.append(excel_file.name)
                if verbose:
                    print(f"  ✓ Validation passed (dry run)")
                continue
            
            # Create backup if requested
            if backup_dir:
                backup_dir = Path(backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime(BACKUP_TIMESTAMP_FORMAT)
                backup_filename = f"{excel_file.stem}_{timestamp}_BACKUP.xlsx"
                backup_path = backup_dir / backup_filename
                
                shutil.copy2(excel_file, backup_path)
                if verbose:
                    print(f"  Backup: {backup_filename}")
            
            # Write sanitized file
            output_path = output_dir / excel_file.name
            df_sanitized.to_excel(output_path, index=False)
            
            result.success.append(excel_file.name)
            if verbose:
                print(f"  ✓ Sanitized successfully")
        
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            result.failed.append((excel_file.name, error_msg))
            if verbose:
                print(f"  ✗ ERROR: {error_msg}")
    
    return result
