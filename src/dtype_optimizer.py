"""
Data type optimizer for memory-efficient pandas DataFrames.

Provides functions to detect optimal data types and downcast large DataFrames
to minimize memory usage while preserving data integrity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union


def detect_optimal_dtype(series: pd.Series, verbose: bool = False) -> str:
    """
    Detect the most appropriate (most memory-efficient) data type for a Series.
    
    Args:
        series: pandas Series to analyze
        verbose: if True, print reasoning for type selection
        
    Returns:
        String representation of optimal numpy/pandas dtype
        
    Examples:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> detect_optimal_dtype(s)
        'int8'
        
        >>> s = pd.Series([1.5, 2.5, 3.5])
        >>> detect_optimal_dtype(s)
        'float32'
    """
    # Handle empty series
    if series.empty or series.isna().all():
        return 'object'
    
    # Get non-null values
    non_null = series.dropna()
    if non_null.empty:
        return 'object'
    
    dtype = series.dtype
    
    # Already object - can't optimize
    if dtype == 'object':
        # Try to infer numeric type
        try:
            numeric_series = pd.to_numeric(non_null, errors='coerce')
            if numeric_series.isna().sum() == 0:
                # All values are numeric, recurse with numeric series
                numeric_series = numeric_series[numeric_series.notna()]
                series = numeric_series
                dtype = numeric_series.dtype
            else:
                return 'object'
        except:
            return 'object'
    
    # Integer types
    if pd.api.types.is_integer_dtype(dtype):
        min_val = non_null.min()
        max_val = non_null.max()
        
        if verbose:
            print(f"  Integer type: range [{min_val}, {max_val}]")
        
        # Keep binary numeric signals as numeric (0/1)
        if set(non_null.unique()).issubset({0, 1}):
            return 'uint8'
        
        # int8: -128 to 127
        if min_val >= -128 and max_val <= 127:
            return 'int8'
        # int16: -32,768 to 32,767
        elif min_val >= -32768 and max_val <= 32767:
            return 'int16'
        # int32: -2,147,483,648 to 2,147,483,647
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return 'int32'
        else:
            return 'int64'
    
    # Unsigned integer types
    elif pd.api.types.is_unsigned_integer_dtype(dtype):
        min_val = non_null.min()
        max_val = non_null.max()
        
        if verbose:
            print(f"  Unsigned int type: range [{min_val}, {max_val}]")
        
        # uint8: 0 to 255
        if max_val <= 255:
            return 'uint8'
        # uint16: 0 to 65,535
        elif max_val <= 65535:
            return 'uint16'
        # uint32: 0 to 4,294,967,295
        elif max_val <= 4294967295:
            return 'uint32'
        else:
            return 'uint64'
    
    # Float types
    elif pd.api.types.is_float_dtype(dtype):
        min_val = non_null.min()
        max_val = non_null.max()
        
        if verbose:
            print(f"  Float type: range [{min_val}, {max_val}]")
        
        # Check if all values are actually integers
        if (non_null == non_null.astype(int)).all():
            # All floats are integers - convert to int
            int_series = non_null.astype(int)
            return detect_optimal_dtype(pd.Series(int_series), verbose=verbose)
        
        # float32: approx ±3.4e38
        # float64: approx ±1.8e308
        # Most PV production data fits in float32 without precision loss
        if np.abs(min_val) <= 3.4e38 and np.abs(max_val) <= 3.4e38:
            return 'float32'
        else:
            return 'float64'
    
    # Boolean type
    elif pd.api.types.is_bool_dtype(dtype):
        return 'bool'
    
    # Datetime types
    elif pd.api.types.is_datetime64_dtype(dtype):
        return 'datetime64[ns]'
    
    # Categorical could be used for repeated strings
    elif pd.api.types.is_object_dtype(dtype):
        unique_ratio = len(non_null.unique()) / len(non_null)
        if unique_ratio < 0.05:  # Less than 5% unique values
            return 'category'
        else:
            return 'object'
    
    else:
        return str(dtype)


def downcast_dataframe(
    df: pd.DataFrame,
    verbose: bool = False,
    exclude_columns: Optional[list] = None,
    target_memory_mb: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    """
    Downcast all columns in a DataFrame to their optimal data types.
    
    Args:
        df: pandas DataFrame to downcast
        verbose: if True, print detailed information
        exclude_columns: list of column names to skip
        target_memory_mb: if specified, report memory reduction
        
    Returns:
        Tuple of (downcasted_df, dtype_mapping_dict)
        where dtype_mapping_dict = {col: (old_dtype, new_dtype), ...}
        
    Examples:
        >>> df = pd.read_csv('large_file.csv')
        >>> df_optimized, mapping = downcast_dataframe(df, verbose=True)
        >>> print(mapping)
        {'power_kw': ('float64', 'float32'), 'count': ('int64', 'int8'), ...}
    """
    exclude_columns = exclude_columns or []
    dtype_mapping = {}
    
    # Calculate original memory
    original_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"DATAFRAME TYPE OPTIMIZATION")
        print(f"{'='*80}")
        print(f"\nOriginal DataFrame:")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {original_memory:.2f} MB")
        print(f"\nAnalyzing columns...")
    
    # Downcast each column
    df_optimized = df.copy()
    
    for col in df.columns:
        if col in exclude_columns:
            if verbose:
                print(f"  ⊘ {col}: SKIPPED (excluded)")
            dtype_mapping[col] = (str(df[col].dtype), str(df[col].dtype))
            continue
        
        old_dtype = str(df[col].dtype)
        new_dtype = detect_optimal_dtype(df[col], verbose=verbose)
        
        try:
            if new_dtype == 'category':
                df_optimized[col] = df[col].astype('category')
            else:
                df_optimized[col] = df[col].astype(new_dtype)
            
            dtype_mapping[col] = (old_dtype, new_dtype)
            
            if verbose and old_dtype != new_dtype:
                old_mem = df[col].memory_usage(deep=True) / 1024  # KB
                new_mem = df_optimized[col].memory_usage(deep=True) / 1024  # KB
                reduction = (old_mem - new_mem) / old_mem * 100 if old_mem > 0 else 0
                print(f"  ✓ {col}: {old_dtype:12} → {new_dtype:12} ({reduction:+.1f}%)")
        except Exception as e:
            if verbose:
                print(f"  ⚠ {col}: Could not convert from {old_dtype} to {new_dtype} ({str(e)[:50]})")
            dtype_mapping[col] = (old_dtype, old_dtype)
            df_optimized[col] = df[col]
    
    # Calculate optimized memory
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"Original Memory:  {original_memory:>10.2f} MB")
        print(f"Optimized Memory: {optimized_memory:>10.2f} MB")
        print(f"Reduction:        {memory_reduction:>10.1f}%")
        print(f"Space Saved:      {original_memory - optimized_memory:>10.2f} MB")
        if target_memory_mb:
            print(f"\nTarget limit:     {target_memory_mb:>10.2f} MB")
            if optimized_memory <= target_memory_mb:
                print(f"✓ FITS within memory budget!")
            else:
                print(f"✗ Still exceeds memory budget by {optimized_memory - target_memory_mb:.2f} MB")
    
    return df_optimized, dtype_mapping


def smart_read_csv(
    filepath: Union[str, pd.io.common.PathLike],
    downcast: bool = True,
    chunksize: Optional[int] = None,
    verbose: bool = False,
    **read_csv_kwargs
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Read CSV with automatic dtype optimization.
    
    Args:
        filepath: path to CSV file
        downcast: if True, automatically downcast dtypes after reading
        chunksize: if specified, returns an iterator of chunks
        verbose: if True, print optimization details
        **read_csv_kwargs: additional arguments to pass to pd.read_csv
        
    Returns:
        Optimized DataFrame (or TextFileReader if chunksize specified)
        
    Examples:
        >>> df = smart_read_csv('large_file.csv', downcast=True, verbose=True)
        >>> for chunk in smart_read_csv('large_file.csv', chunksize=10000):
        ...     process(chunk)
    """
    if verbose:
        print(f"Reading CSV: {filepath}")
    
    # Read with default inference
    df = pd.read_csv(filepath, chunksize=chunksize, **read_csv_kwargs)
    
    if chunksize is not None:
        # Return iterator - can't optimize without reading all
        return df
    
    if downcast:
        df, mapping = downcast_dataframe(df, verbose=verbose)
        return df
    else:
        return df


def estimate_memory_after_downcast(df: pd.DataFrame) -> float:
    """
    Estimate memory usage after optimal downcast without modifying the DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
        
    Returns:
        Estimated memory in MB after downcast
    """
    total_bytes = 0
    
    for col in df.columns:
        optimal_dtype = detect_optimal_dtype(df[col])
        n_rows = len(df)
        
        # Estimate based on dtype
        if optimal_dtype == 'bool':
            bytes_per_val = 1
        elif optimal_dtype in ['int8', 'uint8']:
            bytes_per_val = 1
        elif optimal_dtype in ['int16', 'uint16']:
            bytes_per_val = 2
        elif optimal_dtype in ['int32', 'uint32']:
            bytes_per_val = 4
        elif optimal_dtype == 'float32':
            bytes_per_val = 4
        elif optimal_dtype in ['int64', 'uint64', 'float64', 'datetime64[ns]']:
            bytes_per_val = 8
        elif optimal_dtype == 'category':
            # Rough estimate for categorical
            bytes_per_val = 2
        else:
            # object - use actual memory
            bytes_per_val = df[col].memory_usage(deep=True) / n_rows
        
        total_bytes += bytes_per_val * n_rows
    
    return total_bytes / (1024 ** 2)  # Convert to MB
