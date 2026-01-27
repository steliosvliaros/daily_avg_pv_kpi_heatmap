"""
Metrics Calculation Utilities

This module provides functions for calculating aggregated metrics and KPIs
from time series energy data.
"""

import pandas as pd
import numpy as np


def analyze_month_to_date_by_year(
    df: pd.DataFrame,
    column: str | None = None,
    aggregation: str = 'sum',
    current_date: pd.Timestamp | str | None = None,
) -> pd.Series | list:
    """
    Analyze month-to-date values for a specific column (or all columns) across all years in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame where each column represents a park/series
    column : str or None
        Column name to analyze. If None, aggregates across all columns
    aggregation : str
        Aggregation method: 'sum', 'mean'/'avg'/'average', 'min', 'max', 'median', 'std', 'count'
    current_date : pd.Timestamp, str, or None
        Reference date (default: today). Determines the month-to-date window
        
    Returns
    -------
    pd.Series
        Series indexed by year with aggregated values
    list
        If column is None and not found, returns list of available columns
    """
    # Set current date (default to today)
    if current_date is None:
        current_date = pd.Timestamp.now()
    else:
        current_date = pd.to_datetime(current_date)
    
    # Get the timezone from the dataframe index
    df_tz = df.index.tz
    
    # Make current_date tz-aware if needed to match df
    if df_tz is not None and current_date.tz is None:
        current_date = current_date.tz_localize(df_tz)
    elif df_tz is None and current_date.tz is not None:
        current_date = current_date.tz_localize(None)
    
    # Extract current month and day
    current_month = current_date.month
    current_day = current_date.day
    
    # Map aggregation aliases
    agg_map = {'avg': 'mean', 'average': 'mean'}
    aggregation = agg_map.get(aggregation.lower(), aggregation.lower())
    
    # Validate aggregation method
    valid_aggs = ['sum', 'mean', 'min', 'max', 'median', 'std', 'count']
    if aggregation not in valid_aggs:
        raise ValueError(f"Invalid aggregation '{aggregation}'. Valid options: {valid_aggs}")
    
    # Handle None column - aggregate across all columns
    if column is None:
        # Get all unique years in the dataset
        years = df.index.year.unique()
        results = {}
        
        for year in sorted(years):
            start_date = pd.Timestamp(year=year, month=current_month, day=1)
            end_date = pd.Timestamp(year=year, month=current_month, day=current_day)
            
            # Match timezone
            if df_tz is not None:
                start_date = start_date.tz_localize(df_tz)
                end_date = end_date.tz_localize(df_tz)
            
            mask = (df.index >= start_date) & (df.index <= end_date)
            period_data = df.loc[mask].dropna(how='all')
            
            if len(period_data) == 0:
                continue
            
            if aggregation == 'sum':
                value = period_data.sum().sum()
            elif aggregation == 'mean':
                value = period_data.stack().mean()
            elif aggregation == 'min':
                value = period_data.min().min()
            elif aggregation == 'max':
                value = period_data.max().max()
            elif aggregation == 'median':
                value = period_data.stack().median()
            elif aggregation == 'std':
                value = period_data.stack().std()
            elif aggregation == 'count':
                value = period_data.count().sum()
            
            results[year] = value
        
        result = pd.Series(results, name=f'{aggregation.capitalize()}')
        result.index.name = 'Year'
        
        month_name = current_date.strftime('%B')
        print(f"\n📊 Analysis: {month_name} 1-{current_day} ({aggregation}) for ALL COLUMNS")
        print(f"   Columns count: {len(df.columns)}")
        print(f"   Years analyzed: {len(result)}")
        print(f"   Date range per year: {month_name} 1 - {month_name} {current_day}")
        print(f"\n{result.to_string()}")
        
        return result
    
    # Check if column exists
    if column not in df.columns:
        print(f"❌ Column '{column}' not found in dataframe")
        print(f"\n📋 Available columns ({len(df.columns)}):")
        return list(df.columns)
    
    # Single column analysis
    years = df.index.year.unique()
    results = {}
    
    for year in sorted(years):
        start_date = pd.Timestamp(year=year, month=current_month, day=1)
        end_date = pd.Timestamp(year=year, month=current_month, day=current_day)
        
        # Match timezone
        if df_tz is not None:
            start_date = start_date.tz_localize(df_tz)
            end_date = end_date.tz_localize(df_tz)
        
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_data = df.loc[mask, column].dropna()
        
        if len(period_data) == 0:
            continue
        
        if aggregation == 'sum':
            value = period_data.sum()
        elif aggregation == 'mean':
            value = period_data.mean()
        elif aggregation == 'min':
            value = period_data.min()
        elif aggregation == 'max':
            value = period_data.max()
        elif aggregation == 'median':
            value = period_data.median()
        elif aggregation == 'std':
            value = period_data.std()
        elif aggregation == 'count':
            value = period_data.count()
        
        results[year] = value
    
    result = pd.Series(results, name=f'{aggregation.capitalize()}')
    result.index.name = 'Year'
    
    month_name = current_date.strftime('%B')
    print(f"\n📊 Analysis: {month_name} 1-{current_day} ({aggregation}) for '{str(column)}'")
    print(f"   Column: {column}")
    print(f"   Years analyzed: {len(result)}")
    print(f"   Date range per year: {month_name} 1 - {month_name} {current_day}")
    print(f"\n{result.to_string()}")
    
    return result


def aggregate_month_to_date_by_column(
    df: pd.DataFrame,
    aggregation: str = 'sum',
    current_date: pd.Timestamp | str | None = None,
) -> pd.Series:
    """
    Return month-to-date aggregated values for each column.
    
    This computes the aggregation from the first day of the month up to `current_date`
    for the year of `current_date` (default: today), returning a Series indexed by
    column with one aggregated value per column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame where each column represents a park/series
    aggregation : str
        One of: 'sum', 'mean'/'avg'/'average', 'min', 'max', 'median', 'std', 'count'
    current_date : pd.Timestamp, str, or None
        The reference date defining the month-to-date window. If None, uses today
    
    Returns
    -------
    pd.Series
        Series indexed by column with the month-to-date aggregated value for each column
    
    Examples
    --------
    # Sum of kWh for each park for the current month-to-date
    aggregate_month_to_date_by_column(daily_historical, aggregation='sum')
    
    # Mean kWh for each park month-to-date on a specific date
    aggregate_month_to_date_by_column(daily_historical, aggregation='mean', current_date='2026-01-17')
    """
    # Normalize and validate current_date
    if current_date is None:
        current_date = pd.Timestamp.now()
    else:
        current_date = pd.to_datetime(current_date)
    
    # Ensure datetime index (naive)
    idx = pd.to_datetime(df.index)
    df = df.copy()
    df.index = idx
    
    # Map aggregation aliases
    agg_map = {
        'avg': 'mean',
        'average': 'mean',
    }
    agg = agg_map.get(str(aggregation).lower(), str(aggregation).lower())
    
    valid_aggs = ['sum', 'mean', 'min', 'max', 'median', 'std', 'count']
    if agg not in valid_aggs:
        raise ValueError(f"Invalid aggregation '{aggregation}'. Valid options: {valid_aggs}")
    
    # Build month-to-date window for the current year
    year = int(current_date.year)
    month = int(current_date.month)
    day = int(current_date.day)
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year, month=month, day=day)
    
    # Slice the DataFrame for the period
    mask = (df.index >= start_date) & (df.index <= end_date)
    period_df = df.loc[mask]
    
    if period_df.empty:
        # Return a Series of NaNs with the same columns if there is no data
        return pd.Series([pd.NA] * len(df.columns), index=df.columns, name=f"MTD {agg.capitalize()}")
    
    # Compute aggregation per column
    if agg == 'sum':
        out = period_df.sum(axis=0, skipna=True)
    elif agg == 'mean':
        out = period_df.mean(axis=0, skipna=True)
    elif agg == 'min':
        out = period_df.min(axis=0, skipna=True)
    elif agg == 'max':
        out = period_df.max(axis=0, skipna=True)
    elif agg == 'median':
        out = period_df.median(axis=0, skipna=True)
    elif agg == 'std':
        out = period_df.std(axis=0, skipna=True)
    elif agg == 'count':
        out = period_df.count(axis=0)
    
    month_name = current_date.strftime('%B')
    out.name = f"MTD {agg.capitalize()} ({month_name} 1-{day})"
    return out


def calculate_revenue_from_energy(
    energy_series: pd.Series,
    price_per_kwh: float = 0.2,
    currency: str = "EUR",
) -> pd.Series:
    """
    Calculate revenue from energy data.
    
    Parameters
    ----------
    energy_series : pd.Series
        Series with energy values (kWh)
    price_per_kwh : float
        Price per kWh (default: 0.2)
    currency : str
        Currency label (default: "EUR")
        
    Returns
    -------
    pd.Series
        Revenue series with updated name
    """
    revenue = energy_series * price_per_kwh
    revenue.name = f"Revenue ({currency})"
    return revenue


def calculate_anomaly_metrics(
    power_ratio_pct: pd.DataFrame,
    daily_historical: pd.DataFrame = None,
) -> dict:
    """
    Calculate anomaly detection metrics from power ratio data.
    
    Creates three derived metrics:
    - pi: Copy of power_ratio_pct (Performance Index)
    - score: Robust z-score per park using median/MAD
    - flag: Simple -1/0/+1 classification based on score thresholds
    
    Parameters
    ----------
    power_ratio_pct : pd.DataFrame
        Power ratio percentage (measured/expected * 100)
        Date-indexed with parks as columns
    daily_historical : pd.DataFrame, optional
        Historical daily generation data (kWh/day)
        If None, uses power_ratio_pct index/columns
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'daily_historical': DataFrame of historical data
        - 'pi': DataFrame of performance index (power ratio %)
        - 'score': DataFrame of robust anomaly scores
        - 'flag': DataFrame of -1/0/+1 flags
    """
    # PI is a copy of power ratio
    pi = power_ratio_pct.copy()
    
    # Use provided historical data or create aligned placeholder
    if daily_historical is None:
        # Create aligned placeholder (not recommended for production)
        daily_historical = pd.DataFrame(
            index=power_ratio_pct.index,
            columns=power_ratio_pct.columns,
            dtype=float
        )
    
    # Robust anomaly score per park (median/MAD z-score)
    def _robust_score(series: pd.Series) -> pd.Series:
        """Calculate robust z-score using median and MAD"""
        med = series.median()
        mad = (series - med).abs().median()
        
        # Return zeros if MAD is zero or NaN (no variation)
        if mad == 0 or pd.isna(mad):
            return pd.Series(0.0, index=series.index, dtype=float)
        
        # MAD to standard deviation conversion factor (1.4826)
        # This makes MAD-based z-score comparable to standard z-score
        return (series - med) / (mad * 1.4826)
    
    score = pi.apply(_robust_score, axis=0)
    
    # Simple flag: -1 under (score < -1.5), 0 neutral, +1 over (score > +1.5)
    # -1 = underperforming, 0 = normal, +1 = overperforming (or anomalous high)
    flag = score.copy()
    flag = flag.mask(score > 1.5, 1)
    flag = flag.mask(score < -1.5, -1)
    flag = flag.fillna(0)
    
    return {
        'daily_historical': daily_historical,
        'pi': pi,
        'score': score,
        'flag': flag,
    }
