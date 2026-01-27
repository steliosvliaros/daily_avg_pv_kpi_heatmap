"""
Metrics Calculation Utilities

This module provides functions for calculating aggregated metrics and KPIs
from time series energy data.
"""

import pandas as pd
import numpy as np


def annual_mtd_energy(
    df: pd.DataFrame,
    agg: str | callable = "sum",
    per_park: bool = True,
    current_date: pd.Timestamp | str | None = None,
    parks: list | tuple | None = None,
) -> pd.DataFrame | pd.Series:
    """
    Compute month-to-date (1..current day) aggregates for each calendar year.

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame where each column represents a park/series.
    agg : str or callable, default "sum"
        Aggregation to apply. Strings support pandas aggregations
        ('sum', 'mean'/'avg'/'average', 'min', 'max', 'median', 'std', 'count').
        A callable receives a Series and must return a scalar.
    per_park : bool, default True
        If True, returns a DataFrame with one column per park.
        If False, returns a Series aggregating across all parks per year.
    current_date : pd.Timestamp, str, or None
        Reference date (defaults to now). Defines the month/day for the MTD window.
    parks : list/tuple or None
        Optional subset of park/column names to include.

    Returns
    -------
    pd.DataFrame
        If per_park=True, index is Year and columns are parks.
    pd.Series
        If per_park=False, index is Year with aggregated values across parks.
    """

    # Normalize current_date and timezone alignment
    current_date = pd.to_datetime(current_date) if current_date is not None else pd.Timestamp.now()
    df_tz = df.index.tz
    if df_tz is not None and current_date.tz is None:
        current_date = current_date.tz_localize(df_tz)
    elif df_tz is None and current_date.tz is not None:
        current_date = current_date.tz_localize(None)

    month = current_date.month
    day = current_date.day

    # Resolve aggregation
    agg_alias = {"avg": "mean", "average": "mean"}
    if isinstance(agg, str):
        agg_name = agg_alias.get(agg.lower(), agg.lower())
        valid_aggs = ["sum", "mean", "min", "max", "median", "std", "count"]
        if agg_name not in valid_aggs:
            raise ValueError(f"Invalid aggregation '{agg}'. Valid options: {valid_aggs} or a callable.")
        agg_func = agg_name
        agg_label = agg_name.capitalize()
    elif callable(agg):
        agg_func = agg
        agg_label = getattr(agg, "__name__", "custom")
    else:
        raise TypeError("agg must be a string or callable")

    cols = list(parks) if parks is not None else list(df.columns)
    years = sorted(pd.unique(df.index.year))

    if per_park:
        rows = {}
        for year in years:
            start = pd.Timestamp(year=year, month=month, day=1)
            end = pd.Timestamp(year=year, month=month, day=day)
            if df_tz is not None:
                start = start.tz_localize(df_tz)
                end = end.tz_localize(df_tz)
            mask = (df.index >= start) & (df.index <= end)
            period_df = df.loc[mask, cols]
            if period_df.empty:
                continue
            if callable(agg_func):
                aggregated = period_df.apply(agg_func, axis=0)
            else:
                aggregated = period_df.agg(agg_func)
            rows[year] = aggregated

        result = pd.DataFrame.from_dict(rows, orient="index")
        result.index.name = "Year"
        result.columns.name = "Park"
        return result

    # Aggregate across all parks
    results = {}
    for year in years:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = pd.Timestamp(year=year, month=month, day=day)
        if df_tz is not None:
            start = start.tz_localize(df_tz)
            end = end.tz_localize(df_tz)
        mask = (df.index >= start) & (df.index <= end)
        period_df = df.loc[mask, cols]
        if period_df.empty:
            continue
        
        # Flatten the dataframe to a 1D series of all values
        # For MultiIndex columns, we want all values regardless of structure
        flat_values = period_df.values.flatten()
        # Remove NaNs
        flat_values = flat_values[~pd.isna(flat_values)]
        if len(flat_values) == 0:
            continue
        
        if callable(agg_func):
            results[year] = agg_func(flat_values)
        else:
            # Use numpy functions for aggregation on flattened array
            if agg_func == 'sum':
                results[year] = np.sum(flat_values)
            elif agg_func == 'mean':
                results[year] = np.mean(flat_values)
            elif agg_func == 'min':
                results[year] = np.min(flat_values)
            elif agg_func == 'max':
                results[year] = np.max(flat_values)
            else:
                # Fallback: convert back to series and use pandas method
                results[year] = getattr(pd.Series(flat_values), agg_func)()

    series = pd.Series(results, name=f"MTD {agg_label}")
    series.index.name = "Year"
    return series


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
    
    # Match timezone if the index is timezone-aware
    if df.index.tz is not None:
        start_date = start_date.tz_localize(df.index.tz)
        end_date = end_date.tz_localize(df.index.tz)
    
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
