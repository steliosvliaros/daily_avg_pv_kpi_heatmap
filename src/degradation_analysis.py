"""
Degradation Analysis using STL Decomposition

This module provides time series decomposition and degradation analysis
using Seasonal-Trend decomposition with LOESS (STL).
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.stats import linregress


def analyze_degradation_with_stl(
    series: pd.Series,
    apply_log: bool = False,
    period: int = 365,
    robust: bool = True,
    anomaly_threshold: float = -3.0,
    min_consecutive_days: int = 2,
    return_components: bool = False
):
    """
    Analyze PV degradation using STL decomposition with robust anomaly detection.
    
    Parameters
    -----------
    series: pd.Series
        Time series data (date-indexed) for a single park
    apply_log: bool
        Whether to apply log transformation before STL (default: False)
    period: int
        Seasonal period for STL decomposition (default: 365)
    robust: bool
        Whether to use robust STL fitting (default: True)
    anomaly_threshold: float
        Z-score threshold for anomaly detection (default: -3.0)
    min_consecutive_days: int
        Minimum consecutive days for persistent anomalies (default: 2)
    return_components: bool
        Whether to return full decomposition components (default: False)
    
    Returns
    --------
    dict containing:
        - degradation_rate: Monthly degradation rate (% per month)
        - annual_degradation: Annual degradation rate (% per year)
        - trend_slope: Linear trend coefficient
        - anomaly_count: Total number of anomalous days
        - persistent_anomaly_count: Number of persistent anomalies
        - anomaly_flags: Boolean series marking anomalous days
        - residual_z_scores: Robust z-scores of residuals
        - monthly_medians: Monthly median values with trend
        - stl_result: STL decomposition object (if return_components=True)
    """
    # Remove NaN values and ensure series is sorted
    series_clean = series.dropna().sort_index()
    
    if len(series_clean) < period * 2:
        raise ValueError(f"Insufficient data: need at least {period * 2} observations, got {len(series_clean)}")
    
    # Apply log transformation if requested
    if apply_log:
        series_transformed = np.log(series_clean + 1e-6)
    else:
        series_transformed = series_clean.copy()
    
    # Perform STL decomposition
    stl = STL(series_transformed, period=period, robust=robust)
    stl_result = stl.fit()
    
    # Extract components
    trend = stl_result.trend
    seasonal = stl_result.seasonal
    residual = stl_result.resid
    
    # Compute robust z-scores using MAD (Median Absolute Deviation)
    median_resid = np.median(residual)
    mad = np.median(np.abs(residual - median_resid))
    
    # MAD-based robust z-score
    if mad > 0:
        residual_z_scores = (residual - median_resid) / (1.4826 * mad)
    else:
        residual_z_scores = pd.Series(0, index=residual.index)
    
    # Flag anomalies
    anomaly_flags = residual_z_scores < anomaly_threshold
    
    # Identify persistent anomalies (consecutive days)
    persistent_anomalies = pd.Series(False, index=anomaly_flags.index)
    
    if min_consecutive_days > 1:
        anomaly_groups = (anomaly_flags != anomaly_flags.shift()).cumsum()
        consecutive_counts = anomaly_flags.groupby(anomaly_groups).transform('sum')
        persistent_anomalies = anomaly_flags & (consecutive_counts >= min_consecutive_days)
    else:
        persistent_anomalies = anomaly_flags.copy()
    
    # Fit linear regression to trend
    days_numeric = (trend.index - trend.index[0]).days.values
    slope, intercept, r_value, p_value, std_err = linregress(days_numeric, trend.values)
    
    # Calculate monthly medians
    monthly_trend = trend.resample('MS').median()
    
    # Compute degradation rate
    mean_value = trend.mean()
    if apply_log:
        degradation_per_day = slope
        degradation_per_month = degradation_per_day * 30.44
        annual_degradation = degradation_per_day * 365.25
    else:
        if mean_value != 0:
            degradation_per_day = (slope / mean_value)
            degradation_per_month = degradation_per_day * 30.44 * 100
            annual_degradation = degradation_per_day * 365.25 * 100
        else:
            degradation_per_month = 0
            annual_degradation = 0
    
    # Prepare results
    results = {
        'degradation_rate': degradation_per_month,
        'annual_degradation': annual_degradation if not apply_log else annual_degradation * 100,
        'trend_slope': slope,
        'trend_intercept': intercept,
        'trend_r_squared': r_value**2,
        'anomaly_count': anomaly_flags.sum(),
        'persistent_anomaly_count': persistent_anomalies.sum(),
        'anomaly_percentage': (anomaly_flags.sum() / len(anomaly_flags)) * 100,
        'persistent_anomaly_percentage': (persistent_anomalies.sum() / len(persistent_anomalies)) * 100,
        'anomaly_flags': anomaly_flags,
        'persistent_anomaly_flags': persistent_anomalies,
        'residual_z_scores': residual_z_scores,
        'monthly_medians': monthly_trend,
        'mean_residual_mad': mad,
        'data_points': len(series_clean),
        'date_range': f"{series_clean.index.min().date()} to {series_clean.index.max().date()}",
        'parameters': {
            'apply_log': apply_log,
            'period': period,
            'robust': robust,
            'anomaly_threshold': anomaly_threshold,
            'min_consecutive_days': min_consecutive_days,
        }
    }
    
    if return_components:
        results['stl_result'] = stl_result
        results['trend'] = trend
        results['seasonal'] = seasonal
        results['residual'] = residual
    
    return results
