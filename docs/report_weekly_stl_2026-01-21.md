# STL Decomposition & Degradation Analysis Report
**Report Date:** January 21, 2026

**Park:** [4E_Energeiaki_176_kWp_Likovouni_PCC_PCC_active_energy_export_(kWh)]
**Park Label:** 4E_Energeiaki_176_kWp_Likovouni_PCC_PCC_active_energy_export_(kWh)
**Capacity:** 176 kWp

---

## Executive Summary

This report presents a comprehensive time series decomposition and degradation analysis of 4E_Energeiaki_176_kWp_Likovouni_PCC_PCC_active_energy_export_(kWh) using Seasonal-Trend decomposition using LOESS (STL) with robust anomaly detection based on MAD (Median Absolute Deviation) statistics.

## Key Findings

### Data Overview
- **Data Range:** 2015-01-01 to 2025-04-09
- **Total Data Points:** 3,595 days
- **Analysis Period:** 9.8 years

### Degradation Analysis
- **Monthly Degradation Rate:** +nan% per month 📈
- **Annual Degradation Rate:** -0.5094% per year
- **Trend R² (Goodness of Fit):** 0.0778
- **Trend Slope:** -1.137678e-02 kWh/day

**Interpretation:** ⚠️ **Moderate degradation** - Monitor closely

### Anomaly Detection
- **Total Anomalies Detected:** 469 days (13.05%)
- **Persistent Anomalies:** 252 days (7.01%)
- **Anomaly Threshold:** Z-score < -3.0
- **Persistence Criterion:** ≥ 2 consecutive days
- **Residual MAD:** 68.98 kWh

## STL Decomposition Visualization

![STL Decomposition - 4E_Energeiaki_176_kWp_Likovouni_PCC_PCC_active_energy_export_(kWh)](..\plots\stl_analysis\stl_decomposition_4E_Energeiaki_176_kWp_Likovouni_PCC_PCC_active_energy_export__kWh.png)

### Plot Components

1. **Observed:** Original time series data showing daily energy generation
2. **Trend:** Long-term trend component with linear regression fit (red dashed line)
3. **Seasonal:** Periodic seasonal pattern (365-day period)
4. **Residual:** Remaining variation after removing trend and seasonality
   - 🟠 Orange dots: Individual anomalies (Z-score < -3)
   - 🔴 Red X markers: Persistent anomalies (≥2 consecutive days)
5. **Robust Z-scores:** MAD-based standardized residuals with anomaly threshold

## Methodology

### STL Decomposition
- **Method:** Seasonal-Trend decomposition using LOESS
- **Period:** 365 days (annual seasonality)
- **Robust Fitting:** Enabled (resistant to outliers)
- **Log Transformation:** Not applied

### Anomaly Detection
- **Robust Z-Score Calculation:**
  - Based on Median Absolute Deviation (MAD)
  - Formula: `Z = (residual - median) / (1.4826 × MAD)`
  - Threshold: Z < -3.0 (≈99.7% confidence for normal distribution)
- **Persistence Filter:**
  - Flags clusters of consecutive anomalous days
  - Helps distinguish systematic issues from random fluctuations

### Degradation Calculation
- **Linear Regression:** Fitted to trend component
- **Monthly Medians:** Aggregated from daily trend values
- **Rate Computation:** Relative change per unit time
  - Daily slope converted to monthly/annual percentages
  - Normalized by mean trend value

## Recommendations

### Anomaly Investigation
1. **Review Persistent Anomalies:** Investigate the causes of multi-day performance drops
2. **Correlate with Maintenance Records:** Check if anomalies align with maintenance events
3. **Weather Correlation:** Verify if anomalies coincide with extreme weather events
4. **Equipment Inspection:** Consider on-site inspection for persistent issues

### Degradation Management
1. **Monitor Trend:** Track degradation rate over time to detect acceleration
2. **Compare with Specifications:** Verify if degradation is within warranty limits
3. **Predictive Maintenance:** Plan interventions based on degradation trajectory
4. **Financial Impact:** Update revenue projections to account for degradation

### Data Quality
1. **Fill Data Gaps:** Address any missing data periods to improve analysis
2. **Sensor Calibration:** Verify measurement accuracy, especially if anomalies are frequent
3. **Regular Monitoring:** Repeat this analysis quarterly to track changes

## Technical Details

### Model Parameters
- Seasonal Period: 365 days
- Robust Fitting: Enabled
- Anomaly Threshold: -3.0 (Z-score)
- Minimum Consecutive Days: 2
- Log Transformation: Not applied

### Statistical Metrics
- **Trend Slope:** -1.137678e-02 kWh/day
- **Trend Intercept:** 837.02 kWh
- **R² (Trend Fit):** 0.0778
- **Median Residual:** 0 (by definition)
- **MAD (Residuals):** 68.98 kWh

---

### References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
- Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology*, 49(4), 764-766.

*Report generated on 2026-01-21 17:00:27*
