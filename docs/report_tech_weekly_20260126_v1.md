# Weekly PV Technical Report
**Report Date:** January 26, 2026
**Version:** v1

---

## Executive Summary

This weekly technical report summarizes operational performance across all PV parks, with an emphasis on time-series behavior, month-to-date performance indicators (PI), and revenue totals.

---

## 1. Daily Energy Time Series (All Parks)

![Daily Energy Time Series](..\plots\weekly_analysis\fig1_daily_energy_timeseries_20260126_v1.png)

**Notes:**
- Check for gaps or flatlines indicating telemetry or data acquisition issues.
- Compare inter-park variability to expected seasonal patterns.

---

## 2. Performance Index (PI) Heatmap — Full Period

![PI Heatmap - Full Period](..\plots\weekly_analysis\fig2_pi_heatmap_full_20260126_v1.png)

**Interpretation:** PI ≈ 1 implies measured energy aligns with PVGIS expectation. Sustained PI < 0.8 suggests underperformance; PI > 1.2 may indicate measurement anomalies.

---

## 3. PI Heatmap — January 2026 (Month-to-Date)

![PI Heatmap - MTD](..\plots\weekly_analysis\fig3_pi_heatmap_mtd_20260126_v1.png)

**Operational Checkpoints:**
- Investigate parks with sustained blue regions (PI < 0.8).
- Correlate anomalies with weather, outages, and maintenance logs.

---

## 4. Month-to-Date Revenue by Year — All Parks

Period: January 1–26

![Revenue MTD - All Parks](..\plots\weekly_analysis\fig4_revenue_mtd_all_parks_20260126_v1.png)

**Insights:**
- Highest MTD revenue: 2023 (400,635.07 EUR)
- Lowest MTD revenue: 2026 (185,865.26 EUR)
- Overall change: -53.61% from 2023 to 2026

---

## 5. Month-to-Date Revenue per Park by Year (Grid)

![MTD Revenue Grid](..\plots\weekly_analysis\fig5_revenue_mtd_grid_20260126_v1.png)

**Usage:** Highlights per-park month-to-date revenue against historical years. Use alongside PI heatmaps to differentiate revenue-impacting underperformance from data issues.

---

*Report generated on 2026-01-26 14:11:02*
