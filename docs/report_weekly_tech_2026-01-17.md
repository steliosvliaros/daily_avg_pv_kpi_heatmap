# Weekly PV Technical Report
**Report Date:** January 17, 2026

---

## Executive Summary

This weekly technical report summarizes operational performance across all PV parks, with an emphasis on time-series behavior, month-to-date performance indicators (PI), and energy totals.

---

## 1. Daily Energy Time Series (All Parks)

![Daily Energy Time Series](..\plots\daily_energy_timeseries_grid.png)

**Notes:**
- Check for gaps or flatlines indicating telemetry or data acquisition issues.
- Compare inter-park variability to expected seasonal patterns.

---

## 2. Performance Index (PI) Heatmap — Full Period

![PI Heatmap - Full Period](..\plots\heatmap_pi_full.png)

**Interpretation:** PI ≈ 1 implies measured energy aligns with PVGIS expectation. Sustained PI < 0.8 suggests underperformance; PI > 1.2 may indicate measurement anomalies.

---

## 3. PI Heatmap — January 2026 (Month-to-Date)

![PI Heatmap - MTD](..\plots\heatmap_pi_january2026.png)

**Operational Checkpoints:**
- Investigate parks with sustained blue regions (PI < 0.8).
- Correlate anomalies with weather, outages, and maintenance logs.

---

## 4. Month-to-Date Energy by Year — All Parks

Period: January 1–17

![Energy MTD - All Parks](..\plots\energy_mtd_all_parks.png)

**Insights:**
- Highest MTD energy: 2023 (1,543,956 kWh)
- Lowest MTD energy: 2021 (1,073,941 kWh)
- Overall change: +nan% from 2020 to 2025

---

## 5. Month-to-Date Revenue per Park by Year (Grid)

![MTD Revenue Grid](..\plots\revenue_mtd_grid.png)

**Usage:** Highlights per-park month-to-date revenue against historical years. Use alongside PI heatmaps to differentiate revenue-impacting underperformance from data issues.

---

*Report generated on 2026-01-17 14:29:08*
