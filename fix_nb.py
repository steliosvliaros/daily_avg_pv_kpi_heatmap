import json
from pathlib import Path
path = Path('notebooks/02_SCH_pv-report_PREPROD.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'Create combined annual production + revenue visualization' in src:
        cell['source'] = [
            '# Annual production + revenue chart for all parks\n',
            'from src import visualizations\n',
            'reload_modules(visualizations)\n',
            '\n',
            'annual_energy = annual_mtd_energy(pr_result.measured, agg="sum", per_park=True)\n',
            'annual_revenue = annual_mtd_revenue(pr_result.measured, metadata_path=config.PARK_METADATA_CSV, agg="sum", aggregate_parks=False)\n',
            '\n',
            'annual_production_total = annual_energy.sum(axis=1, numeric_only=True) if hasattr(annual_energy, "sum") else annual_energy\n',
            'annual_revenue_total = annual_revenue.sum(axis=1, numeric_only=True) if hasattr(annual_revenue, "sum") else annual_revenue\n',
            '\n',
            'print("Annual production totals (kWh/year):")\n',
            'print(annual_production_total)\n',
            'print("\\nAnnual revenue totals (EUR/year):")\n',
            'print(annual_revenue_total)\n',
            '\n',
            'fig_prod_rev, saved_prod_rev_path = visualizations.plot_annual_production_and_revenue(\n',
            '    production_series=annual_production_total,\n',
            '    revenue_series=annual_revenue_total,\n',
            '    title="Annual Production & Revenue by Year — All Parks",\n',
            '    production_unit="kWh",\n',
            '    currency="EUR",\n',
            '    config=config,\n',
            '    plot_name="annual_production_revenue_by_year_all_parks",\n',
            '    save=False,\n',
            ')\n',
            '\n',
            'print("\\n✅ This chart uses annual totals, not MTD totals.")\n',
        ]
        break
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('done')
