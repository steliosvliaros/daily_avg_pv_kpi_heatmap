import json
from pathlib import Path
path = Path("notebooks/02_SCH_pv-report_PREPROD.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "# Calculate total annual production and revenue (full year, not MTD)" in src:
        cell["source"] = [
            "# Calculate total annual production and revenue (full year, not MTD)\n",
            "\n",
            "from src.metrics_calculator import calculate_revenue_from_energy, load_park_prices\n",
            "\n",
            "annual_energy = pr_result.measured.groupby(pr_result.measured.index.year).sum(numeric_only=True)\n",
            "prices = load_park_prices(config.PARK_METADATA_CSV)\n",
            "annual_revenue = calculate_revenue_from_energy(\n",
            "    annual_energy,\n",
            "    price_per_kwh=prices,\n",
            "    metadata_path=config.PARK_METADATA_CSV,\n",
            ")\n",
            "\n",
            "annual_production_total = annual_energy.sum(axis=1, numeric_only=True)\n",
            "annual_revenue_total = annual_revenue.sum(axis=1, numeric_only=True)\n",
            "\n",
            "print(\"Total annual production (kWh/year):\")\n",
            "print(annual_production_total)\n",
            "print(\"\\nTotal annual revenue (EUR/year):\")\n",
            "print(annual_revenue_total)\n",
            "\n",
            "yearly_totals = annual_production_total\n",
        ]
        print("fixed cell", i)
        break
else:
    print("target cell not found")
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
