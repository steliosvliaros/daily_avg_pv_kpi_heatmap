import json
from pathlib import Path

runlog_dir = Path('bronze/_ops')
runs = sorted(runlog_dir.glob('run_*.json'))
if runs:
    latest = runs[-1]
    print(f"Latest run: {latest.name}")
    data = json.loads(latest.read_text())
    print(f"Run ID: {data.get('run_id')}")
    files = data.get('files_written', [])
    print(f"Files written: {len(files)}")
    for f in files[:5]:
        print(f"  {f}")
        p = Path(f)
        print(f"    exists: {p.exists()}")
