import pandas as pd

metadata = pd.read_csv('mappings/park_metadata.csv')

# Active parks
active = metadata[metadata['status_effective'] == True]
print(f'Total active parks (status_effective=true): {len(active)}')
print('\nActive parks (park_id):')
for idx, row in active.iterrows():
    print(f"  {row['park_id']}")

# Check which ones from the test output
test_parks = [
    '4e_energeiaki_176_kwp_likovouni',
    '4e_energeiaki_4472_kwp_lexaina',
    '4e_energeiaki_805_kwp_darali',
    'hliatoras_474kwp_andravida',
    'ntarali_bonitas_0_45mw',
    'ntarali_concept_592kw',
    'ntarali_konenergy_590kw',
    'nycontec_993_kwp_giannopouleika',
    'palaionaziro_iraio',
    'solar_concept_276_kwp_likovouni',
    'solar_concept_3721_kwp_lexaina',
    'solar_datum_2910_kwp_lexaina',
    'solar_datum_864_kwp_darali',
    'solar_factory_494kwp_andravida',
    'spes_solaris_1527_kwp_darali',
    'spes_solaris_1986_kwp_lexaina',
    'spes_solaris_201_kwp_konizos',
    'spes_solaris_500_kwp_kavasila',
]

print(f'\nTest output showed {len(test_parks)} parks')
print(f'\nDifference: {len(active) - len(test_parks)} parks in metadata but not in test output')

# Find missing parks
active_ids = set(active['park_id'].values)
test_ids = set(test_parks)
missing = active_ids - test_ids

if missing:
    print(f'\nMissing parks:')
    for p in sorted(missing):
        row = metadata[metadata['park_id'] == p].iloc[0]
        print(f"  {p}")
        print(f"    park_name: {row['park_name']}")
        print(f"    status_effective: {row['status_effective']}")
