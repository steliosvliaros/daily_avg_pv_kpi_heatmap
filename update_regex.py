import re

with open('src/scada_column_sanitizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the CAPACITY_RE line
old_pattern = r'''CAPACITY_RE = re.compile(r"(?i)(?P<num>\d+(?:[.,_]\d+)?)\s*(?P<unit>k\s*w\s*p|kwp|kw|mw|mwp)")'''
new_pattern = r'''CAPACITY_RE = re.compile(r"(?i)_?\d+(?:[.,_]\d+)?\s*_?\s*(?:k\s*w\s*p|kwp|kw|mw|mwp)_?")'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open('src/scada_column_sanitizer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ Updated CAPACITY_RE')
else:
    print('✗ Pattern not found')
