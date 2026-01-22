
from __future__ import annotations

import argparse
import hashlib
import re
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

CAPACITY_RE = re.compile(r"(?i)(?P<num>\d+(?:[.,_]\d+)?)\s*(?P<unit>k\s*w\s*p|kwp|kw|mw|mwp)")
PARK_HEADER_RE = re.compile(r"^\s*\[(?P<park>[^\]]+)\]\s*(?P<rest>.*)$")
TRAILING_UNIT_PARENS_RE = re.compile(r"\((?P<unit>[^()]*)\)\s*$")
TIMESTAMP_COLUMN_RE = re.compile(r"(?i)^(time|timestamp|date|datetime|created|updated|modified|instant)$")

def _is_timestamp_column(col_name: str) -> bool:
    """Check if column name looks like a timestamp/date column."""
    clean_name = col_name.replace('[', '').replace(']', '').strip()
    return bool(TIMESTAMP_COLUMN_RE.match(clean_name))

def _dedupe_tokens_preserve_order(text: str) -> str:
    tokens = text.split()
    seen = set()
    out: List[str] = []
    for t in tokens:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return " ".join(out)

def _snake_case_sql(text: str, *, prefix_if_starts_digit: str) -> str:
    s = text.strip()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    s = s.lower()
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = f"{prefix_if_starts_digit}{s}"
    return s

def _normalize_unit(unit: str) -> str:
    if not unit:
        return ""
    u = unit.strip().lower()
    u = re.sub(r"\s+", "", u)
    return u

def _parse_capacity_to_kwp(text: str) -> Optional[float]:
    m = CAPACITY_RE.search(text)
    if not m:
        return None
    num_str = m.group("num").replace(",", ".").replace("_", ".")
    unit_str = m.group("unit")
    try:
        val = float(num_str)
    except ValueError:
        return None
    unit_norm = _normalize_unit(unit_str)
    if unit_norm in ("mw", "mwp"):
        val *= 1000.0
    return val

def _format_kwp_snake(kwp: float) -> str:
    if abs(kwp - round(kwp)) < 1e-9:
        return f"{int(round(kwp))}kwp"
    else:
        s = f"{kwp:.3f}".rstrip("0").rstrip(".")
        return f"{s}kwp"

def _abbreviate_token(token: str, max_len: int = 4) -> str:
    """Abbreviate a single token to first max_len chars (e.g., 'energeiaki' → 'ener')."""
    return token[:max_len] if len(token) > max_len else token

def _abbreviate_snake_case(snake_str: str, max_len: int = 4) -> str:
    """Abbreviate each token in a snake_case string (e.g., 'pcc_active_energy_export' → 'pcc_act_ener_expo')."""
    if not snake_str:
        return ""
    tokens = snake_str.split("_")
    abbr_tokens = [_abbreviate_token(t, max_len) for t in tokens]
    return "_".join(abbr_tokens)

@dataclass
class SanitizeConfig:
    prompt_missing_capacity: bool = True
    default_capacity_kwp: Optional[float] = None

class ScadaColumnSanitizer:
    def __init__(self, config: SanitizeConfig = None):
        self.config = config or SanitizeConfig()
        self._seen_outputs: Dict[str, int] = {}
        self._park_kwp_cache: Dict[str, float] = {}

    def sanitize_columns(
        self,
        columns: Iterable[str],
        existing_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        sanitized: List[str] = []
        mapping: Dict[str, str] = {}
        existing_mapping = existing_mapping or {}
        for col in columns:
            if col in existing_mapping:
                s = existing_mapping[col]
            else:
                # Check if this is a timestamp column - standardize to "datetime"
                if _is_timestamp_column(col):
                    s = "datetime"
                else:
                    s = self._sanitize_one(col)
            sanitized.append(s)
            mapping[col] = s
        return sanitized, mapping

    def _sanitize_one(self, col_name: str) -> str:
        m = PARK_HEADER_RE.match(col_name)
        if not m:
            before = col_name.replace('[', '').replace(']', '').strip()
            park_name_raw = before
            rest = ""
        else:
            park_name_raw = m.group("park").strip()
            rest = m.group("rest").strip()

        kwp = _parse_capacity_to_kwp(park_name_raw)
        if kwp is None:
            if park_name_raw in self._park_kwp_cache:
                kwp = self._park_kwp_cache[park_name_raw]
            elif _is_timestamp_column(col_name):
                # Timestamp columns don't need capacity
                kwp = None
            else:
                # Missing capacity: prompt user or use default
                if self.config.prompt_missing_capacity:
                    while True:
                        user_input = input(
                            f"⚠️  No capacity found in '{park_name_raw}'. Enter capacity in kWp (e.g., 500, 4.5): "
                        ).strip()
                        if user_input:
                            try:
                                kwp = float(user_input.replace(",", "."))
                                if kwp <= 0:
                                    print("   ❌ Capacity must be positive. Try again.")
                                    continue
                                break
                            except ValueError:
                                print("   ❌ Invalid input. Please enter a number (e.g., 500 or 4.5).")
                                continue
                        else:
                            print("   ⚠️  Capacity is required. Please provide a value.")
                elif self.config.default_capacity_kwp is not None:
                    kwp = self.config.default_capacity_kwp
                # Cache result
                if kwp is not None:
                    self._park_kwp_cache[park_name_raw] = kwp

        if kwp is not None:
            park_name_clean = CAPACITY_RE.sub("", park_name_raw).strip()
        else:
            park_name_clean = park_name_raw.strip()

        park_name_clean = _dedupe_tokens_preserve_order(park_name_clean)
        rest_clean = _dedupe_tokens_preserve_order(rest)

        unit = ""
        m_unit = TRAILING_UNIT_PARENS_RE.search(rest_clean)
        if m_unit:
            unit = m_unit.group("unit").strip()
            rest_clean = TRAILING_UNIT_PARENS_RE.sub("", rest_clean).strip()

        park_snake = _snake_case_sql(park_name_clean, prefix_if_starts_digit="p_")
        rest_snake = _snake_case_sql(rest_clean, prefix_if_starts_digit="m_")
        
        # Handle special cases for units BEFORE converting to snake_case:
        # - If no unit found, default to "num" (numeric value)
        # - If unit is percentage-related, use "pct"
        unit_lower = unit.lower().strip()
        if not unit_lower or unit_lower == "":
            unit_snake = "num"
        elif unit_lower in ("%", "percent", "percentage", "pct"):
            unit_snake = "pct"
        else:
            # Convert other units to snake_case
            unit_snake = _snake_case_sql(unit, prefix_if_starts_digit="u_")
            # If it's empty after conversion, use "num"
            if not unit_snake or unit_snake == "u_":
                unit_snake = "num"

        # Abbreviate long tokens to reduce identifier length and minimize hashing
        # DON'T abbreviate park_snake if it contains capacity info - that will be replaced anyway
        park_snake = _abbreviate_snake_case(park_snake, max_len=4)
        rest_snake = _abbreviate_snake_case(rest_snake, max_len=4)
        # Note: unit_snake is already processed above

        # Build parts list in order, but we'll ensure unit token is preserved
        parts: List[str] = []
        if park_snake:
            parts.append(park_snake)
        if kwp is not None:
            # Format kwp token - this should NOT be abbreviated (e.g., keep "450kwp" not "450k")
            kwp_token = _format_kwp_snake(kwp)
            parts.append(kwp_token)
        if rest_snake:
            parts.append(rest_snake)
        # Always add unit token (now defaults to "num" if empty)
        unit_token = f"u_{unit_snake}" if not unit_snake.startswith("u_") else unit_snake
        parts.append(unit_token)

        joiner = "__"

        def _shorten_with_hash(text: str, max_len: int) -> str:
            if max_len <= 0:
                return hashlib.md5(text.encode("utf-8")).hexdigest()[:max(1, min(10, len(text)))]
            if len(text) <= max_len:
                return text
            h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
            # Keep as much prefix as possible, add '_' + hash
            keep = max_len - (len(h) + 1)
            if keep <= 0:
                return h[:max_len]
            return f"{text[:keep]}_{h}"

        # Compose identifier and enforce Postgres 60-char limit ALWAYS preserving unit token
        identifier = joiner.join(parts) if parts else "col"
        max_len = 60  # PostgreSQL identifier limit
        
        if len(identifier) > max_len:
            # Strategy: preserve unit_token (always present now) and kwp, shorten rest_snake first, then park_snake
            # Calculate fixed length (kwp token + unit token)
            fixed_parts = []
            if kwp is not None:
                fixed_parts.append(_format_kwp_snake(kwp))
            # unit_token is always present now
            fixed_parts.append(unit_token)
            
            # Count joiners: we'll have (park + kwp + rest + unit) parts
            num_total_parts = sum([1 if park_snake else 0, 1 if kwp is not None else 0, 
                                   1 if rest_snake else 0, 1])  # unit always present
            joiner_len = len(joiner) * (num_total_parts - 1)
            fixed_len = sum(len(p) for p in fixed_parts)
            
            # Available space for park and rest
            available = max_len - fixed_len - joiner_len
            
            # Distribute space: try to keep reasonable lengths for park and rest
            # Priority: rest_snake gets more space (it's the measurement), park gets shortened first
            if park_snake and rest_snake:
                # Split available space: give more to rest
                rest_target = int(available * 0.6)
                park_target = available - rest_target
                
                if rest_target > len(rest_snake):
                    # rest doesn't need shortening
                    park_target = available - len(rest_snake)
                    rest_snake_short = rest_snake
                else:
                    rest_snake_short = _shorten_with_hash(rest_snake, max(8, rest_target))
                
                if park_target > len(park_snake):
                    park_snake_short = park_snake
                else:
                    park_snake_short = _shorten_with_hash(park_snake, max(8, park_target))
                
                # Rebuild parts with shortened versions
                parts = []
                parts.append(park_snake_short)
                if kwp is not None:
                    parts.append(_format_kwp_snake(kwp))
                parts.append(rest_snake_short)
                parts.append(unit_token)  # always present
                identifier = joiner.join(parts)
                
            elif rest_snake and not park_snake:
                # Only rest needs shortening
                rest_target = available
                if rest_target < len(rest_snake):
                    rest_snake_short = _shorten_with_hash(rest_snake, max(8, rest_target))
                else:
                    rest_snake_short = rest_snake
                parts = []
                if kwp is not None:
                    parts.append(_format_kwp_snake(kwp))
                parts.append(rest_snake_short)
                parts.append(unit_token)  # always present
                identifier = joiner.join(parts)
                
            elif park_snake and not rest_snake:
                # Only park needs shortening
                park_target = available
                if park_target < len(park_snake):
                    park_snake_short = _shorten_with_hash(park_snake, max(8, park_target))
                else:
                    park_snake_short = park_snake
                parts = []
                parts.append(park_snake_short)
                if kwp is not None:
                    parts.append(_format_kwp_snake(kwp))
                parts.append(unit_token)  # always present
                identifier = joiner.join(parts)

        identifier = self._make_unique(identifier)
        return identifier

    def _make_unique(self, name: str) -> str:
        if name not in self._seen_outputs:
            self._seen_outputs[name] = 1
            return name
        else:
            count = self._seen_outputs[name]
            self._seen_outputs[name] += 1
            return f"{name}_{count}"

    def load_mapping_csv(self, mapping_path: str | Path) -> Dict[str, str]:
        """Load an existing mapping CSV (original,sanitized) into a dict."""
        mapping: Dict[str, str] = {}
        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            return mapping
        with open(mapping_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            has_header = False
            if header and len(header) >= 2:
                # Detect header by name
                if header[0].lower() == "original" and header[1].lower() == "sanitized":
                    has_header = True
            if not has_header and header:
                # header is actually data
                if len(header) >= 2:
                    mapping[header[0]] = header[1]
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0]] = row[1]
        return mapping

    def save_outputs(
        self,
        output_dir: str | Path,
        sanitized: List[str],
        mapping: Dict[str, str],
        prompt_replace: bool = True,
    ) -> None:
        """
        Save sanitized columns and mapping to CSV files with replace/append prompt.
        
        Parameters
        ----------
        output_dir : str | Path
            Output directory for CSV files
        sanitized : List[str]
            List of sanitized column names
        mapping : Dict[str, str]
            Mapping of original to sanitized names
        prompt_replace : bool
            If True, prompt user when files exist (default: True).
            If False, always replace existing files.
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cols_file = output_dir / "park_columns.csv"
        mapping_file = output_dir / "park_power_mapping.csv"
        
        # Check if files exist and prompt user
        cols_exists = cols_file.exists()
        mapping_exists = mapping_file.exists()
        
        if (cols_exists or mapping_exists) and prompt_replace:
            print(f"\n⚠️  Output files already exist:")
            if cols_exists:
                print(f"   - {cols_file}")
            if mapping_exists:
                print(f"   - {mapping_file}")
            
            while True:
                choice = input("\n   ➜ Replace (r) or Append (a)? [r/a]: ").strip().lower()
                if choice in ("r", "replace"):
                    mode = "replace"
                    break
                elif choice in ("a", "append"):
                    mode = "append"
                    break
                else:
                    print("   ❌ Invalid choice. Please enter 'r' (replace) or 'a' (append).")
        else:
            mode = "replace"
        
        # Save park_columns.csv
        if mode == "replace":
            with open(cols_file, "w", encoding="utf-8") as f:
                for c in sanitized:
                    f.write(c + "\n")
            print(f"✅ {cols_file.name} (replaced)")
        else:
            existing_cols: set[str] = set()
            if cols_file.exists():
                with open(cols_file, "r", encoding="utf-8") as f:
                    existing_cols = {line.strip() for line in f if line.strip()}
            to_append_cols = [c for c in sanitized if c not in existing_cols]
            if to_append_cols:
                with open(cols_file, "a", encoding="utf-8") as f:
                    for c in to_append_cols:
                        f.write(c + "\n")
                print(f"✅ {cols_file.name} (appended {len(to_append_cols)} new)")
            else:
                print(f"ℹ️  {cols_file.name} already up to date; nothing appended")
        
        # Save park_power_mapping.csv
        if mode == "replace":
            with open(mapping_file, "w", encoding="utf-8") as f:
                f.write("original,sanitized\n")
                for o, s in mapping.items():
                    o2 = o.replace('"', '""')
                    s2 = s.replace('"', '""')
                    f.write(f'"{o2}","{s2}"\n')
            print(f"✅ {mapping_file.name} (replaced)")
        else:
            existing_sanitized: set[str] = set()
            if mapping_file.exists():
                with open(mapping_file, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header_skipped = False
                    for row in reader:
                        if not header_skipped:
                            header_skipped = True
                            # if header missing sanitized column name, skip gracefully
                            continue
                        if len(row) >= 2:
                            existing_sanitized.add(row[1])
            to_append_rows = [(o, s) for o, s in mapping.items() if s not in existing_sanitized]
            if to_append_rows:
                with open(mapping_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    for o, s in to_append_rows:
                        writer.writerow([o, s])
                print(f"✅ {mapping_file.name} (appended {len(to_append_rows)} new)")
            else:
                print(f"ℹ️  {mapping_file.name} already up to date; nothing appended")


def read_columns_only(path: str, sheet_name: Optional[str] = None, sep: Optional[str] = None, encoding: Optional[str] = None) -> List[str]:
    path_lower = path.lower()
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        if sheet_name is None:
            xl = pd.ExcelFile(path)
            first_sheet = xl.sheet_names[0]
            df = xl.parse(first_sheet, nrows=0)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name, nrows=0)
        # If sheet_name is None and multiple sheets exist, default opens first sheet
    else:
        kwargs = {"nrows": 0}
        if sep is not None:
            kwargs["sep"] = sep
        if encoding is not None:
            kwargs["encoding"] = encoding
        df = pd.read_csv(path, **kwargs)
    return df.columns.tolist()


def main():
    ap = argparse.ArgumentParser(description="SCADA column name sanitizer (SQL-safe)")
    ap.add_argument("--path", required=True, help="Path to Excel or CSV file")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    ap.add_argument("--sep", default=None, help="CSV delimiter (optional, default ',')")
    ap.add_argument("--encoding", default=None, help="CSV encoding (optional)")
    ap.add_argument("--out", default=None, help="Optional path to save sanitized names (one per line)")
    ap.add_argument("--map_out", default=None, help="Optional path to save mapping CSV: original,sanitized")
    ap.add_argument("--no_prompt", action="store_true", help="Disable prompting for missing capacities")
    ap.add_argument("--default_kwp", default=None, help="Default kWp if prompting disabled (float)")
    args = ap.parse_args()

    cols = read_columns_only(args.path, sheet_name=args.sheet, sep=args.sep, encoding=args.encoding)

    cfg = SanitizeConfig(
        prompt_missing_capacity=not args.no_prompt,
        default_capacity_kwp=float(args.default_kwp) if args.default_kwp is not None else None,
    )
    sanitizer = ScadaColumnSanitizer(config=cfg)

    sanitized, mapping = sanitizer.sanitize_columns(cols)

    print(sanitized)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for c in sanitized:
                f.write(c + "\\n")

    if args.map_out:
        with open(args.map_out, "w", encoding="utf-8") as f:
            f.write("original,sanitized\\n")
            for o, s in mapping.items():
                o2 = o.replace('"', '""')
                s2 = s.replace('"', '""')
                f.write(f'"{o2}","{s2}"\\n')

if __name__ == "__main__":
    main()
