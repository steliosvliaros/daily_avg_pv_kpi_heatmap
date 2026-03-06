    def _sanitize_one(self, col_name: str) -> str:
        """
        Two-step sanitization:
        1. Extract park info and look up canonical park_id from metadata
        2. Sanitize signal name with unit
        3. Concatenate: park_id__capacity_kwp__signal_name__unit
        """
        m = PARK_HEADER_RE.match(col_name)
        if not m:
            before = col_name.replace('[', '').replace(']', '').strip()
            park_name_raw = before
            rest = ""
        else:
            park_name_raw = m.group("park").strip()
            rest = m.group("rest").strip()

        # Step 1: Try to get canonical park_id from metadata
        kwp = _parse_capacity_to_kwp(park_name_raw)
        park_id = self._lookup_park_id_from_metadata(park_name_raw, kwp)
        
        # If park_id found in metadata, strip capacity from it
        # Otherwise, fall back to sanitizing park name
        if park_id is not None:
            # park_id came from metadata - strip capacity from it too
            park_id = CAPACITY_RE.sub("", park_id).strip()
            park_name_clean = park_id.replace("_", " ")  # For token matching in signal dedup
        else:
            # Get capacity if not already parsed
            if kwp is None:
                if park_name_raw in self._park_kwp_cache:
                    kwp = self._park_kwp_cache[park_name_raw]
                elif not _is_timestamp_column(col_name):
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

            # Build park_id from sanitized name (ONLY, without capacity)
            # Capacity goes in the measurement/signal part, not the park name
            park_name_clean = CAPACITY_RE.sub("", park_name_raw).strip()
            park_name_clean = _dedupe_tokens_preserve_order(park_name_clean)
            park_snake = _snake_case_sql(park_name_clean, prefix_if_starts_digit="p_")
            
            # park_id is ONLY the park name (no capacity)
            park_id = park_snake

        # Step 2: Sanitize signal name with unit
        rest_clean = _dedupe_tokens_preserve_order(rest)
        
        # Remove park name tokens from signal to avoid redundancy
        # e.g., if park is "Spes Solaris" and signal contains "Spes Solaris Average", 
        # remove those tokens to get just "Average"
        park_tokens = set(park_name_clean.lower().split())
        signal_tokens = rest_clean.lower().split()
        filtered_tokens = [t for t in signal_tokens if t.lower() not in park_tokens]
        if filtered_tokens:
            rest_clean = " ".join(filtered_tokens)
        
        # Extract unit from signal
        unit = ""
        m_unit = TRAILING_UNIT_PARENS_RE.search(rest_clean)
        if m_unit:
            unit = m_unit.group("unit").strip()
            rest_clean = TRAILING_UNIT_PARENS_RE.sub("", rest_clean).strip()

        rest_snake = _snake_case_sql(rest_clean, prefix_if_starts_digit="m_")
        
        # Normalize unit
        unit_lower = unit.lower().strip()
        if not unit_lower or unit_lower == "":
            unit_snake = ""
        elif unit_lower in ("%", "percent", "percentage", "pct"):
            unit_snake = "pct"
        else:
            unit_snake = _snake_case_sql(unit, prefix_if_starts_digit="u_")
            if not unit_snake or unit_snake == "u_":
                unit_snake = ""

        # Step 3: Concatenate park_id__capacity_kwp__signal_name__unit
        joiner = "__"
        parts: List[str] = []
        
        if park_id:
            parts.append(park_id)
        # Add capacity as part of the identifier
        if kwp is not None:
            parts.append(_format_kwp_snake(kwp))
        if rest_snake:
            parts.append(rest_snake)
        if unit_snake:
            parts.append(unit_snake)

        identifier = joiner.join(parts) if parts else "col"
        identifier = self._make_unique(identifier)
        return identifier
