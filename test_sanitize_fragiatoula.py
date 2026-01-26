#!/usr/bin/env python3
"""Test the Fragiatoula sanitization fix."""

from pathlib import Path
from src.scada_column_sanitizer import ScadaColumnSanitizer, SanitizeConfig

def test_fragiatoula_sanitization():
    """
    Test that park name tokens (including capacity) are properly stripped from signal names.
    
    Original:  [Fragiatoula_Utilitas_4866kWp] Fragiatoula_Utilitas_4866kWp Average Irradiance (W*m^-2)
    Expected:  fragiatoula_utilitas__4866_kwp__average_irradiance__w_m_2
    """
    
    # Setup config to disable interactive prompts
    config = SanitizeConfig(
        prompt_missing_capacity=False,
        park_metadata_path=Path("mappings/park_metadata.csv")
    )
    
    sanitizer = ScadaColumnSanitizer(config=config)
    
    test_column = "[Fragiatoula_Utilitas_4866kWp] Fragiatoula_Utilitas_4866kWp Average Irradiance (W*m^-2)"
    expected = "fragiatoula_utilitas__4866_kwp__average_irradiance__w_m_2"
    
    result = sanitizer._sanitize_one(test_column)
    
    print(f"Original:  {test_column}")
    print(f"Expected:  {expected}")
    print(f"Result:    {result}")
    print()
    
    if result == expected:
        print("✅ TEST PASSED: Sanitization is correct!")
        return True
    else:
        print("❌ TEST FAILED: Sanitization does not match expected output")
        print()
        print("Analysis:")
        print(f"  - Park name should be stripped: 'Fragiatoula_Utilitas_4866kWp'")
        print(f"  - Capacity should be extracted: '4866_kwp'")
        print(f"  - Signal should be: 'average_irradiance'")
        print(f"  - Unit should be: 'w_m_2'")
        return False

if __name__ == "__main__":
    import sys
    success = test_fragiatoula_sanitization()
    sys.exit(0 if success else 1)
