import re

COL_RE = re.compile(r"^(?P<park>.+?)__(?P<cap>\d+(?:\.\d+)?kwp)__(?P<meas>.+?)(?:__(?P<unit>u_[a-z0-9_]+))?$")

test_columns = [
    "p_4e_ener_liko__176kwp__pcc_curr_thd_r_of_neut__u_pct",
    "p_4e_ener_lexa__4472kwp__pcc_acti_ener_expo__u_kwh",
    "frag_util__4866kwp__pcc_acti_ener_expo__u_kwh",
]

for col in test_columns:
    m = COL_RE.match(col)
    if m:
        print(f"\n✅ {col}")
        print(f"   park:     {m.group('park')}")
        print(f"   cap:      {m.group('cap')}")
        print(f"   meas:     {m.group('meas')}")
        print(f"   unit:     {m.group('unit')}")
    else:
        print(f"\n❌ NO MATCH: {col}")
