"""Dry-run join only (no fetch) across candidate leagues to see mapping coverage."""
import sys, importlib.util
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("p", HERE / "pipeline.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

CANDIDATES = [
    # (oracle_league_code, lolesports_slug)
    ("LEC", "lec"),
    ("LCK", "lck"),
    ("LCS", "lcs"),
    ("LPL", "lpl"),
    ("LTA N", "lta_n"),
    ("LTA S", "lta_s"),
    ("MSI", "msi"),
    ("WLDs", "worlds"),
    ("Worlds", "worlds"),
    ("First Stand", "first_stand"),
    ("LFL", "lfl"),
    ("LJL", "ljl_challengers"),
    ("EM", "emea_masters"),
]

out = HERE / "leagues_tmp"
out.mkdir(exist_ok=True)

for league, slug in CANDIDATES:
    out_dir = out / league.replace(" ", "_")
    out_dir.mkdir(exist_ok=True)
    n_oracle = p.extract_oracle(league, out_dir / "oracle.csv")
    if n_oracle == 0:
        print(f"{league:12s}  skip (no oracle rows)")
        continue
    n_esp = p.enumerate_esports(slug, out_dir / "esports.csv")
    if n_esp == 0:
        print(f"{league:12s}  oracle={n_oracle:4d}  esports=0 (slug {slug!r} not found or empty)")
        continue
    n_match, n_unmatched = p.join_sides(
        out_dir / "oracle.csv", out_dir / "esports.csv",
        out_dir / "mapping.csv", out_dir / "unmatched.csv",
        p.DEFAULT_ALIASES,
    )
    pct = 100 * n_match / n_oracle
    print(f"{league:12s}  oracle={n_oracle:4d}  esports={n_esp:4d}  matched={n_match:4d} ({pct:5.1f}%)  unmatched={n_unmatched}")
