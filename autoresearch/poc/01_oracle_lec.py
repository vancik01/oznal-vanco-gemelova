"""Extract LEC 2025 games from Oracle's Elixir CSV (Oracle side of the join)."""
import csv
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
OUT_PATH = Path(__file__).parent / "oracle_lec.csv"

want_cols = ["gameid", "url", "league", "date", "game", "side", "position", "teamname"]

games = defaultdict(dict)  # gameid -> {"date":..., "game":..., "url":..., "blue":..., "red":...}

with CSV_PATH.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["league"] != "LEC":
            continue
        if row["position"] != "team":
            continue
        gid = row["gameid"]
        g = games[gid]
        g["date"] = row["date"]
        g["game_num"] = row["game"]
        g["url"] = row["url"]
        g["league"] = row["league"]
        side_key = "blue" if row["side"] == "Blue" else "red"
        g[side_key] = row["teamname"]

with OUT_PATH.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gameid", "date", "game_num", "blue", "red", "url"])
    for gid, g in sorted(games.items(), key=lambda kv: kv[1]["date"]):
        w.writerow([gid, g["date"], g["game_num"], g.get("blue", ""), g.get("red", ""), g["url"]])

print(f"LEC 2025 games: {len(games)}")
print(f"with URL populated: {sum(1 for g in games.values() if g['url'])}")
print(f"date range: {min(g['date'] for g in games.values())} ... {max(g['date'] for g in games.values())}")
teams = {t for g in games.values() for t in (g.get("blue",""), g.get("red","")) if t}
print(f"unique teams: {len(teams)}")
print(f"sample teams: {sorted(teams)[:10]}")
