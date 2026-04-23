"""Enumerate LEC 2025 games from lolesports API (esports side of the join)."""
import csv
import json
import subprocess
from pathlib import Path

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
BASE = "https://esports-api.lolesports.com/persisted/gw"
OUT_PATH = Path(__file__).parent / "esports_lec.csv"


def get(url):
    out = subprocess.check_output(["curl", "-sSfL", "-H", f"x-api-key: {API_KEY}", url])
    return json.loads(out)


leagues = get(f"{BASE}/getLeagues?hl=en-US")["data"]["leagues"]
lec = next(l for l in leagues if l["slug"] == "lec")
print(f"LEC leagueId: {lec['id']}")

tourneys = get(f"{BASE}/getTournamentsForLeague?hl=en-US&leagueId={lec['id']}")["data"]["leagues"][0]["tournaments"]
lec_2025 = [t for t in tourneys if "2025" in t["slug"]]
print(f"LEC 2025 tournaments: {[t['slug'] for t in lec_2025]}")

rows = []
for t in lec_2025:
    events = get(f"{BASE}/getCompletedEvents?hl=en-US&tournamentId={t['id']}")["data"]["schedule"]["events"]
    print(f"  {t['slug']}: {len(events)} events")
    for ev in events:
        if not ev.get("match"):
            continue
        teams = ev["match"].get("teams", [])
        if len(teams) != 2:
            continue
        t0, t1 = teams
        for idx, g in enumerate(ev.get("games") or [], start=1):
            rows.append({
                "tournament": t["slug"],
                "startTime": ev["startTime"],
                "match_id": ev["match"]["id"],
                "esports_game_id": g["id"],
                "game_num": idx,
                "team_a_code": t0.get("code", ""),
                "team_a_name": t0.get("name", ""),
                "team_b_code": t1.get("code", ""),
                "team_b_name": t1.get("name", ""),
                "state": g.get("state", ""),
                "has_vods": 1 if g.get("vods") else 0,
            })

with OUT_PATH.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\ntotal esports game rows (incl. unplayed games in BO5s): {len(rows)}")
played = [r for r in rows if r["state"] not in ("unneeded", "unstarted")]
print(f"rows with state played: {len(played)}")
print(f"unique states: {sorted(set(r['state'] for r in rows))}")
print(f"unique team names: {sorted(set(r['team_a_name'] for r in rows) | set(r['team_b_name'] for r in rows))}")
