"""Join Oracle LEC rows to esports games on (date_day, team_set, game_num).

Keying strategy:
  - date_day: YYYY-MM-DD portion of the datetime (both sides are UTC)
  - team_set: frozenset of normalized team names (order-independent, since Oracle splits
              into blue/red but esports API doesn't encode side in getCompletedEvents)
  - game_num: game number within the series (1/2/3/4/5)

Normalization: uppercase + strip + handle known alias ("Rogue" <-> "Shifters" rebrand
  if it turns out to matter, "GiantX" <-> "GIANTX").
"""
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ORACLE = HERE / "oracle_lec.csv"
ESPORTS = HERE / "esports_lec.csv"
MAPPING_OUT = HERE / "gameid_mapping_lec.csv"
UNMATCHED_OUT = HERE / "unmatched_oracle_lec.csv"


ALIASES = {
    # Oracle's Elixir kept legacy names; lolesports uses current branding.
    "TEAMBDS": "SHIFTERS",
}


def norm(name: str) -> str:
    n = name.strip().upper().replace(" ", "")
    return ALIASES.get(n, n)


def load_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


oracle = load_csv(ORACLE)
esports = load_csv(ESPORTS)

# Build index of esports games keyed by (date_day, team_set, game_num)
idx = {}
for r in esports:
    date_day = r["startTime"][:10]
    teams = frozenset([norm(r["team_a_name"]), norm(r["team_b_name"])])
    key = (date_day, teams, int(r["game_num"]))
    # Collisions possible if same two teams play twice on same day (rare but happens in BO5 days).
    # But game_num should disambiguate within a single BO; across multiple BOs on same day same teams won't happen.
    idx.setdefault(key, []).append(r)

matches = []
unmatched = []
for r in oracle:
    date_day = r["date"][:10]
    blue, red = norm(r["blue"]), norm(r["red"])
    teams = frozenset([blue, red])
    game_num = int(r["game_num"])
    key = (date_day, teams, game_num)
    hits = idx.get(key, [])
    if len(hits) == 1:
        h = hits[0]
        matches.append({
            "oracle_gameid": r["gameid"],
            "esports_game_id": h["esports_game_id"],
            "esports_match_id": h["match_id"],
            "tournament": h["tournament"],
            "date": r["date"],
            "start_time": h["startTime"],
            "blue": r["blue"],
            "red": r["red"],
            "game_num": game_num,
        })
    else:
        unmatched.append({**r, "n_hits": len(hits)})

with MAPPING_OUT.open("w", newline="") as f:
    if matches:
        w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
        w.writeheader()
        w.writerows(matches)

with UNMATCHED_OUT.open("w", newline="") as f:
    if unmatched:
        w = csv.DictWriter(f, fieldnames=list(unmatched[0].keys()))
        w.writeheader()
        w.writerows(unmatched)

print(f"Oracle LEC games:          {len(oracle)}")
print(f"esports LEC game rows:     {len(esports)}  (includes unplayed BO5 games)")
print(f"matched:                   {len(matches)}  ({100*len(matches)/len(oracle):.1f}%)")
print(f"unmatched:                 {len(unmatched)}")

if unmatched:
    print("\n--- unmatched samples ---")
    for r in unmatched[:10]:
        print(f"  {r['date'][:10]}  {r['blue']} vs {r['red']}  g{r['game_num']}  n_hits={r['n_hits']}  gameid={r['gameid']}")

# Sanity: check the feed on one mapped pair
if matches:
    print("\n--- verifying one mapped pair against the livestats feed ---")
    m = matches[0]
    date_day = m["start_time"][:10]
    starting = f"{date_day}T{m['start_time'][11:16]}:00.000Z"
    # bump start by +20min to be well inside the game
    import datetime as dt
    t = dt.datetime.fromisoformat(m["start_time"].replace("Z", "+00:00")) + dt.timedelta(minutes=25)
    starting = t.strftime("%Y-%m-%dT%H:%M:00.000Z")
    url = f"https://feed.lolesports.com/livestats/v1/window/{m['esports_game_id']}?startingTime={starting}"
    print(f"URL: {url}")
    out = subprocess.check_output(["curl", "-sSfL", url])
    d = json.loads(out)
    frames = d.get("frames", [])
    if frames:
        f0 = frames[-1]
        b, r = f0["blueTeam"], f0["redTeam"]
        print(f"  frames returned: {len(frames)}  patch: {d['gameMetadata']['patchVersion']}")
        print(f"  ts: {f0['rfc460Timestamp']}  state: {f0['gameState']}")
        print(f"  blue: {b['totalKills']}k / {b['totalGold']}g / dragons={b['dragons']}")
        print(f"  red:  {r['totalKills']}k / {r['totalGold']}g / dragons={r['dragons']}")
    else:
        print("  no frames returned - game may not have a feed")
