"""End-to-end per-league pipeline: extract Oracle -> enumerate esports -> join -> fetch frames.

Usage:  python3 pipeline.py <LEAGUE_NAME> [--workers 24] [--alias 'OracleName=EsportsName']

Writes per-league artifacts into autoresearch/poc/leagues/<LEAGUE>/:
  oracle_games.csv       - oracle side of join
  esports_games.csv      - esports-api side
  mapping.csv            - (oracle_gameid, esports_game_id, ...) pairs
  unmatched.csv          - oracle rows with no esports counterpart
  frames/*.ndjson        - per-minute window frames for each mapped game
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parents[1]
ORACLE_CSV = PROJECT / "data" / "2025_LoL_esports_match_data_from_OraclesElixir.csv"

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
ESPORTS_BASE = "https://esports-api.lolesports.com/persisted/gw"
FEED_URL = "https://feed.lolesports.com/livestats/v1/window/{gid}?startingTime={ts}"

COARSE_STEP_MIN = 10
COARSE_MAX_MIN = 180
MINUTES_TO_CAPTURE = range(0, 26)

DEFAULT_ALIASES = {
    "TEAMBDS": "SHIFTERS",
}


def snap_10s(t: dt.datetime) -> dt.datetime:
    return t.replace(second=(t.second // 10) * 10, microsecond=0)


def http_get(url: str, api_key: str | None = None):
    cmd = ["curl", "-sSfL", "--max-time", "15"]
    if api_key:
        cmd += ["-H", f"x-api-key: {api_key}"]
    cmd.append(url)
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return json.loads(raw) if raw.strip() else None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def norm(name: str, aliases: dict) -> str:
    n = name.strip().upper().replace(" ", "")
    return aliases.get(n, n)


# ---------- Oracle extraction ----------
def extract_oracle(league: str, out_path: Path) -> int:
    games = defaultdict(dict)
    with ORACLE_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["league"] != league or row["position"] != "team":
                continue
            gid = row["gameid"]
            g = games[gid]
            g["date"] = row["date"]
            g["game_num"] = row["game"]
            g["url"] = row["url"]
            side_key = "blue" if row["side"] == "Blue" else "red"
            g[side_key] = row["teamname"]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gameid", "date", "game_num", "blue", "red", "url"])
        for gid, g in sorted(games.items(), key=lambda kv: kv[1]["date"]):
            w.writerow([gid, g["date"], g["game_num"], g.get("blue", ""), g.get("red", ""), g["url"]])
    return len(games)


# ---------- Esports API enumeration ----------
def enumerate_esports(league_slug: str, out_path: Path) -> int:
    leagues = http_get(f"{ESPORTS_BASE}/getLeagues?hl=en-US", API_KEY)["data"]["leagues"]
    match = next((l for l in leagues if l["slug"].lower() == league_slug.lower()), None)
    if not match:
        print(f"  [enumerate] league slug {league_slug!r} not found")
        return 0
    tourneys = http_get(f"{ESPORTS_BASE}/getTournamentsForLeague?hl=en-US&leagueId={match['id']}", API_KEY)["data"]["leagues"][0]["tournaments"]
    t_2025 = [t for t in tourneys if "2025" in t["slug"]]
    rows = []
    for t in t_2025:
        events = http_get(f"{ESPORTS_BASE}/getCompletedEvents?hl=en-US&tournamentId={t['id']}", API_KEY)["data"]["schedule"]["events"]
        for ev in events:
            m = ev.get("match") or {}
            teams = m.get("teams") or []
            if len(teams) != 2:
                continue
            for idx, g in enumerate(ev.get("games") or [], start=1):
                rows.append({
                    "tournament": t["slug"],
                    "startTime": ev["startTime"],
                    "match_id": m.get("id", ""),
                    "esports_game_id": g["id"],
                    "game_num": idx,
                    "team_a_name": teams[0].get("name", ""),
                    "team_b_name": teams[1].get("name", ""),
                    "has_vods": 1 if g.get("vods") else 0,
                })
    with out_path.open("w", newline="") as f:
        if not rows:
            return 0
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return len(rows)


# ---------- Join ----------
def join_sides(oracle_path: Path, esports_path: Path, mapping_path: Path, unmatched_path: Path, aliases: dict) -> tuple[int, int]:
    oracle = list(csv.DictReader(oracle_path.open()))
    esports = list(csv.DictReader(esports_path.open()))
    idx = {}
    for r in esports:
        date_day = r["startTime"][:10]
        teams = frozenset([norm(r["team_a_name"], aliases), norm(r["team_b_name"], aliases)])
        key = (date_day, teams, int(r["game_num"]))
        idx.setdefault(key, []).append(r)
    matches, unmatched = [], []
    for r in oracle:
        date_day = r["date"][:10]
        teams = frozenset([norm(r["blue"], aliases), norm(r["red"], aliases)])
        key = (date_day, teams, int(r["game_num"]))
        hits = idx.get(key, [])
        if len(hits) == 1:
            h = hits[0]
            matches.append({
                "oracle_gameid": r["gameid"],
                "esports_game_id": h["esports_game_id"],
                "tournament": h["tournament"],
                "date": r["date"],
                "start_time": h["startTime"],
                "blue": r["blue"],
                "red": r["red"],
                "game_num": int(r["game_num"]),
            })
        else:
            unmatched.append({**r, "n_hits": len(hits)})
    if matches:
        with mapping_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
            w.writeheader(); w.writerows(matches)
    if unmatched:
        with unmatched_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(unmatched[0].keys()))
            w.writeheader(); w.writerows(unmatched)
    return len(matches), len(unmatched)


# ---------- Frame fetching ----------
def find_game_start(gid: str, scheduled: dt.datetime) -> dt.datetime | None:
    for offset in range(0, COARSE_MAX_MIN + 1, COARSE_STEP_MIN):
        t = snap_10s(scheduled + dt.timedelta(minutes=offset))
        d = http_get(FEED_URL.format(gid=gid, ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")))
        if not d:
            continue
        for f in (d.get("frames") or []):
            if f.get("gameState") == "in_game":
                return dt.datetime.fromisoformat(f["rfc460Timestamp"].replace("Z", "+00:00"))
    return None


def fetch_game(gid: str, scheduled_iso: str, out_dir: Path):
    out_file = out_dir / f"{gid}.ndjson"
    if out_file.exists() and out_file.stat().st_size > 200:
        return gid, "cached", 0
    scheduled = dt.datetime.fromisoformat(scheduled_iso.replace("Z", "+00:00"))
    start = find_game_start(gid, scheduled)
    if start is None:
        out_file.write_text("")
        return gid, "no_feed", 0
    n = 0
    with out_file.open("w") as fh:
        for m in MINUTES_TO_CAPTURE:
            t = snap_10s(start + dt.timedelta(minutes=m))
            d = http_get(FEED_URL.format(gid=gid, ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")))
            if not d:
                continue
            frames = d.get("frames") or []
            if not frames:
                continue
            f = frames[0]
            fh.write(json.dumps({
                "gid": gid, "minute": m, "ts": f.get("rfc460Timestamp"),
                "state": f.get("gameState"),
                "blue": f.get("blueTeam"), "red": f.get("redTeam"),
            }, separators=(",", ":")) + "\n")
            n += 1
            if f.get("gameState") == "finished":
                break
    return gid, "ok", n


def fetch_all(mapping_path: Path, out_dir: Path, workers: int, label: str):
    rows = list(csv.DictReader(mapping_path.open()))
    print(f"[{label}] games to fetch: {len(rows)}  workers: {workers}")
    done = status = {"ok": 0, "cached": 0, "no_feed": 0}
    done = 0
    status = {"ok": 0, "cached": 0, "no_feed": 0}
    total_frames = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_game, r["esports_game_id"], r["start_time"], out_dir): r for r in rows}
        for fut in as_completed(futures):
            gid, s, nf = fut.result()
            status[s] += 1
            total_frames += nf
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"[{label}] {done}/{len(rows)}  ok={status['ok']} cached={status['cached']} no_feed={status['no_feed']}  total_frames={total_frames}")
    print(f"[{label}] DONE  {status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("league", help="Oracle's league code (e.g. LEC, LCK, LCS)")
    ap.add_argument("--slug", help="lolesports league slug (defaults to lowercased league)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--alias", action="append", default=[], help="oracle=esports name alias (repeat)")
    args = ap.parse_args()

    league = args.league
    slug = args.slug or league.lower()
    aliases = dict(DEFAULT_ALIASES)
    for a in args.alias:
        k, v = a.split("=", 1)
        aliases[k.strip().upper().replace(" ", "")] = v.strip().upper().replace(" ", "")

    out_root = HERE / "leagues" / league
    out_root.mkdir(parents=True, exist_ok=True)
    frames_dir = PROJECT / "data" / "frames" / f"{league.lower()}_window"
    frames_dir.mkdir(parents=True, exist_ok=True)

    label = league
    print(f"[{label}] ==== pipeline start ====")

    oracle_path = out_root / "oracle_games.csv"
    esports_path = out_root / "esports_games.csv"
    mapping_path = out_root / "mapping.csv"
    unmatched_path = out_root / "unmatched.csv"

    n_oracle = extract_oracle(league, oracle_path)
    print(f"[{label}] oracle games: {n_oracle}")

    n_esports = enumerate_esports(slug, esports_path)
    print(f"[{label}] esports game rows: {n_esports}")
    if n_oracle == 0 or n_esports == 0:
        print(f"[{label}] aborting - no games on one side")
        return

    n_match, n_unmatched = join_sides(oracle_path, esports_path, mapping_path, unmatched_path, aliases)
    print(f"[{label}] matched: {n_match}/{n_oracle} ({100*n_match/n_oracle:.1f}%)  unmatched: {n_unmatched}")

    if n_match == 0:
        return

    fetch_all(mapping_path, frames_dir, args.workers, label)


if __name__ == "__main__":
    main()
