"""Fetch per-minute frames for a league using its auto_coverage mapping.csv.

Usage:  python3 fetch_from_mapping.py <LEAGUE_FOLDER> [--workers 24]
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parents[1]
AUTO_DIR = HERE / "auto_coverage"

FEED_URL = "https://feed.lolesports.com/livestats/v1/window/{gid}?startingTime={ts}"
COARSE_STEP_MIN = 10
COARSE_MAX_MIN = 180
MINUTES = range(0, 26)


def snap_10s(t): return t.replace(second=(t.second // 10) * 10, microsecond=0)


def get(url):
    try:
        raw = subprocess.check_output(["curl", "-sSfL", "--max-time", "15", url], stderr=subprocess.DEVNULL)
        return json.loads(raw) if raw.strip() else None
    except Exception:
        return None


def find_start(gid, scheduled):
    for off in range(0, COARSE_MAX_MIN + 1, COARSE_STEP_MIN):
        t = snap_10s(scheduled + dt.timedelta(minutes=off))
        d = get(FEED_URL.format(gid=gid, ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")))
        if not d: continue
        for f in (d.get("frames") or []):
            if f.get("gameState") == "in_game":
                return dt.datetime.fromisoformat(f["rfc460Timestamp"].replace("Z", "+00:00"))
    return None


def fetch_game(gid, scheduled_iso, out_dir):
    out_file = out_dir / f"{gid}.ndjson"
    if out_file.exists() and out_file.stat().st_size > 200:
        return "cached", 0
    scheduled = dt.datetime.fromisoformat(scheduled_iso.replace("Z", "+00:00"))
    start = find_start(gid, scheduled)
    if start is None:
        out_file.write_text(""); return "no_feed", 0
    n = 0
    with out_file.open("w") as fh:
        for m in MINUTES:
            t = snap_10s(start + dt.timedelta(minutes=m))
            d = get(FEED_URL.format(gid=gid, ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")))
            if not d: continue
            frames = d.get("frames") or []
            if not frames: continue
            f = frames[0]
            fh.write(json.dumps({
                "gid": gid, "minute": m, "ts": f.get("rfc460Timestamp"),
                "state": f.get("gameState"), "blue": f.get("blueTeam"), "red": f.get("redTeam"),
            }, separators=(",", ":")) + "\n")
            n += 1
            if f.get("gameState") == "finished": break
    return "ok", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("league")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    mapping_path = AUTO_DIR / args.league / "mapping.csv"
    if not mapping_path.exists():
        print(f"no mapping at {mapping_path}"); return
    out_dir = PROJECT / "data" / "frames" / f"{args.league.lower().replace(' ', '_')}_window"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(mapping_path.open()))
    print(f"[{args.league}] fetching {len(rows)} games with {args.workers} workers")
    status = {"ok": 0, "cached": 0, "no_feed": 0}; total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_game, r["esports_game_id"], r["start_time"], out_dir): r for r in rows}
        done = 0
        for f in as_completed(futs):
            s, n = f.result(); status[s] += 1; total += n; done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"[{args.league}] {done}/{len(rows)}  {status}  frames={total}")
    print(f"[{args.league}] DONE  {status}")


if __name__ == "__main__":
    main()
