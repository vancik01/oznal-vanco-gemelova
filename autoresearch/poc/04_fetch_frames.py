"""Fetch one frame per minute for each mapped LEC game (minutes 0..25 from game start).

Two-phase:
  1) coarse probe every 5 min up to +3h from scheduled, find first in_game frame
  2) for minute m in 0..25, query at (game_start + m*60s), keep the first frame

Output: NDJSON at data/frames/lec_window/{esports_game_id}.ndjson
  one line per minute: {gid, minute, ts, state, blue, red}
"""
import csv
import datetime as dt
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parents[1]
MAPPING = HERE / "gameid_mapping_lec.csv"
OUT_DIR = PROJECT / "data" / "frames" / "lec_window"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEED_URL = "https://feed.lolesports.com/livestats/v1/window/{gid}?startingTime={ts}"
COARSE_STEP_MIN = 5
COARSE_MAX_MIN = 180
MINUTES_TO_CAPTURE = range(0, 26)  # 0..25
WORKERS = 8


def snap_10s(t: dt.datetime) -> dt.datetime:
    return t.replace(second=(t.second // 10) * 10, microsecond=0)


def get(url: str):
    try:
        raw = subprocess.check_output(
            ["curl", "-sSfL", "--max-time", "15", url],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(raw) if raw.strip() else None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def find_game_start(gid: str, scheduled: dt.datetime) -> dt.datetime | None:
    for offset in range(0, COARSE_MAX_MIN + 1, COARSE_STEP_MIN):
        t = snap_10s(scheduled + dt.timedelta(minutes=offset))
        d = get(FEED_URL.format(gid=gid, ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")))
        if not d:
            continue
        for f in (d.get("frames") or []):
            if f.get("gameState") == "in_game":
                return dt.datetime.fromisoformat(f["rfc460Timestamp"].replace("Z", "+00:00"))
    return None


def fetch_game(gid: str, scheduled_iso: str) -> tuple[str, str, int]:
    out_file = OUT_DIR / f"{gid}.ndjson"
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
            d = get(FEED_URL.format(gid=gid, ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")))
            if not d:
                continue
            frames = d.get("frames") or []
            if not frames:
                continue
            f = frames[0]
            fh.write(json.dumps({
                "gid": gid,
                "minute": m,
                "ts": f.get("rfc460Timestamp"),
                "state": f.get("gameState"),
                "blue": f.get("blueTeam"),
                "red": f.get("redTeam"),
            }, separators=(",", ":")) + "\n")
            n += 1
            if f.get("gameState") == "finished":
                break

    return gid, "ok", n


def main():
    rows = list(csv.DictReader(MAPPING.open()))
    print(f"games to fetch: {len(rows)}")

    done = 0
    status_count = {"ok": 0, "cached": 0, "no_feed": 0}
    total_frames = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(fetch_game, r["esports_game_id"], r["start_time"]): r
            for r in rows
        }
        for fut in as_completed(futures):
            gid, status, nf = fut.result()
            status_count[status] += 1
            total_frames += nf
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}  ok={status_count['ok']} no_feed={status_count['no_feed']}  total_frames={total_frames}")

    print(f"\nfinal: {status_count}")
    sizes = [p.stat().st_size for p in OUT_DIR.glob("*.ndjson")]
    nonempty = [s for s in sizes if s > 200]
    if nonempty:
        print(f"nonempty NDJSON: {len(nonempty)}  total: {sum(nonempty)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
