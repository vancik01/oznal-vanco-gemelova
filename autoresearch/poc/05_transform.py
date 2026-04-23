"""Transform per-minute NDJSON frames into tidy tables.

Emits:
  frames_per_minute.csv - one row per (esports_gid, minute, side)
  events.csv            - derived events (first_blood, first_dragon, etc.) per (gid, side)

Per-minute team columns: kills, gold, xp_proxy (none in window), cs (per-team sum),
  dragons_count, dragon_types (pipe-sep), towers, barons, inhibitors.
Per-minute per-player columns are aggregated to team level here; participant-level
  table is saved separately at frames_per_player.csv.

Note: window endpoint has no XP field. We derive team gold/cs/level from participants.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parents[1]
FRAMES_DIR = PROJECT / "data" / "frames" / "lec_window"
MAPPING = HERE / "gameid_mapping_lec.csv"

OUT_TEAM = HERE / "frames_per_minute.csv"
OUT_PLAYER = HERE / "frames_per_player.csv"
OUT_EVENTS = HERE / "events.csv"


def team_row(gid, minute, side, ts, state, team):
    ps = team.get("participants") or []
    return {
        "gid": gid, "minute": minute, "side": side, "ts": ts, "state": state,
        "kills": team.get("totalKills", 0),
        "gold": team.get("totalGold", 0),
        "dragons_count": len(team.get("dragons") or []),
        "dragon_types": "|".join(team.get("dragons") or []),
        "towers": team.get("towers", 0),
        "barons": team.get("barons", 0),
        "inhibitors": team.get("inhibitors", 0),
        "cs_sum": sum(p.get("creepScore", 0) for p in ps),
        "level_sum": sum(p.get("level", 0) for p in ps),
        "level_max": max((p.get("level", 0) for p in ps), default=0),
        "n_alive": sum(1 for p in ps if p.get("currentHealth", 0) > 0),
    }


def player_rows(gid, minute, side, team):
    # Convention: pid 1-5 = blue (top/jng/mid/bot/sup), 6-10 = red in same order.
    roles = ["top", "jng", "mid", "bot", "sup"]
    out = []
    ps = team.get("participants") or []
    for p in ps:
        pid = p.get("participantId")
        if pid is None:
            continue
        slot = (pid - 1) % 5
        out.append({
            "gid": gid, "minute": minute, "side": side,
            "participantId": pid, "role": roles[slot],
            "gold": p.get("totalGold", 0),
            "level": p.get("level", 0),
            "kills": p.get("kills", 0),
            "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
            "cs": p.get("creepScore", 0),
            "hp": p.get("currentHealth", 0),
            "maxhp": p.get("maxHealth", 0),
        })
    return out


def detect_events(frames_by_minute, gid):
    """Scan frames in minute order, emit event rows when a counter first increments."""
    events = []
    # First-events trackers
    firsts = {
        "first_blood": None,
        "first_dragon": None,
        "first_tower": None,
        "first_baron": None,
        "first_inhib": None,
    }
    prev_blue = None
    prev_red = None
    for minute in sorted(frames_by_minute):
        f = frames_by_minute[minute]
        b, r = f["blue"], f["red"]

        for side, team, prev in (("blue", b, prev_blue), ("red", r, prev_red)):
            if prev is None:
                continue
            # kill events
            if team.get("totalKills", 0) > prev.get("totalKills", 0) and firsts["first_blood"] is None:
                firsts["first_blood"] = (minute, side)
            # dragon events
            new_dragons = (team.get("dragons") or [])[len(prev.get("dragons") or []):]
            for d in new_dragons:
                if firsts["first_dragon"] is None:
                    firsts["first_dragon"] = (minute, side, d)
                events.append({"gid": gid, "minute": minute, "side": side, "type": "dragon", "subtype": d})
            # tower events
            if team.get("towers", 0) > prev.get("towers", 0):
                if firsts["first_tower"] is None:
                    firsts["first_tower"] = (minute, side)
                events.append({"gid": gid, "minute": minute, "side": side, "type": "tower", "subtype": ""})
            # baron events
            if team.get("barons", 0) > prev.get("barons", 0):
                if firsts["first_baron"] is None:
                    firsts["first_baron"] = (minute, side)
                events.append({"gid": gid, "minute": minute, "side": side, "type": "baron", "subtype": ""})
            # inhibitor events
            if team.get("inhibitors", 0) > prev.get("inhibitors", 0):
                if firsts["first_inhib"] is None:
                    firsts["first_inhib"] = (minute, side)
                events.append({"gid": gid, "minute": minute, "side": side, "type": "inhibitor", "subtype": ""})

        prev_blue, prev_red = b, r

    return events, firsts


def main():
    gids = [p.stem for p in FRAMES_DIR.glob("*.ndjson") if p.stat().st_size > 200]
    print(f"non-empty NDJSON files: {len(gids)}")

    team_rows_out = []
    player_rows_out = []
    event_rows_out = []
    firsts_rows = []

    for gid in gids:
        p = FRAMES_DIR / f"{gid}.ndjson"
        frames_by_minute = {}
        for line in p.read_text().splitlines():
            if not line:
                continue
            frames_by_minute[json.loads(line)["minute"]] = json.loads(line)

        for minute, f in frames_by_minute.items():
            for side, team in (("blue", f["blue"]), ("red", f["red"])):
                team_rows_out.append(team_row(gid, minute, side, f["ts"], f["state"], team))
                player_rows_out.extend(player_rows(gid, minute, side, team))

        events, firsts = detect_events(frames_by_minute, gid)
        event_rows_out.extend(events)
        firsts_rows.append({
            "gid": gid,
            "first_blood_minute": firsts["first_blood"][0] if firsts["first_blood"] else None,
            "first_blood_side": firsts["first_blood"][1] if firsts["first_blood"] else None,
            "first_dragon_minute": firsts["first_dragon"][0] if firsts["first_dragon"] else None,
            "first_dragon_side": firsts["first_dragon"][1] if firsts["first_dragon"] else None,
            "first_dragon_type": firsts["first_dragon"][2] if firsts["first_dragon"] else None,
            "first_tower_minute": firsts["first_tower"][0] if firsts["first_tower"] else None,
            "first_tower_side": firsts["first_tower"][1] if firsts["first_tower"] else None,
        })

    def write(path, rows):
        if not rows:
            print(f"  (no rows for {path.name})"); return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  {path.name}: {len(rows)} rows")

    write(OUT_TEAM, team_rows_out)
    write(OUT_PLAYER, player_rows_out)
    write(OUT_EVENTS, event_rows_out)
    write(HERE / "first_events.csv", firsts_rows)

    print(f"\ngames processed: {len(gids)}")
    print(f"first_blood detected: {sum(1 for r in firsts_rows if r['first_blood_minute'] is not None)}")
    print(f"first_dragon detected: {sum(1 for r in firsts_rows if r['first_dragon_minute'] is not None)}")


if __name__ == "__main__":
    main()
