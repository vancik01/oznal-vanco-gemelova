"""Build 2 feature matrices keyed on (oracle_gameid, side) with the result label.

A) oracle_features.csv - baseline: stats that Oracle already provides at 15 min
B) frame_features.csv  - new: derived from per-minute frames (trajectories + timings)

Both joined on the same (oracle_gameid, side) keys so we can compare fairly.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parents[1]
ORACLE_CSV = PROJECT / "data" / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
MAPPING = HERE / "gameid_mapping_lec.csv"
TEAM_FRAMES = HERE / "frames_per_minute.csv"
PLAYER_FRAMES = HERE / "frames_per_player.csv"
FIRST_EVENTS = HERE / "first_events.csv"

OUT_ORACLE = HERE / "oracle_features.csv"
OUT_FRAME = HERE / "frame_features.csv"
OUT_COMBINED = HERE / "combined_features.csv"


def load_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def f(x, default=0.0):
    try:
        return float(x) if x not in ("", None, "NA") else default
    except ValueError:
        return default


def i(x, default=0):
    try:
        return int(float(x)) if x not in ("", None, "NA") else default
    except ValueError:
        return default


def main():
    # 1. Oracle team rows for LEC, indexed by (gameid, side)
    oracle_team = {}
    with ORACLE_CSV.open() as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            if row["league"] != "LEC" or row["position"] != "team":
                continue
            oracle_team[(row["gameid"], row["side"])] = row
    print(f"oracle LEC team rows: {len(oracle_team)}")

    # 2. oracle_gameid -> esports_game_id
    mapping = {r["oracle_gameid"]: r["esports_game_id"] for r in load_csv(MAPPING)}
    print(f"mapping entries: {len(mapping)}")

    # 3. Frame data, indexed by (esports_gid, minute, side)
    frames = defaultdict(dict)
    for r in load_csv(TEAM_FRAMES):
        key = (r["gid"], int(r["minute"]), r["side"].lower())
        frames[key] = r
    print(f"frame rows: {len(frames)}")

    # 4. Per-player frames for role-specific features
    player_frames = defaultdict(list)  # (gid, minute, side) -> list of rows
    for r in load_csv(PLAYER_FRAMES):
        key = (r["gid"], int(r["minute"]), r["side"].lower())
        player_frames[key].append(r)

    # 5. First-event table
    firsts = {r["gid"]: r for r in load_csv(FIRST_EVENTS)}

    # --- Build feature rows ---
    oracle_feats = []
    frame_feats = []

    for (ogid, side), orow in oracle_team.items():
        if ogid not in mapping:
            continue
        egid = mapping[ogid]
        side_l = side.lower()

        # Oracle baseline features at 15 min (plus minimal pre-game meta)
        orf = {
            "ogid": ogid, "side": side, "result": i(orow["result"]),
            "side_blue": 1 if side == "Blue" else 0,
            "gold_15": f(orow["goldat15"]),
            "golddiff_15": f(orow["golddiffat15"]),
            "xp_15": f(orow["xpat15"]),
            "xpdiff_15": f(orow["xpdiffat15"]),
            "cs_15": f(orow["csat15"]),
            "csdiff_15": f(orow["csdiffat15"]),
            "kills_15": f(orow["killsat15"]),
            "deaths_15": f(orow["deathsat15"]),
            "assists_15": f(orow["assistsat15"]),
            "firstblood": f(orow["firstblood"]),
            "firstdragon": f(orow["firstdragon"]),
            "firstherald": f(orow["firstherald"]),
            "firsttower": f(orow["firsttower"]),
        }
        oracle_feats.append(orf)

        # Frame-derived features
        # Get team snapshots at minutes 0..15 for this side
        snaps = {}
        for m in range(0, 16):
            snaps[m] = frames.get((egid, m, side_l))
        opp_side = "red" if side_l == "blue" else "blue"
        opp_snaps = {}
        for m in range(0, 16):
            opp_snaps[m] = frames.get((egid, m, opp_side))

        # Skip game if we don't have minute 15
        if not snaps.get(15) or not opp_snaps.get(15):
            continue

        def g(m, key, snap=snaps):
            return f(snap[m][key]) if snap.get(m) else 0.0

        # Raw snapshots at 5/10/15
        gold_5 = g(5, "gold"); gold_10 = g(10, "gold"); gold_15 = g(15, "gold")
        opp_gold_5 = g(5, "gold", opp_snaps); opp_gold_10 = g(10, "gold", opp_snaps); opp_gold_15 = g(15, "gold", opp_snaps)

        cs_5 = g(5, "cs_sum"); cs_10 = g(10, "cs_sum"); cs_15 = g(15, "cs_sum")
        opp_cs_15 = g(15, "cs_sum", opp_snaps)

        kills_5 = g(5, "kills"); kills_10 = g(10, "kills"); kills_15 = g(15, "kills")
        opp_kills_15 = g(15, "kills", opp_snaps)

        level_max_15 = g(15, "level_max")
        opp_level_max_15 = g(15, "level_max", opp_snaps)

        dragons_15 = g(15, "dragons_count")
        opp_dragons_15 = g(15, "dragons_count", opp_snaps)
        towers_15 = g(15, "towers")
        opp_towers_15 = g(15, "towers", opp_snaps)

        # Gold trajectory: linear slope over minutes 0-15 and 0-10 / 10-15
        import statistics
        def slope(xs, ys):
            if len(xs) < 2: return 0.0
            n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
            num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
            den = sum((x-mx)**2 for x in xs) or 1
            return num/den

        gold_series = [(m, g(m, "gold")) for m in range(0, 16) if snaps.get(m)]
        opp_gold_series = [(m, g(m, "gold", opp_snaps)) for m in range(0, 16) if opp_snaps.get(m)]
        gold_diff_series = []
        for m in range(0, 16):
            if snaps.get(m) and opp_snaps.get(m):
                gold_diff_series.append((m, g(m, "gold") - g(m, "gold", opp_snaps)))

        gold_slope_0_15 = slope([x for x,_ in gold_series], [y for _,y in gold_series])
        gold_slope_0_10 = slope([x for x,y in gold_series if x<=10], [y for x,y in gold_series if x<=10])
        gold_slope_10_15 = slope([x for x,y in gold_series if x>=10], [y for x,y in gold_series if x>=10])
        golddiff_slope_0_15 = slope([x for x,_ in gold_diff_series], [y for _,y in gold_diff_series])

        # Kill tempo: kills in windows [0-5], [5-10], [10-15]
        kills_0_5 = kills_5 - g(0, "kills")
        kills_5_10 = kills_10 - kills_5
        kills_10_15 = kills_15 - kills_10

        # Objective timings (minutes). If not happened by min 15, encode as 99 (sentinel).
        fe = firsts.get(egid, {})
        def ev_minute(prefix, my_side):
            m = fe.get(f"{prefix}_minute")
            if not m: return 99.0
            s = fe.get(f"{prefix}_side")
            if s == my_side and int(float(m)) <= 15:
                return float(m)
            return 99.0 if s != my_side else 99.0  # we only want: time if WE did it, else 99

        # First blood/dragon/tower from MY perspective: 0..15 if we did it, else 99
        first_blood_me = ev_minute("first_blood", side_l)
        first_dragon_me = ev_minute("first_dragon", side_l)
        first_tower_me = ev_minute("first_tower", side_l)
        first_dragon_type = fe.get("first_dragon_type") or ""
        got_first_dragon = 1.0 if fe.get("first_dragon_side") == side_l else 0.0
        got_first_blood = 1.0 if fe.get("first_blood_side") == side_l else 0.0
        got_first_tower = 1.0 if fe.get("first_tower_side") == side_l else 0.0

        # Role-specific gold diffs at minute 10 and 15 (H2)
        role_gold = {}
        for m in (10, 15):
            my_p = {r["role"]: f(r["gold"]) for r in player_frames.get((egid, m, side_l), [])}
            op_p = {r["role"]: f(r["gold"]) for r in player_frames.get((egid, m, opp_side), [])}
            for role in ("top", "jng", "mid", "bot", "sup"):
                role_gold[f"golddiff_{role}_{m}"] = my_p.get(role, 0) - op_p.get(role, 0)

        frame_feats.append({
            "ogid": ogid, "side": side, "result": i(orow["result"]),
            "side_blue": 1 if side == "Blue" else 0,
            # Minute 15 snapshot (reproducing Oracle stats but from our feed)
            "gold_15_f": gold_15,
            "golddiff_15_f": gold_15 - opp_gold_15,
            "cs_15_f": cs_15,
            "csdiff_15_f": cs_15 - opp_cs_15,
            "kills_15_f": kills_15,
            "killdiff_15_f": kills_15 - opp_kills_15,
            "dragons_15_f": dragons_15,
            "dragdiff_15_f": dragons_15 - opp_dragons_15,
            "towers_15_f": towers_15,
            "towerdiff_15_f": towers_15 - opp_towers_15,
            "levelmax_15_f": level_max_15,
            "levelmax_diff_15_f": level_max_15 - opp_level_max_15,
            # Trajectory
            "gold_slope_0_15": gold_slope_0_15,
            "gold_slope_0_10": gold_slope_0_10,
            "gold_slope_10_15": gold_slope_10_15,
            "gold_accel": gold_slope_10_15 - gold_slope_0_10,
            "golddiff_slope_0_15": golddiff_slope_0_15,
            # Kill tempo
            "kills_0_5": kills_0_5, "kills_5_10": kills_5_10, "kills_10_15": kills_10_15,
            # Event timings & ownership
            "first_blood_me": first_blood_me,
            "first_dragon_me": first_dragon_me,
            "first_tower_me": first_tower_me,
            "got_first_blood": got_first_blood,
            "got_first_dragon": got_first_dragon,
            "got_first_tower": got_first_tower,
            "first_dragon_chemtech": 1.0 if got_first_dragon and first_dragon_type == "chemtech" else 0.0,
            "first_dragon_infernal": 1.0 if got_first_dragon and first_dragon_type == "infernal" else 0.0,
            "first_dragon_ocean": 1.0 if got_first_dragon and first_dragon_type == "ocean" else 0.0,
            "first_dragon_cloud": 1.0 if got_first_dragon and first_dragon_type == "cloud" else 0.0,
            "first_dragon_mountain": 1.0 if got_first_dragon and first_dragon_type == "mountain" else 0.0,
            "first_dragon_hextech": 1.0 if got_first_dragon and first_dragon_type == "hextech" else 0.0,
            **role_gold,
        })

    def write(path, rows):
        if not rows: print(f"  (empty) {path}"); return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  {path.name}: {len(rows)} rows x {len(rows[0])} cols")

    write(OUT_ORACLE, oracle_feats)
    write(OUT_FRAME, frame_feats)

    # Combined: inner-join on (ogid, side), keep label once
    oracle_by_key = {(r["ogid"], r["side"]): r for r in oracle_feats}
    frame_by_key = {(r["ogid"], r["side"]): r for r in frame_feats}
    shared = set(oracle_by_key) & set(frame_by_key)
    combined = []
    for k in shared:
        r = {**oracle_by_key[k]}
        # drop duplicated key columns from frame side
        for col, val in frame_by_key[k].items():
            if col not in ("ogid", "side", "result", "side_blue"):
                r[col] = val
        combined.append(r)
    write(OUT_COMBINED, combined)


if __name__ == "__main__":
    main()
