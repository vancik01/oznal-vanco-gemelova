"""Auto-discover best lolesports slug per Oracle league, auto-generate team aliases,
and report coverage across every league.

For each Oracle league:
  1. Pull its Oracle games + unique teams.
  2. For every lolesports league, enumerate 2025 tournaments + games + teams.
  3. Score fuzzy similarity between Oracle team roster and each esports roster.
     Pick the slug with highest overlap.
  4. Within that pairing, greedily generate aliases (Oracle team -> esports team)
     using string similarity, threshold >= 0.6.
  5. Run join with aliases and record match rate.
"""
import csv
import json
import subprocess
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parents[1]
ORACLE_CSV = PROJECT / "data" / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
BASE = "https://esports-api.lolesports.com/persisted/gw"
OUT_DIR = HERE / "auto_coverage"
OUT_DIR.mkdir(exist_ok=True)

ALIAS_THRESHOLD = 0.6
MIN_OVERLAP = 3  # minimum team overlap to consider a slug match


def http(url):
    try:
        raw = subprocess.check_output(["curl", "-sSfL", "--max-time", "20",
                                        "-H", f"x-api-key: {API_KEY}", url],
                                       stderr=subprocess.DEVNULL)
        return json.loads(raw) if raw.strip() else None
    except Exception:
        return None


def norm(s):
    return s.strip().upper().replace(" ", "").replace(".", "").replace("'", "")


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def load_oracle_by_league():
    by_league = defaultdict(list)
    with ORACLE_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["position"] != "team":
                continue
            by_league[r["league"]].append(r)
    return by_league


def oracle_games(team_rows):
    """group oracle team-rows into (gameid, date, game_num, blue, red)"""
    by_gid = defaultdict(dict)
    for r in team_rows:
        g = by_gid[r["gameid"]]
        g["date"] = r["date"]; g["game_num"] = int(r["game"])
        if r["side"] == "Blue": g["blue"] = r["teamname"]
        else: g["red"] = r["teamname"]
    return [(gid, g["date"], g["game_num"], g.get("blue", ""), g.get("red", ""))
            for gid, g in by_gid.items() if "blue" in g and "red" in g]


def enumerate_all_esports_leagues():
    """Returns dict: slug -> list of (date_day, game_num, team_a, team_b, esports_game_id)"""
    data = http(f"{BASE}/getLeagues?hl=en-US")
    if not data:
        return {}
    leagues = data["data"]["leagues"]
    print(f"lolesports leagues total: {len(leagues)}")
    result = {}
    for l in leagues:
        slug = l["slug"]
        td = http(f"{BASE}/getTournamentsForLeague?hl=en-US&leagueId={l['id']}")
        if not td:
            continue
        tourneys = td["data"]["leagues"][0]["tournaments"]
        t_2025 = [t for t in tourneys if "2025" in t["slug"] or (t.get("startDate","")[:4]=="2025")]
        rows = []
        for t in t_2025:
            ed = http(f"{BASE}/getCompletedEvents?hl=en-US&tournamentId={t['id']}")
            if not ed:
                continue
            for ev in ed["data"]["schedule"]["events"]:
                m = ev.get("match") or {}
                teams = m.get("teams") or []
                if len(teams) != 2:
                    continue
                for idx, g in enumerate(ev.get("games") or [], start=1):
                    rows.append((ev["startTime"][:10], idx, teams[0]["name"], teams[1]["name"], g["id"], ev["startTime"]))
        if rows:
            result[slug] = rows
            print(f"  {slug:35s} tournaments={len(t_2025)}  game rows={len(rows)}")
    return result


def score_slug_match(oracle_teams, esports_teams):
    """how many oracle team names have a near-match in esports teams"""
    if not oracle_teams or not esports_teams:
        return 0
    e_norm = [norm(t) for t in esports_teams]
    matched = 0
    for ot in oracle_teams:
        on = norm(ot)
        if on in e_norm:
            matched += 1; continue
        # fuzzy
        best = max((sim(on, en) for en in e_norm), default=0.0)
        if best >= ALIAS_THRESHOLD:
            matched += 1
    return matched


def generate_aliases(oracle_teams, esports_teams):
    """greedy fuzzy alias generation, threshold ALIAS_THRESHOLD"""
    aliases = {}
    e_norm = {norm(t): t for t in esports_teams}
    for ot in oracle_teams:
        on = norm(ot)
        if on in e_norm:
            continue  # exact match
        best_sim = 0.0; best = None
        for en in e_norm:
            s = sim(on, en)
            if s > best_sim:
                best_sim = s; best = en
        if best and best_sim >= ALIAS_THRESHOLD:
            aliases[on] = best
    return aliases


def do_join(oracle_game_list, esports_rows, aliases):
    """Index esports by normalized names (no aliasing). Apply aliases on Oracle side only."""
    idx = {}
    for dt, gn, ta, tb, egid, stime in esports_rows:
        teams = frozenset([norm(ta), norm(tb)])
        idx.setdefault((dt, teams, gn), []).append((egid, stime, ta, tb))
    matches = []
    for gid, date, gn, blue, red in oracle_game_list:
        dkey = date[:10]
        bn = aliases.get(norm(blue), norm(blue))
        rn = aliases.get(norm(red), norm(red))
        teams = frozenset([bn, rn])
        hits = idx.get((dkey, teams, gn), [])
        if len(hits) >= 1:  # first hit; ambiguity rare in practice
            egid, stime, ta, tb = hits[0]
            matches.append({
                "oracle_gameid": gid, "esports_game_id": egid,
                "start_time": stime, "date": date, "game_num": gn,
                "blue": blue, "red": red,
            })
    return matches


def main():
    oracle = load_oracle_by_league()
    print(f"Oracle leagues: {len(oracle)}")
    total_oracle_games = sum(len(oracle_games(rows)) for rows in oracle.values())
    print(f"Total Oracle games: {total_oracle_games}\n")

    all_esports = enumerate_all_esports_leagues()
    print(f"\nEsports slugs with 2025 games: {len(all_esports)}\n")

    summary = []
    per_league_mapping = {}

    for oleague, rows in sorted(oracle.items(), key=lambda kv: -len(kv[1])):
        games = oracle_games(rows)
        oracle_team_set = set()
        for _, _, _, b, r in games:
            oracle_team_set.add(b); oracle_team_set.add(r)

        # Score every slug by team-roster overlap
        best_slug = None; best_score = 0; best_matches_pct = 0.0
        best_aliases = {}; best_mapping = []
        for slug, erows in all_esports.items():
            etm = set()
            for _, _, a, b, _, _ in erows:
                etm.add(a); etm.add(b)
            score = score_slug_match(oracle_team_set, etm)
            if score < MIN_OVERLAP:
                continue
            # generate aliases + do the full join
            aliases = generate_aliases(oracle_team_set, etm)
            m = do_join(games, erows, aliases)
            pct = len(m) / len(games) if games else 0
            if pct > best_matches_pct or (pct == best_matches_pct and score > best_score):
                best_slug = slug; best_score = score
                best_matches_pct = pct; best_aliases = aliases; best_mapping = m

        summary.append({
            "oracle_league": oleague,
            "oracle_games": len(games),
            "best_slug": best_slug or "",
            "team_overlap": best_score,
            "matched": len(best_mapping),
            "match_pct": f"{100*best_matches_pct:.1f}",
            "n_aliases": len(best_aliases),
        })
        per_league_mapping[oleague] = (best_slug, best_aliases, best_mapping)

    # Write summary
    with (OUT_DIR / "coverage_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # Write per-league mapping + aliases
    for oleague, (slug, aliases, matches) in per_league_mapping.items():
        if not matches:
            continue
        d = OUT_DIR / oleague.replace(" ", "_").replace("/", "_")
        d.mkdir(exist_ok=True)
        with (d / "mapping.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
            w.writeheader(); w.writerows(matches)
        with (d / "aliases.json").open("w") as f:
            json.dump({"slug": slug, "aliases": aliases}, f, indent=2, ensure_ascii=False)

    # Pretty print
    print(f"\n{'OracleLeague':<15} {'Games':>6} {'BestSlug':<25} {'Ovl':>4} {'Matched':>8} {'Pct':>6} {'Ali':>4}")
    for s in sorted(summary, key=lambda x: -x["matched"]):
        print(f"{s['oracle_league']:<15} {s['oracle_games']:>6} {s['best_slug']:<25} {s['team_overlap']:>4} {s['matched']:>8} {s['match_pct']:>5}% {s['n_aliases']:>4}")

    total_matched = sum(s["matched"] for s in summary)
    print(f"\nTOTAL matched games across all leagues: {total_matched}")


if __name__ == "__main__":
    main()
