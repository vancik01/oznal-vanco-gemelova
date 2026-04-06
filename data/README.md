# 2025 LoL Esports Match Data (Oracle's Elixir)

## Overview

Professional League of Legends esports match data from the 2025 competitive season, sourced from Oracle's Elixir.

- **Rows**: 120,636
- **Columns**: 165 (142 numeric, 23 text)
- **Date range**: 2025-01-11 to 2025-12-29
- **Games**: 10,053 unique matches
- **Teams**: 445 unique teams
- **Leagues**: 45 professional leagues (LPL, LCK, LEC, LFL, LAS, EM, ...)
- **Patches**: 24 (15.01 - 15.24)

## Row Structure

Each game produces **12 rows** - one per player (5 per team) + one team-level summary per side:

| Position value | Description | Rows |
|---|---|---|
| `top`, `jng`, `mid`, `bot`, `sup` | Individual player stats | 100,530 (5 x 20,106) |
| `team` | Aggregated team-level stats | 20,106 |

Each game has exactly 2 sides: **Blue** and **Red** (50/50 split).

## Target Variable

- `result`: **0** = loss, **1** = win
- Perfectly balanced: 60,318 wins / 60,318 losses

## Data Completeness

- `complete`: 110,832 rows (91.9%) - full early-game timeline data available
- `partial`: 9,804 rows (8.1%) - missing early-game snapshots (goldat10, xpat10, etc.)

## Column Groups

### Metadata
| Column | Description |
|---|---|
| `gameid` | Unique game identifier |
| `league` | Tournament/league name |
| `year` | Season year (2025) |
| `split` | Season split (Winter, Spring, Summer, etc.) |
| `playoffs` | 0 = regular season, 1 = playoffs |
| `date` | Match datetime |
| `game` | Game number within a series |
| `patch` | Game patch version |
| `side` | Blue / Red |
| `position` | top, jng, mid, bot, sup, team |
| `playername` | Player name (NULL for team rows) |
| `teamname` | Team name |
| `datacompleteness` | complete / partial |

### Draft (team rows)
| Column | Description |
|---|---|
| `firstPick` | 1 = team had first pick |
| `champion` | Champion played (player rows only) |
| `ban1` - `ban5` | Champions banned |
| `pick1` - `pick5` | Champions picked in draft order |

### Game Outcome
| Column | Description |
|---|---|
| `result` | 0 = loss, 1 = win |
| `gamelength` | Game duration in seconds |

### Combat Stats
| Column | Description |
|---|---|
| `kills`, `deaths`, `assists` | KDA stats |
| `teamkills`, `teamdeaths` | Team totals |
| `doublekills`, `triplekills`, `quadrakills`, `pentakills` | Multi-kills |
| `firstblood`, `firstbloodkill`, `firstbloodassist`, `firstbloodvictim` | First blood events |
| `team kpm` | Team kills per minute |
| `ckpm` | Combined kills per minute (both teams) |
| `damagetochampions`, `dpm`, `damageshare` | Damage stats |
| `damagetakenperminute`, `damagemitigatedperminute` | Defensive stats |

### Objectives
| Column | Description |
|---|---|
| `firstdragon`, `dragons`, `opp_dragons` | Dragon control |
| `elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `chemtechs`, `hextechs` | Drake types |
| `elders`, `opp_elders` | Elder dragon |
| `firstherald`, `heralds`, `opp_heralds` | Rift Herald |
| `void_grubs`, `opp_void_grubs` | Void Grubs |
| `firstbaron`, `barons`, `opp_barons` | Baron Nashor |
| `atakhans`, `opp_atakhans` | Atakhan |
| `firsttower`, `towers`, `opp_towers` | Towers |
| `firstmidtower`, `firsttothreetowers` | Tower milestones |
| `turretplates`, `opp_turretplates` | Turret plates |
| `inhibitors`, `opp_inhibitors` | Inhibitors |
| `damagetotowers` | Damage to towers |

### Vision
| Column | Description |
|---|---|
| `wardsplaced`, `wpm` | Wards placed / per minute |
| `wardskilled`, `wcpm` | Wards killed / per minute |
| `controlwardsbought` | Control wards purchased |
| `visionscore`, `vspm` | Vision score / per minute |

### Economy
| Column | Description |
|---|---|
| `totalgold`, `earnedgold`, `earned gpm` | Gold stats |
| `earnedgoldshare` | % of team gold |
| `goldspent` | Gold spent |
| `gspd` | Gold spent percentage difference |
| `gpr` | Gold percent rating |
| `total cs`, `minionkills`, `monsterkills` | Creep score |
| `monsterkillsownjungle`, `monsterkillsenemyjungle` | Jungle camps |
| `cspm` | CS per minute |

### Early-Game Snapshots (at 10, 15, 20, 25 min)

Available for each timestamp (`X` = 10, 15, 20, 25):

| Column | Description |
|---|---|
| `goldatX`, `xpatX`, `csatX` | Gold, XP, CS at minute X |
| `opp_goldatX`, `opp_xpatX`, `opp_csatX` | Opponent's values |
| `golddiffatX`, `xpdiffatX`, `csdiffatX` | Differences (you - opponent) |
| `killsatX`, `assistsatX`, `deathsatX` | KDA at minute X |
| `opp_killsatX`, `opp_assistsatX`, `opp_deathsatX` | Opponent KDA |

**Availability**: ~92% for @10/@15, slightly less for @20/@25 (games that ended early).

## Columns with High Missing Values (>80%)

These columns are mostly available only for team rows in "complete" games:

- `dragons (type unknown)` - 98.6% missing
- `monsterkillsownjungle`, `monsterkillsenemyjungle` - 91.9% missing
- Drake subtypes, `elders`, `atakhans`, `firstherald`, `firstbaron`, `firstmidtower`, `firsttothreetowers`, `turretplates` - ~84.7% missing (available only on team rows)

## Top Leagues by Row Count

| League | Rows |
|---|---|
| LPL (China) | 9,660 |
| LCK (Korea) | 6,660 |
| LCKC (Korea Challengers) | 6,504 |
| LAS (Latin America South) | 5,868 |
| EM (EMEA Masters) | 5,280 |
| LJL (Japan) | 4,488 |
| LFL (France) | 3,816 |
| LEC (Europe) | 3,672 |
