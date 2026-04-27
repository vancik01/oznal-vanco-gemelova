# Per-Lane Feature Enrichment Experiment

**Date:** 2026-04-26 23:38
**Question:** Would adding per-lane (role) kills/deaths/assists differentials to the H2 feature set add real predictive value, justifying an "S1 part 2 with enriched metrics"?
**Verdict:** **No.** Adds 20 features, gains <0.005 AUC (within sampling noise), every new feature ranks in the bottom 30% of the model.

---

## Setup

Three feature sets compared, same 80/20 split (`set.seed(42)`), same RF hyperparameter search (`mtry ∈ {round(sqrt(p)), round(p/3), round(p/2)}`, `ntree = 500`), 5-fold CV optimised on ROC, parallel across 13 cores. Train rows: 7,388. Test rows: 1,848. Restricted to the games shared across all three sets so the test sample is identical for every model.

| Set | Definition | Count |
|---|---|---|
| **H1** | Current team-level features only | 20 |
| **H2** | H1 + per-role gold/xp/cs differentials at @10 and @15 (current Rmd) | 50 |
| **H3** | H2 + per-role kills/deaths/assists/kill-momentum at @15 (proposed) | 70 |

H3 features built by joining the Blue and Red player rows of the same role and computing role-vs-lane-opponent diffs. Source: `players` table from `2025_LoL_esports_match_data_from_OraclesElixir.csv`.

---

## Headline numbers

| Model | Features | Test AUC | Test Acc | ΔAUC vs H1 | ΔAcc vs H1 | mtry* |
|---|---|---|---|---|---|---|
| **H1** (team only) | 20 | 0.8274 | 0.7451 | - | - | 4 |
| **H2** (+ role g/x/c) | 50 | 0.8309 | 0.7408 | +0.0036 | **-0.0043** | 7 |
| **H3** (+ role k/d/a) | 70 | 0.8316 | 0.7457 | +0.0043 | +0.0005 | 8 |

Approx. binomial SE on AUC at n=1848 is ±0.008-0.010. **Both H2 and H3 lifts are well below detection threshold.** With a different seed they could flip sign. H2 actively loses 0.4pp of accuracy vs H1 - so adding features can hurt, not just fail to help.

---

## Where the new features rank in the H3 RF

Top 15 of 70 features:

| Rank | Feature | Importance |
|---|---|---|
| 1 | golddiffat15 | 100.0 |
| 2 | xpdiffat15 | 79.7 |
| 3 | csdiffat15 | 65.2 |
| 4 | gold_momentum | 51.2 |
| 5 | firsttower | 47.8 |
| 6 | winrate_diff | 47.7 |
| 7 | kda_15 | 35.5 |
| 8 | plate_diff | 33.0 |
| 9 | kill_diff_15 | 31.8 |
| **10** | **bot_golddiffat15** | **31.0** |
| 11 | cs_momentum | 30.7 |
| 12 | xp_momentum | 29.4 |
| 13 | death_diff_15 | 28.4 |
| 14 | mid_golddiffat15 | 24.9 |
| 15 | jng_xpdiffat15 | 23.3 |

**All proposed (H3-only) features rank 46-69** out of 70:

| Rank | Feature | Importance |
|---|---|---|
| 46 | sup_assists_diff_15 | 11.1 |
| 48 | jng_assists_diff_15 | 8.6 |
| 49 | mid_assists_diff_15 | 8.2 |
| 50 | jng_deaths_diff_15 | 7.5 |
| 51 | bot_assists_diff_15 | 6.5 |
| 53 | sup_deaths_diff_15 | 6.3 |
| 54 | bot_kills_diff_15 | 6.3 |
| 55 | jng_kills_diff_15 | 6.1 |
| 58 | mid_kills_diff_15 | 5.6 |
| 59 | top_assists_diff_15 | 5.6 |
| 60 | mid_deaths_diff_15 | 5.3 |
| 61 | top_deaths_diff_15 | 5.0 |
| 62 | top_kills_diff_15 | 4.9 |
| 63 | bot_deaths_diff_15 | 4.8 |
| 64 | bot_kill_momentum | 4.6 |
| 65 | jng_kill_momentum | 4.4 |
| 66 | mid_kill_momentum | 4.4 |
| 67 | top_kill_momentum | 2.9 |
| 68 | sup_kills_diff_15 | 2.9 |
| 69 | sup_kill_momentum | 0.86 |

Best new feature is `sup_assists_diff_15` at importance 11.0 - **11% of the strongest feature** (`golddiffat15` = 100.0). Worst is `sup_kill_momentum` ≈ 0.

Per-role best/mean ranks for the new features:

| Role | Best rank | Mean rank |
|---|---|---|
| bot | 51 | 58 |
| jng | 48 | 54 |
| mid | 49 | 58 |
| sup | 46 | 59 |
| top | 59 | 62 |

No role's k/d/a features break into the meaningful zone.

---

## Why this isn't surprising

1. **Information overlap.** Team `kill_diff_15` is exactly the sum of role `kills_diff_15`. The per-role split decomposes the same signal and adds noise. RF can sometimes exploit decomposition if specific roles' kills matter differently - but empirically here, they don't.
2. **Sparse integer counts at @15.** A typical Blue ADC has 0-2 kills by 15 min with high variance. Per-role kill diffs are noisy 0/±1/±2 integers - signal-to-noise is much worse than the continuous gold/xp/cs diffs.
3. **Strong baseline already.** Gold/XP/CS at 15 min already capture the consequences of early kills (kills → gold lead → resource diffs). The k/d/a layer is a redundant proxy.

---

## Side observation: H2 vs H1

H2's gold/xp/cs role diffs do add a tiny bit (+0.0036 AUC) and `bot_golddiffat15` ranks at #10 across all features - so the H2 hypothesis question (is bot lane the most predictive role?) is well-motivated. But H2 also slightly hurts accuracy. **The current Rmd correctly frames H2 as a hypothesis-testing exercise, not as a performance enhancement.**

---

## Recommendation

- **Do not add H3 to the analysis.** It costs 20 features, complicates interpretation, gains nothing measurable, and any LASSO/Elastic Net pass would zero them out anyway.
- **Keep H2 as a hypothesis test, not a "richer model" play.** The Rmd already does this.
- If a third feature-engineering hypothesis is wanted, better candidates than per-role k/d/a:
  - **Per-role gold share at 15** (bot ADC's % of team gold) - relative resource concentration, not redundant with absolute diff.
  - **Lane-pair synergy** (bot + sup combined diff) - matches how the bot lane actually plays.
  - **Side × patch interaction** - does the Blue-side advantage shift across patches 15.01-15.24?

---

## Reproducing

Script: `projekt/scratch_role_importance.R` (deleted after run; see git history if needed). The same logic can be re-derived from this report:
- Build `games_h3` by joining Blue and Red player rows of the same role and taking diffs.
- Train RF on H1, H2, H3 with shared `set.seed(42)` train/test split restricted to games present in all three sets.
- Compare test AUC and `varImp`.
