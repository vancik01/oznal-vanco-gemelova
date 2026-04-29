# OZNAL Consultation Notes - 2026-04-24

Raw notes from the mandatory consultation, summarised and turned into an action list.

## Raw notes (as given)

- Štruktúra pipeline - S1 nad S3
- Baseline - 73%, náš model 75.3%. Je to ok?
    - Try 10m & 15m
- Vyhodnotenie hypotéz podla stat. signifikantnosti?
    - Súčasť EDA - chi^2 i guess?
        - 2 hypotézy - minion kills, Blue / Red side
    - ML hypotéza - vieme predikovať
- explore space? To be safe?

### TODOs
- Support for our claim - theoretical max at ~75%
    - Articles / other work - mention theoretical max
    - Game showcase (optional)
    - Different data sources - engineering of features
        - 10m vs 15m
        - Per minute data -> ~400 features
- Onepager - what is that?
- Shiny app
    - Hyperparams + prefilled our best solution
    - Cards -> methods (method1: Raw model, method2: ...)
    - Not design necessary
    - Hyperparams
    - Check what Tuning
    - Vizualizácia - ROC
    - Feature selection -> show list of features
- Correlation map -> scale from -1 to 1
- Separate games -> keep only single entry per game - merge features to single row -> predict left_side_win -> 0/1
- Check imbalance after merge
- S1 -> keep multicollinearity -> feed all features - we'll see

---

## Summary

The notes cluster into three concerns:

### 1. Framing / narrative
- Current pipeline runs S1 -> S3. Confirm this is the right order given that feature engineering should logically come before model comparison.
- Best model (75.3%) is only ~2pp above the naive "gold-lead wins" baseline (72.9%). Need defensible framing that this is near a theoretical ceiling, not weak modelling.
- Hypothesis evaluation needs a statistical-significance component in EDA (e.g. chi^2 for binary side/minion-kill style hypotheses), not just ML AUC.

### 2. Data experiments to strengthen the "near-theoretical-max" claim
- Try 10-min features alone and 15-min features alone (currently momentum features combine both).
- Explore per-minute data -> ~400 features (if available in the source).
- Pull published numbers (Spaargaren 77.99%, Lafrance & Grewal 77.1%, etc. - already in the related-work table) and use them explicitly as the ceiling argument.

### 3. Data-shape change: one row per game (currently two)
- Collapse each game from two mirror rows (Blue/Red) to a single row, merging features and predicting `left_side_win` in {0, 1}.
- Check class imbalance after the merge (Blue-side win rate ~53%, so expect mild imbalance).
- For S1, keep multicollinearity in and feed all features - "we'll see what happens". Regularisation in S3 is where collinearity gets handled.

### 4. Shiny app spec (Adriana's scope)
- Hyperparameter controls for each model, pre-filled with the best solution we found.
- One card per method (raw model, tuned, feature-selected, etc.).
- No visual design polish required.
- Must show tuning grid results, ROC overlay, and the selected-feature list from S3.
- Correlation map needs a proper symmetric -1 to +1 scale (current corrplot already does this - verify it reads correctly).

### 5. "Onepager"
- Ask at next consult - scope is unclear.
- Per prior project memory: it's the 5-point executive summary, human-written, no LLM text. Scoring 5 pts, part of the 60-pt total.

---

## Action list

### High priority - data / modelling
- [ ] Restructure to single row per game (`left_side_win` target). Rebuild train/test split on this schema. Verify accuracy stays in the 75%+ range and check post-merge class balance.
- [ ] Run 3 feature-set variants through the S1 pipeline: `@10 only`, `@15 only`, `@10+@15 momentum` (current). Produce a 3-way comparison table to show where the extra signal actually lives.
- [ ] Investigate per-minute data in Oracle's Elixir - does the 2025 CSV expose per-minute timelines, or is 10/15 the ceiling of what's there? If per-minute available, engineer the ~400-feature set and run it as an optional high-feature experiment.
- [ ] Add chi^2 / proportion tests to EDA for binary hypotheses (Blue vs Red side win rate, first-blood win rate, objective win rates). Report p-values alongside the current bar plots so hypothesis verification has a statistical-significance story, not just a ML AUC one.
- [ ] S1 multicollinearity variant: run LR/RF/NB once with all raw + diff + opp features included (no pruning) to explicitly demonstrate the LR rank-deficiency warning and motivate the S3 selection story.

### High priority - narrative
- [ ] Fill in `# Conclusions` section (currently three TODO stubs at L1856-1866): H1 findings, H2 findings, S1xS3 synthesis.
- [ ] Add explicit "theoretical ceiling" paragraph to the related-work section citing the 77-78% band from 3+ papers and framing our 75.3% as within 2-3pp of that ceiling on 2025 data (which Tsang 2025 shows is harder than prior seasons).

### Medium priority - Shiny app
- [ ] Hyperparameter inputs per model, defaulting to best-found values (`mtry = rf_model$bestTune$mtry`, `k = knn_model$bestTune$k`, etc.).
- [ ] Method cards: Raw / Tuned / Feature-selected tab layout.
- [ ] ROC overlay across selected models (reuse `roc_*` objects).
- [ ] Feature-selection tab showing `overlap_df` (already saved to `shiny/models/overlap_df.rds`).
- [ ] Verify correlation heatmap legend runs -1 -> +1 symmetric.

### Low priority / clarify
- [ ] Onepager: confirm scope with Adriana (she owns it per WORKPLAN.md). Prior memory already has the spec - 5-pt executive summary, written manually, no LLM.
- [ ] Game showcase: optional, the notes flag it as a "maybe". Skip unless time allows.

### Open question
- S1 order: the notes say "S1 over S3" with a question mark. Decide whether to reorder the Rmd so feature selection runs before model comparison, or keep the current "S1 comparison -> S3 selection -> synthesis" order and defend that ordering in the narrative.
