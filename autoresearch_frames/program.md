# Autoresearch - LoL frame-based early-game prediction

Goal: maximize **test AUC-ROC** for team win prediction using the scraped per-minute frame data plus compatible Oracle team labels.

## Workflow
1. Run `Rscript prepare_dataset.R` when the underlying frame archive or mapping logic changes.
2. Read `experiment.R` and the latest entries in `journal.md`.
3. Form a hypothesis for the next change.
4. Edit **only** `experiment.R`.
5. Run: `Rscript experiment.R`
6. Parse the final metrics from stdout.
7. Append a run entry to `journal.md`:
   - run number, timestamp, duration
   - one-sentence hypothesis
   - key model / feature changes
   - AUC, ACC, baseline ACC
   - KEEP or DISCARD
8. If KEEP: copy `experiment.R` to `best.R` and update the "Current best" section.
9. If DISCARD: revert `experiment.R` from `best.R`.
10. Repeat.

## Hard rules
- Keep `set.seed(42)` fixed.
- Split train/test on `oracle_gameid`, not on rows.
- Use only information available by minute 15 or earlier.
- No external data.
- `prepare_dataset.R` is fixed infrastructure, not part of the search loop.
- `experiment.R` must print these final lines:
  - `AUC: <float>`
  - `ACC: <float>`
  - `BASELINE_ACC: <float>`
  - `N_FEATURES: <int>`
  - `DURATION_SEC: <float>`

## Fair game
- Model family in R (`glmnet`, `xgboost`, calibrated blends, etc.)
- Feature subset selection from the prepared frame table
- Alternative imputations, standardization, interactions, regularization
- Ensembling and threshold tuning inside the training split

## Dataset
- Source frames: `data/frames/*/*.ndjson`
- Labels: Oracle team rows from `data/2025_LoL_esports_match_data_from_OraclesElixir.csv`
- Join key: Oracle ↔ esports mappings discovered under `autoresearch/poc`

## Current best
See `journal.md`.
