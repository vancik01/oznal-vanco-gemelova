# Autoresearch - LoL early-game H1 prediction

Goal: maximize **test AUC-ROC on H1** (team-level early-game prediction, team rows only, holdout = 20% of games).

## Loop
1. Read `experiment.R` and the last few entries in `journal.md`.
2. Form a hypothesis for what to change next (architecture, hyperparams, features, ensembling...).
3. Edit **only** `experiment.R`.
4. Run: `Rscript experiment.R 2>&1 | tee /tmp/ar_last.log`
5. Parse the `METRIC: <auc>` line from stdout.
6. Append an entry to `journal.md`:
   - run number, timestamp, duration
   - one-sentence hypothesis ("changed X because Y")
   - key diff
   - AUC, ACC, N_FEATURES
   - KEEP or DISCARD vs current best
7. If KEEP: copy `experiment.R` to `best.R`. Update "Current best" at top of `journal.md`.
8. If DISCARD: revert `experiment.R` from `best.R`.
9. Repeat.

## Hard rules (do NOT change)
- `set.seed(42)` stays fixed.
- Train/test split is 80/20 on `gameid` with that seed. Do not change the split logic.
- H1 scope: team rows only (`position == "team"`). Do not use player rows.
- No post-15-min features (no `totaldragons`, `totaltowers`, `gamelength`, end-of-game stats). 15-min snapshots and earlier only.
- No external data.
- Output format: final line must be `METRIC: <float>`. Also print `ACC:`, `N_FEATURES:`, `DURATION_SEC:`.
- Each run < ~3 minutes. Use multicore (`nthread`, `doParallel`) - 14 cores available.

## Fair game
- Model family (xgboost, glmnet/LASSO/Elastic Net, RF, CART, KNN, stacking, blending).
- Hyperparameters and tuning strategies (grid, random, CV-based).
- New engineered features from existing early-game columns.
- Regularization, class weighting, threshold tuning.
- Ensembles and model averaging.
- Cross-validation or inner holdout within the training set for hyperparameter selection.

## Current best
See top of `journal.md`.
