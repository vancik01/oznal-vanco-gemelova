# Autoresearch Journal - H1 test AUC-ROC

Metric: **test AUC-ROC** on H1 holdout (20% of games, seed 42). Higher is better.

## Current best

- **Run 22** - AUC **0.84565** - XGB + bagged-LASSO (B=20) OOF blend, blend alpha=0.04 (basically pure LASSO). ACC **0.75703**. Bagging the final LASSO over 20 bootstraps reduces variance marginally beyond Run 18.

---

## Runs

### Run 1 - 2026-04-19 - baseline
- Hypothesis: establish starting point with clean XGBoost + inner val set for early stopping.
- Config: XGBoost, eta=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30. Inner val = 15% of train games.
- AUC: **0.84125** | ACC: 0.75108 | N_FEATURES: 21 | BEST_NROUNDS: 91 | DUR: 2.0s
- Decision: **KEEP** (first run, establishes best).

### Run 2 - 2026-04-19 - more features
- Hypothesis: more engineered features (interactions plate*tower, grub*herald, gold*tower, raw xp/cs at 15, economy ratios, obj_total) give XGBoost more signal.
- AUC: **0.83858** (-0.00267) | ACC: 0.74648 | N_FEATURES: 31 | BEST_NROUNDS: 125 | DUR: 2.6s
- Decision: **DISCARD**. Extra features added noise and hurt. Model was already capturing these interactions implicitly via tree splits.

### Run 3 - 2026-04-19 - parallel CV hyperparameter grid
- Hypothesis: baseline eta/depth/mcw were untuned. Proper 5-fold CV with a small grid should find better settings.
- Config: grid 3 eta x 3 depth x 2 mcw = 18 combos, each 5-fold CV, grid parallelized via `foreach %dopar%` (14 workers, nthread=1 per worker). Final retrain on full train at CV-best iter with nthread=14.
- CV best: eta=0.03 depth=3 mcw=5 ss=0.8 cs=0.8 nrounds=330 cvAUC=0.84413
- AUC: **0.84232** (+0.00107) | ACC: 0.74702 | N_FEATURES: 21 | BEST_NROUNDS: 330 | DUR: 18.1s
- Decision: **KEEP**. Shallower trees (depth 3) + slower eta + more regularization (mcw=5) generalizes better than baseline depth=4/eta=0.05/mcw=1.

### Run 4 - 2026-04-19 - XGB + LASSO blend
- Hypothesis: diverse linear model (LASSO on standardized features) might blend with XGBoost for variance reduction.
- Config: XGB (same as Run 3) + cv.glmnet alpha=1 lambda.min, blend weight picked on inner val_inner.
- Result: blend picked best_alpha=1.0 (XGB only) on inner val, so blend collapsed to XGB.
- AUC: **0.84232** (tied) | ACC: 0.74702 | DUR: 20.4s
- Decision: **DISCARD** (no gain). LASSO underperforms standalone and adds no complementary signal here.

### Run 5 - 2026-04-19 - seed-bagged XGBoost (14 seeds)
- Hypothesis: averaging 14 XGBs with different seeds reduces variance on the holdout.
- Config: Run 3 best params, retrained 14x with seeds 1001..1014, each worker single-threaded, predictions averaged.
- AUC: **0.84226** (-0.00006) | ACC: 0.75000 | DUR: 21.2s
- Decision: **DISCARD**. XGBoost's internal row/col subsampling already mixes enough randomness; additional seed-bagging gives no measurable gain at 330 rounds.

### Run 6 - 2026-04-19 - monotonic constraints
- Hypothesis: encode domain priors (golddiff → win, deaths → loss) as monotone constraints on 18 features to reduce overfit.
- Config: `monotone_constraints` set on 16 +1 / 2 -1 / 3 zero features; rest of pipeline unchanged.
- CV best: eta=0.05 depth=5 mcw=5 nrounds=112 cvAUC=0.84385
- AUC: **0.84205** (-0.00027) | ACC: 0.75027 | DUR: 20.5s
- Decision: **DISCARD**. Constraints are too rigid; real relationships are noisy (e.g. grub_diff=4 dip) and monotone priors hurt marginally.

### Run 7 - 2026-04-19 - wider random hyperparameter search
- Hypothesis: adding gamma/lambda/alpha regularization and a bigger random search (48 combos) finds a better regularization point.
- Config: random 48-point search over eta/depth/mcw/ss/cs/gamma/lambda/alpha, 5-fold CV, parallel.
- CV best: eta=0.03 depth=3 mcw=3 ss=0.7 cs=0.8 nrounds=370 cvAUC=0.84440 (slightly higher than Run 3's 0.84413)
- AUC: **0.84224** (-0.00008) | ACC: 0.74946 | DUR: 43.9s
- Decision: **DISCARD**. Small CV bump did not transfer to test, suggesting we've hit the ceiling of single-model XGB on this feature set.

### Run 8 - 2026-04-19 - OOF stacking XGB + Random Forest
- Hypothesis: RF captures different non-linear patterns; stacked via logistic meta gives diverse blend.
- Config: 5-fold OOF preds from XGB (Run 3 params) + RF (400 trees, mtry=5); glm(y ~ xgb + rf) as meta. Full-data retrain for test.
- OOF: xgb=0.84309 rf=0.83492 blend=0.84303; meta weight xgb=4.86, rf=0.39 (meta already discounts RF).
- AUC: **0.84214** (-0.00018) | ACC: 0.74973 | DUR: 36.0s
- Decision: **DISCARD**. RF is a strictly weaker learner here and adds no complementary signal. Meta correctly shrinks its weight but can't manufacture gains.

### Run 9 - 2026-04-19 - add raw xp/cs/gold at 15
- Hypothesis: keep diffs AND absolute state; XGB is scale-invariant so raw values just add extra info without hurt. (Run 2 failed because of noisy interactions, not the raw values.)
- Config: add xp_at_15, cs_at_15, gold_at_15 to feature set (24 features total); rest unchanged.
- CV best: eta=0.03 depth=3 mcw=5 nrounds=264 cvAUC=0.84392 (slightly below Run 3's 0.84413)
- AUC: **0.84194** (-0.00038) | ACC: 0.74675 | DUR: 20.6s
- Decision: **DISCARD**. Raw absolute values add noise without signal — the diffs already capture the decision boundary.

### Run 10 - 2026-04-19 - game-level CV folds
- Hypothesis: default `xgb.cv` uses row-level random folds → Blue/Red of the same game may split across folds, causing subtle leakage that biases nrounds high. Group-level folds keep both rows of a game together, giving a more honest CV estimate.
- Config: custom `folds=` argument built by sampling 5 fold ids over unique gameids then mapping rows.
- CV best: eta=0.03 depth=3 mcw=1 nrounds=355 cvAUC=0.84409 (vs Run 3's 0.84413 — slightly lower, reflecting more honest CV)
- AUC: **0.84233** (+0.00001) | ACC: **0.75216** (+0.51pp vs Run 3) | DUR: 18.5s
- Decision: **KEEP**. AUC essentially tied, but accuracy jumped +0.5pp and the CV is methodologically cleaner.

### Run 11 - 2026-04-19 - add league as integer
- Hypothesis: 45 distinct leagues have different game paces/metas; `league_id` gives the model a per-league prior.
- Config: `league_id = as.integer(factor(league))`, added to feature set (22 features).
- CV best: cvAUC=0.84465 (+0.00056 vs Run 10), BUT test AUC dropped.
- AUC: **0.84146** (-0.00087) | ACC: 0.74811 | DUR: 20.0s
- Decision: **DISCARD**. Label-encoded 45-level categorical creates spurious ordinal splits. CV gain is overfit to fold-specific league distributions; doesn't generalize to held-out games.

### Run 12 - 2026-04-19 - Bayesian-shrunk team_winrate + games_reliability
- Hypothesis: raw wins/games is noisy for teams with few prior games (e.g. 1/1 = 100%). Shrinking toward 0.5 with prior of 10 pseudo-games gives a calibrated strength estimate. `games_reliability = min(games_before, opp_games_before)` lets the model discount unreliable winrate_diff.
- Config: `team_winrate = (wins + 5) / (games + 10)`; added `games_reliability`.
- CV best: eta=0.03 depth=3 mcw=5 nrounds=366 cvAUC=0.84434
- AUC: **0.84380** (+0.00147) | ACC: 0.75000 | N_FEATURES: 22 | DUR: 19.7s
- Decision: **KEEP**. Biggest single jump in the search. No leakage (still only prior games used). The shrinkage prevents the model from over-trusting fresh team winrates.

### Run 13 - 2026-04-19 - rolling team early-game strength diffs
- Hypothesis: team_winrate is binary-outcome-derived and coarse; a rolling avg of prior golddiffat15/xpdiffat15/csdiffat15 captures "how this team usually does at 15" directly.
- Config: added team_gd15_avg / team_xd15_avg / team_cd15_avg (shrunk by same PRIOR_N=10), built gd15_avg_diff / xd15_avg_diff / cd15_avg_diff vs opponent.
- AUC: **0.84382** (+0.00002) | ACC: 0.75054 (+0.05pp) | N_FEATURES: 25 | DUR: 20.3s
- Decision: **KEEP**. Gain is within noise but feature is principled and doesn't hurt; team_winrate already captures most of the signal via wins/losses, so extra early-game strength averages are mostly redundant.

### Run 14 - 2026-04-19 - wider hyperparameter search on new features
- Hypothesis: optimal hyperparams may have shifted with stronger team-history features.
- Config: 56-combo random search over eta/depth/mcw/ss/cs/gamma/lambda/alpha.
- CV best: eta=0.02 depth=3 mcw=1 ss=0.7 cs=0.6 nrounds=485 cvAUC=0.84418
- AUC: **0.84349** (-0.00033) | ACC: 0.75081 | DUR: 64.5s
- Decision: **DISCARD**. CV bump didn't transfer (classic overfit-to-CV). Run 13 grid (eta=0.03 depth=3 mcw=5) was already well-placed.

### Runs 15 & 16 - 2026-04-19 - PRIOR_N shrinkage sweep
- Run 15 (PRIOR_N=20): AUC **0.84179** (-0.00203) - too much shrinkage, winrate information lost
- Run 16 (PRIOR_N=5):  AUC **0.84327** (-0.00055) - too little shrinkage, noisy for new teams
- Decision: **DISCARD both**. PRIOR_N=10 is the sweet spot (~= mean # games per team / 3).

### Run 17 - 2026-04-19 - side-specific team winrate
- Hypothesis: some teams are Blue-side specialists; team's side-specific winrate should pick this up.
- Config: added `team_side_winrate` = rolling shrunk winrate grouped by (teamid, side); `side_winrate_diff = team_side - opp_side`.
- AUC: **0.84365** (-0.00017) | ACC: 0.75135 | N_FEATURES: 26 | DUR: 21.6s
- Decision: **DISCARD**. Redundant with overall team_winrate plus the existing `side` flag; halves the sample per team so estimates are noisier.

### Run 18 - 2026-04-19 - XGB + LASSO OOF blend (on Run 13 features)
- Hypothesis: LASSO previously lost to XGB on simpler features (Run 4), but with the enriched team-history features it may now be competitive.
- Config: 5-fold OOF preds from XGB (Run 13 params) + cv.glmnet alpha=1; blend weight swept over [0,1] on OOF.
- OOF: xgb=0.84337 glm=**0.84579**. Blend picks alpha=0.04 → almost pure LASSO.
- AUC: **0.84557** (+0.00175) | ACC: **0.75866** (+0.81pp) | N_FEATURES: 25 | DUR: 28.1s
- Decision: **KEEP**. Biggest single-run gain in the search. Key insight: the new team-history features are **linearly separable** for this task; XGBoost's non-linear capacity is actually hurting generalization slightly. LASSO's L1 shrinkage also prunes noisy features.

### Run 19 - 2026-04-19 - Elastic Net alpha sweep
- Hypothesis: mix of L1/L2 may beat pure LASSO.
- Config: sweep alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}.
- CV AUC essentially identical across all alphas (0.847 ± 0.00001). Picked alpha=0.3.
- AUC: **0.84555** (-0.00002) | ACC: 0.75595 | DUR: 40.7s
- Decision: **DISCARD**. Loss function is flat in alpha — no meaningful L1 vs L2 tradeoff for these features.

### Run 20 - 2026-04-19 - polynomial + interaction features for LASSO
- Hypothesis: LASSO is linear; adding squared terms (signed: x*abs(x)) + pairwise interactions gives it non-linear capacity that XGBoost captured.
- Config: +7 features (gd15_sq, xd15_sq, plate_sq, wr_sq, gd15_x_tower, gd15_x_wr, plate_x_tower). Same XGB + LASSO OOF blend.
- OOF: xgb=0.84344 glm=0.84586 blend=0.84589 (alpha=0.12). OOF AUC slightly up.
- AUC: **0.84511** (-0.00046) | ACC: 0.75568 | N_FEATURES: 32 | DUR: 36.6s
- Decision: **DISCARD**. OOF improved but test dropped → classic OOF overfit. Extra features hurt LASSO's lambda selection.

### Run 21 - 2026-04-19 - LASSO lambda.1se (more regularization)
- Hypothesis: lambda.1se is sparser / more conservative than lambda.min → less overfit, may beat 0.84557.
- Config: swap `s = "lambda.min"` → `s = "lambda.1se"` in both OOF preds and final LASSO.
- OOF: xgb=0.84337 glm=0.84380 (LASSO dropped a lot). Blend alpha=0.46 (much higher XGB weight — LASSO is weaker now).
- AUC: **0.84440** (-0.00117) | ACC: 0.75487 | DUR: 28.0s
- Decision: **DISCARD**. lambda.1se throws away useful features; lambda.min is already well-regularized by 5-fold CV.

### Run 22 - 2026-04-19 - bagged LASSO (B=20 bootstraps) on final fit
- Hypothesis: bag 20 bootstrap cv.glmnet fits and average test preds — reduces LASSO variance by ~1/sqrt(B).
- Config: replace final `glm_full` with mean of predictions from 20 cv.glmnet fits, each on a bootstrap sample. OOF procedure and XGB unchanged.
- AUC: **0.84565** (+0.00008) | ACC: 0.75703 | N_FEATURES: 25 | DUR: 32.5s
- Decision: **KEEP**. Gain is within noise but directionally correct — bagging is theoretically sound and doesn't overfit OOF.

### Run 23 - 2026-04-19 - bagged LASSO B=50
- Hypothesis: more bags → less variance → higher test AUC.
- AUC: **0.84560** (-0.00005) | ACC: 0.75812 | DUR: 39.6s
- Decision: **DISCARD**. Past B=20 the LASSO ensemble has converged; extra bags add noise from lambda-path instability.

### Run 24 - 2026-04-19 - alpha-grid bagged glmnet (alphas {0.2,0.5,0.8,1.0} × B=20)
- Hypothesis: averaging over a grid of elastic-net alphas diversifies the linear ensemble.
- AUC: **0.84554** (-0.00011) | ACC: 0.75785 | DUR: 45.5s
- Decision: **DISCARD**. Consistent with Run 19 (alpha sweep was flat). Lower alphas add ridge-like shrinkage that hurts slightly vs pure L1 on this feature set.

### Run 25 - 2026-04-19 - pairwise interactions via model.matrix(~.^2)
- Hypothesis: expand 25 → 325 features (25 raw + 300 pairwise). Let LASSO select; may find non-obvious interactions (e.g. gd15 × firsttower).
- OOF: glm dropped to 0.84560 (vs 0.84579 baseline). Interactions hurt OOF too.
- AUC: **0.84523** (-0.00042) | ACC: 0.75460 | DUR: 141.8s
- Decision: **DISCARD**. 300 interaction candidates overwhelm the lambda-path; LASSO selects spurious interactions that don't generalize.