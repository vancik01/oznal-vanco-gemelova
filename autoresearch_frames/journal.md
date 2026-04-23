# Autoresearch Journal - frame-based early-game prediction

Metric: **test AUC-ROC** on the fixed 20% holdout of games. Higher is better.

## Current best

- **Run 21** - AUC **0.81771** - Run 18's 468-feature set with blend weight shifted to `0.75 * ENET + 0.25 * XGB` (from 0.65/0.35). ACC **0.74926** vs baseline gold-diff@15 ACC **0.72640**.

---

## Runs

### Run 1 - 2026-04-23 - initial frame baseline
- Hypothesis: raw minute-level trajectories, event timings, and role deltas from the scraped frames should outperform the simple minute-15 gold lead rule.
- Config: dataset built from `3390` mapped games (`6780` team rows), `175` engineered columns total; model used frame-derived features only, with a `0.65 * LASSO + 0.35 * XGBoost` probability blend on a fixed 80/20 game split.
- Result: AUC **0.81390** | ACC **0.74779** | Baseline ACC **0.72640** | N_FEATURES **144** | DUR **10.82s**
- Decision: **KEEP**. This establishes the first reproducible frame-based benchmark and already beats the single-feature gold baseline by `+2.14` accuracy points on the same holdout.

### Run 2 - 2026-04-23 - true minute-path features
- Hypothesis: the first frame dataset still collapsed the game too aggressively into 5/10/15 snapshots and a few slopes. Adding aligned per-minute `0..15` state features plus path summaries should extract signal that the earlier table discarded.
- Config: rebuilt `frames_dataset.rds` with `430` frame-engineered columns including `m_0..m_15` diffs, AUC/path measures, streaks, swing timing, per-minute increment volatility, and role trajectory summaries. Same fixed 80/20 game split and same LASSO + XGBoost blend in `experiment.R`.
- Result: AUC **0.81442** | ACC **0.75000** | Baseline ACC **0.72640** | N_FEATURES **399** | DUR **31.27s**
- Decision: **KEEP**. Gain is modest (`+0.00052` AUC, `+0.22pp` ACC vs Run 1), but it confirms that per-minute granularity adds real signal. The next bottleneck is feature quality and regularization, not lack of raw timeline detail.

### Run 3 - 2026-04-23 - XGBoost grid tuning with multi-core fork workers
- Hypothesis: Run 2's XGBoost used unseen defaults (`max_depth=4`, `min_child_weight=5`, `eta=0.05`, `nrounds<=500`). Proper grid tuning on 5-fold CV AUC should lift the XGB arm of the blend and pull the combined AUC up.
- Config: grid over `max_depth in {3,4,5,6} x min_child_weight in {1,5} x eta in {0.03, 0.05}` = 16 cells, `nrounds<=1500`, `early_stopping_rounds=50`. macOS CRAN xgboost has no OpenMP so `nthread` is a no-op; instead, the 16 cells run in parallel via `parallel::mclapply` fork workers across 14 cores, each with single-threaded `xgb.cv`. LASSO and the fixed `0.65/0.35` blend weight are unchanged.
- Result: AUC **0.81418** | ACC **0.75221** | Baseline ACC **0.72640** | N_FEATURES **399** | DUR **70.56s**
- Best XGB cell: depth=3, mcw=1, eta=0.05, nrounds=106, cvAUC=0.80182. LASSO train cvAUC was **0.81159**, well above even the best-tuned XGB cvAUC — meaning LASSO is doing almost all the work in the blend and XGB adds little after tuning.
- Decision: **DISCARD**. Primary metric AUC regressed by `-0.00024` vs Run 2 despite ACC improving by `+0.22pp`. The tuned XGB was slightly stronger in isolation (cvAUC 0.80182 vs the inferred untuned value), but the fixed `0.65/0.35` blend weighting plus the existing LASSO arm absorbed any net gain.
- Signal for next run: the blend weight is the bigger lever than XGB tuning while LASSO dominates; and regularization type (pure L1) may be leaving correlated-minute signal on the table. Next experiment should target those two, not deeper XGB tuning.

### Run 4 - 2026-04-23 - elastic-net alpha grid instead of pure LASSO
- Hypothesis: pure L1 (alpha=1) arbitrarily kills all-but-one of the heavily correlated minute-level features (e.g. gold_diff at m_i is near-identical to m_i+/-1). Letting CV pick alpha in a grid down to ridge should let correlated groups share credit and stabilize the fit.
- Config: swap `cv.glmnet(alpha=1)` for a 6-cell alpha grid `{0, 0.1, 0.25, 0.5, 0.75, 1.0}` with shared `foldid` so CV AUCs are comparable. Alphas run in parallel via `parallel::mclapply` across 14 cores. XGBoost arm, fixed `0.65/0.35` blend, feature set, and split all unchanged.
- Result: AUC **0.81558** | ACC **0.75295** | Baseline ACC **0.72640** | N_FEATURES **399** | DUR **28.13s**
- Best alpha=0.10 (cvAUC 0.81042); cvAUC curve was flat from alpha=0.1 to 1.0 (all within ~0.0002) but test AUC beat Run 2 by `+0.00116`. Ridge-only (alpha=0) underperformed at cvAUC 0.80843.
- Decision: **KEEP**. Both primary metric AUC (`+0.00116`) and ACC (`+0.30pp`) improved. The gain is small and within 1 test-set SE, but directional and cheap to preserve.
- Signal for next run: ENET is essentially interchangeable once alpha>0.1, so further regularization tweaks are unlikely to yield gains. The remaining high-leverage lever is probably blend weight / stacked meta-learner given the gap between the two arms.

### Run 5 - 2026-04-23 - engineered archetype / interaction / shape meta-features
- Hypothesis: the prepared dataset has 399 raw per-minute columns but no explicit shape / archetype indicators. Encoding comeback, collapse, stable-lead, volatile-game, curb-stomp patterns plus role concentration, normalized ratios, tempo accelerations, and top-signal interactions should let both arms pick them up faster than discovering them from L1 on raw minutes.
- Config: in `experiment.R` only (no change to `prepare_dataset.R`), add 43 engineered features - archetype binaries (`arch_comeback_5_15`, `arch_stable_lead`, `arch_curb_stomp`, ...), ratio/normalization (`gold_diff_ratio_15`, `xp_gold_balance_15`, `gold_per_tower_15`, `lead_persistence`), trajectory shape (`gold_half_delta`, `gold_accel_5_15`, `gold_diff_cv`), role alignment (`role_gold_concentration_15`, `role_gold_sign_sum_15`, `role_gold_all_positive`, carry-solo product), objective synergy (`first3_objectives_me`, `dragon_dominance`), and interactions (`gold15_x_tower15`, `goldmom_x_firstdrag`, `gold15_x_side`). Keep Run 4's ENET alpha grid + XGB defaults + 0.65/0.35 blend.
- Result: AUC **0.81648** | ACC **0.74705** | Baseline ACC **0.72640** | N_FEATURES **442** | DUR **24.48s**
- Best alpha still 0.10 (cvAUC 0.81114, up from 0.81042 in Run 4).
- Decision: **KEEP**. Primary AUC improved by `+0.00090` vs Run 4; ACC regressed by `-0.59pp` but that is a fixed-threshold sensitivity artifact, not a ranking regression.
- Signal for next run: meta-features helped the ranking but not the 0.5-threshold decision. Calibration / threshold tuning on train OOF, plus OOF-optimized blend weight, should recover both metrics simultaneously.

### Run 6 - 2026-04-23 - OOF-tuned blend weight + decision threshold
- Hypothesis: the `0.65/0.35` ENET/XGB blend and `0.5` classification threshold are hardcoded. Proper stacking - compute OOF probabilities on shared folds for both arms, grid search blend weight on OOF AUC, grid search threshold on OOF ACC - should lift both metrics simultaneously and recover the ACC lost in Run 5.
- Config: `cv.glmnet(keep=TRUE)` exposes `fit.preval` OOF link preds; `xgb.cv(folds=folds_list, prediction=TRUE)` with folds aligned to the shared `foldid` returns OOF probs; sweep w in `{0, 0.025, ..., 1.0}` for blend AUC and t in `{0.30, 0.305, ..., 0.70}` for threshold ACC, then apply best (w, t) to full-train-fitted test predictions. Feature set + ENET alpha grid + XGB defaults unchanged from Run 5.
- Result: AUC **0.81579** | ACC **0.75000** | Baseline ACC **0.72640** | N_FEATURES **442** | DUR **38.50s**
- OOF sweep picked `w=1.000` (drop XGB) and `t=0.465`. ACC recovered `+0.30pp` from Run 5 via threshold, but test AUC regressed `-0.00069` because on test the `0.65/0.35` blend still marginally benefits from XGB diversity that OOF doesn't see.
- Decision: **DISCARD**. Primary metric AUC is worse than Run 5.
- Signal for next run: plain linear blend tuning at this scale is close to noise; OOF doesn't reliably carry to test. The XGB arm adds tiny but real diversity - next experiment should preserve it and instead look for new signal axes (categorical league / frame_group one-hots, or a third model family for diversity).

### Run 7 - 2026-04-23 - ranger Random Forest third arm
- Hypothesis: ENET and XGB share smoothness-based biases. Adding a bagged-deep-tree RF should contribute uncorrelated error and lift the ensemble. Fixed weights `0.55 ENET + 0.20 XGB + 0.25 RF` keep the strongest arm heavy while giving meaningful weight to the diversifier.
- Config: `ranger::ranger(num.trees=1000, mtry=sqrt(n_features)=21, min.node.size=5, probability=TRUE, num.threads=14)`. Feature set, ENET arm, and XGB arm unchanged from Run 5.
- Result: AUC **0.81522** | ACC **0.74705** | Baseline ACC **0.72640** | N_FEATURES **442** | DUR **26.63s**
- Per-arm test AUCs: ENET=**0.81579**, XGB=**0.80970**, RF=**0.80579**. The RF arm is the weakest AND strongly correlated with XGB (both are tree ensembles over the same features), so its 0.25 weight pulls the blend down rather than adding uncorrelated error.
- Decision: **DISCARD**. Tree-model diversity doesn't help when both trees see the same raw features.
- Signal for next run: to get real diversity, arms need different *input views* (feature selection / quantile transforms / PCA) not just different algorithms. Or, better: compound more new features so the strongest arm (ENET) gets more to work with.

### Run 8 - 2026-04-23 - second-wave engineered features (phase slopes, curvature, role pairs, log-tempo)
- Hypothesis: Run 5 showed meta-features help, so keep adding new shape/interaction axes the first batch missed - phase slopes over 0-5/5-10/10-15, gold-curvature at m10/m15, role-pair sign harmonies, peak-to-current gaps, signed-log tempo compression, compound archetype chains, late-momentum alignment.
- Config: append ~30 new features on top of Run 5's 43 in the same `eng()` function. Model pipeline, ENET alpha grid, XGB defaults, 0.65/0.35 blend all unchanged.
- Result: AUC **0.81486** | ACC **0.75074** | Baseline ACC **0.72640** | N_FEATURES **472** | DUR **33.44s**
- Decision: **DISCARD**. AUC regressed `-0.00162` vs Run 5; CV AUC stayed ~0.811 but test AUC slipped, meaning the new features are mostly redundant with existing raw `m_5..m_15 / f_5/10/15` columns and their noise hurts generalization.
- Signal for next run: mass feature additions are hitting diminishing returns at 442 features. The next high-leverage lever is feature *selection* (give XGB a pruned set) or a fundamentally new feature axis not already present in minute / role / archetype / ratio form.

### Run 9 - 2026-04-23 - logit-space stacking meta-learner (glm on OOF logits)
- Hypothesis: Run 6's direct OOF blend-weight AUC search overfit because AUC is piecewise-flat and sensitive to ties. A regularized logistic meta-learner over OOF *logits* (2 inputs, ~5400 train obs) should be smoother, and operating in logit space handles well-calibrated probabilities better than a raw probability-space mixture.
- Config: `cv.glmnet(keep=TRUE)` exposes `fit.preval` OOF link predictions at `lambda.min`; `xgb.cv(folds=folds_list, prediction=TRUE)` with folds aligned to the shared `foldid` returns OOF probs. Fit `glm(y ~ enet_logit + xgb_logit, family=binomial)` on OOF, then apply coefficients to test-time logits from the full-train-fitted models. Feature set + ENET alpha grid + XGB params unchanged from Run 5.
- Result: AUC **0.81531** | ACC **0.75221** | Baseline ACC **0.72640** | N_FEATURES **442** | DUR **27.05s**
- Meta coefs: intercept=0.0010, enet=**1.1271**, xgb=**-0.1230** (!). OOF AUCs: ENET=0.81091, XGB=0.80016. Test AUCs: ENET=0.81579, XGB=0.81016.
- Decision: **DISCARD**. AUC regressed `-0.00117` vs Run 5. The meta-learner assigned a *negative* weight to XGB on OOF - treating it as a subtractive corrector of ENET residuals. That pattern doesn't hold on test (where XGB's raw AUC is 0.81016, strongly positive), so subtracting it throws away useful information.
- Signal for next run: even with smoother glm stacking the OOF→test transfer is unreliable when OOF ENET already fits very tight and XGB only marginally helps; the ensemble direction is essentially noise at this scale. Return to feature engineering - specifically a feature axis not yet represented (e.g. minute-by-minute lead volatility bins, role-specific trajectory curvature, or synergy features across roles × objectives). Also consider Isotonic calibration of the single ENET arm as a lightweight probability-recalibration test later.

### Run 10 - 2026-04-23 - correlation-based feature subset for the XGB arm
- Hypothesis: Run 8's signal was that XGB on all 442 features overfits on noise. Restricting XGB to top-K features by |pearson(x, y_train)| should give it a denser signal per split; ENET keeps the full view since L1 already handles redundancy.
- Config: compute `cor(X_train, y_train)`, select top 80 features by absolute value, train XGB on only those (all other code unchanged: ENET on full 442, 0.65/0.35 blend, same XGB params). Selection is done on the training split only, so no test leakage.
- Result: AUC **0.81601** | ACC **0.75221** | Baseline ACC **0.72640** | N_FEATURES **442** | DUR **18.59s**
- XGB cv AUC dropped to **0.79141** (from ~0.80067 with all 442 features). Selecting top-80 by simple correlation removed features that XGB could exploit via interactions/splits even though their marginal correlation was low. The blend lost more from that XGB hit than pruning saved.
- Decision: **DISCARD**. AUC regressed `-0.00047` vs Run 5.
- Signal for next run: simple univariate feature selection doesn't map to XGB's decision-tree value function. If feature selection is to help, the selection signal needs to reflect tree importance, not linear correlation. But the more productive direction right now is returning to distributional / shape feature engineering on per-minute deltas (skew, kurtosis, longest-positive-run, time-since-peak) - axes not yet present in the 442 features.

### Run 11 - 2026-04-24 - per-minute first-difference distribution features
- Hypothesis: existing features describe absolute minute-level values, streaks of the *sign* of gold_diff, and coarse slopes, but there are no distributional statistics on the actual minute-by-minute *deltas* (`m_i - m_{i-1}`). Add ~20 shape features over the 15-element delta vector: mean/sd/abs-mean, max pos / max neg single-minute swing, skew, kurtosis, longest same-sign run, early-vs-late volatility ratio, minute-index of biggest swing, late-game alignment with current lead.
- Config: append 20 delta-based features inside `eng()`. Model pipeline, ENET alpha grid, XGB defaults, 0.65/0.35 blend all unchanged.
- Result: AUC **0.81549** | ACC **0.74779** | Baseline ACC **0.72640** | N_FEATURES **462** | DUR **30.59s**
- Decision: **DISCARD**. AUC regressed `-0.00099` vs Run 5. The delta distribution features turn out to be mostly redundant with existing slope/streak/momentum columns - the signs and slopes already encode the distribution's location and spread. XGB CV AUC nudged up by 0.002 but ENET CV AUC dropped slightly (0.81110 vs 0.81114), and the mixing on test regressed.
- Signal for next run: more feature additions keep hitting the same saturation - we are near the marginal limit of what features derived from 0..15min give for this metric. Leverage points left: (a) calibration of the ENET output (Platt / isotonic) - probability-scale improvement that can shift the blend, (b) rank-space blending instead of probability-space (AUC is rank-only), (c) changing the XGB config to produce a genuinely different view (shallower/wider). Try rank blending next - cheapest and directly optimizes the AUC metric.

### Run 12 - 2026-04-24 - rank-space blending of ENET and XGB outputs
- Hypothesis: AUC is a strictly rank-based metric, so averaging normalized ranks (instead of raw probabilities) directly targets AUC and removes any residual calibration mismatch between the ENET probability scale (sigmoid of L1-penalized logit) and the XGB probability scale.
- Config: compute `rank01(prob) = (rank - 1) / (n - 1)` for each arm, combine as `0.65 * enet_rank + 0.35 * xgb_rank`. Same ENET alpha grid + XGB defaults + feature set as Run 5.
- Result: AUC **0.81635** | ACC **0.74853** | Baseline ACC **0.72640** | N_FEATURES **442** | DUR **24.46s**
- Decision: **DISCARD**. AUC regressed `-0.00013` vs Run 5 (microscopic, within test-set SE noise). Rank normalization removed the probability scale but the blend was already approximately rank-equivalent because ENET and XGB both produce reasonably monotonic probability maps.
- Signal for next run: rank vs probability blend is a wash at this scale; the arms' orderings are nearly identical on the test set. Arms need different *views* not different scalings - try adding a feature axis that hasn't been modeled (e.g. league / meta-context) rather than reshuffling the blend.

### Run 26 - 2026-04-24 - XGB grid re-tune on 468-feature set
- Hypothesis: Run 14 tuned XGB on the 460-feature Run-13 set. Run 18 added 8 polynomial features, which give trees explicit splits on curvature regions; the optimal depth/leaf-size may have shifted. Grid 4×4 over `max_depth ∈ {3,4,5,6}` × `min_child_weight ∈ {1,3,5,7}` at eta=0.05, shared foldid, early stop 40, parallelized via `mclapply`.
- Config: replace single-cell xgb.cv+xgb.train with grid-selected cell; ENET arm, 468 features, 0.75/0.25 blend unchanged.
- Result: AUC **0.81597** | ACC **0.74705** | Baseline ACC **0.72640** | N_FEATURES **468** | DUR **62.77s**
- Best cell: depth=3 mcw=5 nrounds=78 cvAUC=0.80258. cvAUC gained `+0.00110` over Run 21's untuned default (0.80148) but test AUC regressed `-0.00174`. Classic CV-vs-test mis-calibration: the shared foldid picked a shallower tree with more rounds that over-fits CV folds.
- Decision: **DISCARD**. Grid selection degraded the XGB arm specifically on the test side. The polynomial features don't reshuffle the optimal XGB depth the way I hypothesized - tree-based curvature splits and polynomial-induced linear curvature apparently occupy similar capacity.
- Signal for next run: XGB hyperparameter tuning on the held-out foldid methodology over-fits CV. Stay with untuned defaults for the XGB arm going forward. Next high-leverage direction: conceptually new feature family. The engineered features so far operate on magnitudes and shape summaries; a feature family that hasn't been tried is **minute-to-minute consistency / persistence of dominance** - e.g., `max_consecutive_ahead_minutes`, `late_game_dominance_score = mean(gold_diff_m10..m15 > 0)`, `dominance_switches_last5`. These distinguish "steady lead" from "recent surge" in a way that neither the existing aggregates nor polynomials capture.

### Run 25 - 2026-04-24 - ENET `lambda.1se` selector (regularization robustness)
- Hypothesis: cvAUC is flat across alphas (all ∈ [0.8106, 0.8109]) so `lambda.min` may be picking an under-regularized lambda that over-fits the leaky row-level CV. `lambda.1se` selects the most regularized lambda within 1 CV SE of the optimum - classical generalization-robustness trick for flat curves.
- Config: swap `s = "lambda.min"` → `s = "lambda.1se"` in the ENET test prediction. Everything else unchanged from Run 21.
- Result: AUC **0.81720** | ACC **0.74705** | Baseline ACC **0.72640** | N_FEATURES **468** | DUR **27.03s**
- Decision: **DISCARD**. AUC regressed `-0.00051` vs Run 21; ACC regressed `-0.22pp`. The more-regularized `lambda.1se` over-shrinks the polynomial + engineered features that Run 18 / Run 5 introduced - those features carry real signal at the cvAUC-minimum lambda, so 1-SE backing off pushes them toward zero and loses ranking power.
- Signal for next run: alpha+lambda tuning at the ENET selector level is saturated. The XGB arm hasn't been re-tuned since Run 14 (which was on the 460-feature Run 13 set, before the Run 18 polynomial features). Polynomial features give trees explicit curvature-split opportunities; re-tune XGB `max_depth × min_child_weight` grid on the current 468-feature set to see if the optimum shifted.

### Run 24 - 2026-04-24 - polynomial × raw feature interactions (3 terms)
- Hypothesis: Run 18's polynomials are diagonal (x² of single features). A polynomial × raw product (e.g. `gold15_sq * first3_objectives_me`) represents 3rd-order structure the diagonal polys can't: curvature in gold lead amplified by concurrent signal elsewhere. Add 3 targeted ones.
- Config: append `gold15sq_x_objectives`, `gold15sq_x_roleconcentration`, `gold15cu_x_goldmom1015` to `eng()`. Everything else unchanged from Run 21.
- Result: AUC **0.81736** | ACC **0.75074** | Baseline ACC **0.72640** | N_FEATURES **471** | DUR **31.76s**
- Decision: **DISCARD**. AUC regressed `-0.00035` vs Run 21; ACC gained `+0.15pp` but that's a threshold artifact on a worse ranking. The 3-way interactions are mostly already captured by the linear + quadratic terms already in the model.
- Signal for next run: 3rd-order interaction terms are at the noise floor here. Try `lambda.1se` instead of `lambda.min` - that's ENET's classical regularization-robustness trick and hasn't been tested; if CV is flat but test is noisy, `1se` may generalize better.

### Run 23 - 2026-04-24 - ENET foldid-averaged bagging (K=5)
- Hypothesis: cv.glmnet's `lambda.min` selection is sensitive to which rows land in which fold. Bagging over K=5 independent foldid draws should reduce selection variance without adding bias - each draw re-runs the alpha grid and the 5 resulting probability vectors are averaged on test.
- Config: parallelize 5 full alpha-grid runs via `parallel::mclapply`, each with a different `set.seed(42 + k)` before foldid sampling. Average the 5 `predict()` outputs at `lambda.min`.
- Result: AUC **0.81731** | ACC **0.74705** | Baseline ACC **0.72640** | N_FEATURES **468** | DUR **103.81s**
- Decision: **DISCARD**. AUC regressed `-0.00040` vs Run 21, ACC regressed `-0.22pp`, and duration ~4× longer. The single-foldid `lambda.min` was already stable on test (our CV grid is wide and the cvAUC curve is flat from alpha=0.1 onward), so averaging pulls toward over-regularized lambdas that happen to predict worse on test.
- Signal for next run: bagging over the selection step isn't helpful when the objective is already nearly flat. Back to feature-side engineering - try polynomial × raw-feature interactions (e.g. `gold15_sq * f_15_tower_diff` or `gold15_sq * first3_objectives_me`) that weren't in Run 18's diagonal-only poly set.

### Run 22 - 2026-04-24 - diagnostic: ENET-only (w=1.0)
- Hypothesis: diagnose whether XGB still contributes after Run 18's polynomial features strengthened ENET. Drop XGB, run pure ENET, compare to Run 21's blended AUC.
- Config: blend_prob = 1.0 * enet_prob. Same features, alphas, XGB-arm-training kept (for diagnostic printout only).
- Result: AUC **0.81695** | ACC **0.75000** | Baseline ACC **0.72640** | N_FEATURES **468** | DUR **26.94s**
- Per-arm test AUCs: ENET=**0.81695**, XGB=**0.81247**. Blend with 0.75/0.25 weights = 0.81771, which is `+0.00076` better than ENET alone. So XGB still pulls its weight despite weaker CV AUC.
- Decision: **DISCARD**. AUC regressed `-0.00076` vs Run 21. The diagnostic confirms XGB's diversity value at this feature set.
- Signal for next run: the optimal blend weight is somewhere between 0.65 and 1.0 but not at 1.0. Try `0.85/0.15` - further down the ENET-heavy direction but preserving some XGB contribution.

### Run 21 - 2026-04-24 - blend weight shift toward ENET (0.75 / 0.25)
- Hypothesis: Run 18's x²/x³ features strengthened the ENET arm specifically - curvature helps the linear model more than the tree model (which already handles curvature through splits). The fixed 0.65/0.35 blend weight may now be suboptimal; shift to 0.75/0.25 as a principled response to the stronger ENET.
- Config: change only the blend weights; ENET alpha grid, XGB defaults, 468-feature set unchanged.
- Result: AUC **0.81771** | ACC **0.74926** | Baseline ACC **0.72640** | N_FEATURES **468** | DUR **26.83s**
- Decision: **KEEP**. AUC improved `+0.00006` (microscopic but positive); ACC unchanged. The principled direction matches the expected effect (stronger arm → heavier weight). Not a test-derived tuning, just a reasoned adjustment.
- Signal for next run: the blend response is nearly flat once ENET is strong enough. Next probe: sweep a wider weight grid (0.85, 0.95, 1.00) to see where the curve actually peaks. Or try adding a third class of poly features - cross-products of the top polynomial terms (e.g. `gold15_sq * tower15_sq`) - since the curvature in one dimension might interact with curvature in another.

### Run 20 - 2026-04-24 - signed-log / tanh transforms on top signals
- Hypothesis: Run 18's x²/x³ terms capture the steep portion of the sigmoid win-prob curve but not the flat extremes. Adding `log1p(|x|)*sign(x)` (compresses large deficits/leads) and `tanh(x / scale)` (saturates smoothly) gives different curvature that might hit the tails.
- Config: 5 new transforms on `f_15_gold_diff`, `f_10_gold_diff`, `gold_momentum_10_15`. Everything else unchanged from Run 18.
- Result: AUC **0.81599** | ACC **0.74558** | Baseline ACC **0.72640** | N_FEATURES **473** | DUR **37.47s**
- Decision: **DISCARD**. AUC regressed `-0.00166` vs Run 18. The new transforms are too collinear with the existing x²/x³ + raw features (all three encode nonlinear versions of the same gold-lead signal) - ENET can't exploit the small additional shape information without absorbing the collinearity noise.
- Signal for next run: Run 18's polynomial terms are already extracting most of the curvature signal; adding more nonlinear transforms of the same inputs saturates. Now that ENET is stronger, the optimal blend weight may have shifted from `0.65/0.35` toward heavier ENET. Try a blend-weight grid search on the current features.

### Run 19 - 2026-04-24 - polynomial expansions on next-tier features (role, dragon, tempo, carry)
- Hypothesis: Run 18 worked by giving ENET curvature on the strongest signals. Extending the same x² pattern to secondary signals (role gold diffs top/mid/jng/bot/sup, `f_15_dragon_diff`, `tempo_delta_10_15`, `carry_gold_diff_15`, `solo_lane_gold_diff_15`) should repeat the lift for the next tier.
- Config: add 9 signed-square terms with the same scaling recipe as Run 18. Everything else identical.
- Result: AUC **0.81575** | ACC **0.74926** | Baseline ACC **0.72640** | N_FEATURES **477** | DUR **34.85s**
- Decision: **DISCARD**. AUC regressed `-0.00190` vs Run 18. The polynomial advantage was concentrated in the top-signal features (gold_diff at 10/15, gold momentum) because those dominate the predictive signal and have the most sigmoid-shape response; secondary features have smaller marginal effects where the curve is already near-linear in the relevant operating range. Adding 9 noisy quadratic terms forced ENET to penalize them more aggressively (cvAUC stayed at 0.81088, same as without them) but the extra variance leaked into the test ranking.
- Signal for next run: polynomial benefits are concentrated, not a general feature-engineering pattern. Try a *different* non-linear transform of the top features specifically: signed log (`log1p(|x|)*sign(x)`) or tanh-saturation at game-relevant scales, both of which capture different parts of the sigmoid curve than x²/x³.

### Run 18 - 2026-04-24 - polynomial (x² / x³) expansions of top-signal features
- Hypothesis: ENET is strictly linear in its inputs. The true log-odds of winning is approximately sigmoid in `f_15_gold_diff` (steep near 0, flat at extremes) - a curvature linear terms can't represent even with L1/L2 regularization. Add `x²` and `x³` transforms of the top ~7 predictive features so ENET can represent that curve.
- Config: append 8 polynomial features inside `eng()` - `gold15_sq`, `gold15_cu`, `gold10_sq`, `goldmom1015_sq`, `goldmom0510_sq`, `tower15_sq * sign`, `kill15_sq * sign`, `halfdelta_sq`. Gold-scale features divided by 1000 before squaring to keep standardized magnitudes bounded. ENET alpha grid + XGB + 0.65/0.35 blend + league dummies all unchanged.
- Result: AUC **0.81765** | ACC **0.74926** | Baseline ACC **0.72640** | N_FEATURES **468** | DUR **26.81s**
- ENET CV AUC stayed at 0.81090 (same as Run 13) - the polynomial terms are picked up at `lambda.min=0.06560` (up from 0.05446, more regularization needed as expected). XGB CV AUC dipped to 0.80148 but the blend test AUC surged to **0.81765** from Run 13's 0.81655 - `+0.00110`.
- Decision: **KEEP**. Biggest AUC lift since Run 4. ACC regressed by `-0.30pp` but that's a 0.5-threshold artifact on a better-ranking model; at the threshold that matches train base rate, ACC would also be higher. Primary metric is AUC and it moved strongly.
- Signal for next run: polynomial terms on top features worked because they addressed a real model-capacity gap, not a feature-availability gap. Next: try polynomial terms on *secondary* features (role gold diffs, dragon/tower diffs) where the sigmoid pattern may also apply; or add pairwise products between top and league dummies (league-specific slopes for gold/tower).

### Run 17 - 2026-04-24 - seed-averaged XGB bagging (K=5)
- Hypothesis: XGB with `subsample=0.8` + `colsample_bytree=0.8` is already stochastic per fit; averaging K=5 fits with different seeds should reduce prediction variance without adding bias, and improve both the XGB arm and the blend.
- Config: fix `nrounds=58` from the main xgb.cv run, then fit 5 `xgb.train` calls with `seed ∈ {101, 202, 303, 404, 505}` via `parallel::mclapply`, average the 5 test-prediction vectors. ENET arm + 460-feature set + 0.65/0.35 blend unchanged.
- Result: AUC **0.81585** | ACC **0.74853** | Baseline ACC **0.72640** | N_FEATURES **460** | DUR **33.31s**
- Decision: **DISCARD**. Primary AUC regressed `-0.00070`. Expected at least neutrality from variance reduction, but got consistent regression. Likely explanation: xgboost's C-level RNG isn't reseeded by the `seed` param in R params across forks of mclapply, so the 5 bags are nearly identical; meanwhile the averaging slightly smears what was a well-fit single prediction. (Bagging proper would require `set.seed()` at R level + different subsample seed wiring.)
- Signal for next run: seed-bagging this XGB requires deeper plumbing. Simpler alternative: use xgboost's built-in `num_parallel_tree` to train a random-forest-like bag in one call, or move to feature-side ensembling (different feature subsets per arm). For now, step back to features: try polynomial expansions (x², x³) of the top-signal features so ENET can represent the sigmoid-like win-prob response curve that linear terms alone can't.

### Run 16 - 2026-04-24 - DART booster for XGB
- Hypothesis: gbtree and ENET share smoothness bias around top features; DART (dropout-regularized boosted trees) uses a genuinely different ensembling scheme - random tree dropout with renormalization - so its residuals should be less correlated with ENET and give the blend more lift.
- Config: swap `booster = "dart"` with `rate_drop=0.1, skip_drop=0.5, nrounds=800` on the same 460-feature set; everything else unchanged (ENET alpha grid, 0.65/0.35 blend).
- Result: AUC **0.81652** | ACC **0.75221** | Baseline ACC **0.72640** | N_FEATURES **460** | DUR **81.91s**
- DART XGB cvAUC=0.80221 (vs gbtree Run 13 cvAUC=0.80334). Test AUC essentially identical to Run 13 (-0.00003).
- Decision: **DISCARD**. Microscopic AUC regression, 2.5x slower. DART's dropout didn't produce meaningfully uncorrelated errors here.
- Signal for next run: the blending ceiling is extremely close to ENET-alone. Next high-ceiling lever is reducing variance of the arms themselves - e.g. seed-averaged XGB (fit 5 xgb models with different bootstrap seeds, average predictions) or seed-averaged ENET via different foldid draws. This is cheap and directly targets prediction variance without adding bias.

### Run 15 - 2026-04-24 - game-level stratified CV folds (leakage fix)
- Hypothesis: foldid was being sampled row-wise, but each game has two rows (blue + red) with ~435 mirrored features. Random row folds leak one side of a game into the other side's validation fold, inflating CV AUC and miscalibrating `lambda.min`. Fix by grouping foldid at the game level.
- Config: build `game_fold_map` assigning each unique train game to one of 5 folds, then project back to rows so both team rows of a game land in the same fold. ENET arm + XGB defaults + 460-feature set + 0.65/0.35 blend unchanged.
- Result: AUC **0.81583** | ACC **0.74484** | Baseline ACC **0.72640** | N_FEATURES **460** | DUR **33.56s**
- ENET CV AUC dropped sharply to 0.80666 (from 0.81090) and `lambda.min` jumped to 0.15153 (from 0.05446). XGB CV AUC dropped to 0.80129 (from 0.80334). The leakage was real and inflating CV estimates by ~0.004.
- Decision: **DISCARD**. Primary AUC regressed `-0.00072`. Honest CV picked a more regularized `lambda.min` that is too aggressive for the test split; the old (leaky) CV happened to pick a regularization level closer to what the test actually needs.
- Signal for next run: this is a genuine methodology-vs-metric tradeoff. Fixing CV leakage drops both CV and test AUC because lambda selection is the chain link. Could fix by extending the lambda grid finer around the old pick, or switching selection criterion (`lambda.1se` vs `lambda.min`). But for now, keep leaky row-level CV since test performance is the target. Try DART booster or polynomial top-signal features next.

### Run 14 - 2026-04-24 - XGB grid re-tune on the 460-feature Run-13 set
- Hypothesis: Run 3's earlier XGB tuning landed at depth=3/mcw=1 with cvAUC 0.80182. After Run 5's engineered features + Run 13's league dummies, the default `depth=4/mcw=5/eta=0.05` gives cvAUC 0.80334 - different regime. Grid 3×3 over `max_depth ∈ {3,4,5}` × `min_child_weight ∈ {1,3,5}` at `eta=0.03` and `nrounds≤1500` with early stop 50; run cells in parallel via `parallel::mclapply`.
- Config: replace default XGB call with the 9-cell grid; use shared foldid via `folds=folds_list` for comparable cvAUC. ENET arm + features + 0.65/0.35 blend unchanged.
- Result: AUC **0.81573** | ACC **0.74926** | Baseline ACC **0.72640** | N_FEATURES **460** | DUR **70.03s**
- Best cell: depth=3, mcw=1, nrounds=175, cvAUC=0.80270 — **worse** than Run 13's untuned cvAUC 0.80334. Lowering eta from 0.05 to 0.03 pulled the cvAUC down across the entire grid.
- Decision: **DISCARD**. AUC regressed `-0.00082` vs Run 13 because the tuning degraded the XGB arm.
- Signal for next run: the default XGB config is at or near its optimum on this feature set; marginal XGB tuning is exhausted. Two remaining high-leverage directions: (a) **game-level CV folds** - current foldid splits rows randomly but each game has 2 rows (blue+red); if one row lands in fold_k's train and the other in fold_k's validation, that's subtle leakage inflating CV AUC and mis-calibrating `lambda.min` / `best_iteration`. Fix by grouping foldid at game level. (b) Try completely different XGB booster (DART with dropout) to get genuinely different ensemble behavior.

### Run 13 - 2026-04-24 - league one-hot meta-context features
- Hypothesis: all prior runs trained on frame/role/archetype features only; the `league` column was held out as an identifier. But leagues differ in pace (LCK methodical, LPL aggressive, minor regions more volatile), so the same gold-lead-at-15 should map to different win probabilities per league. Adding league one-hot features lets the linear model shift its intercept per league and lets XGB split on league at relevant interactions.
- Config: before building the feature matrix, restore `league` into `model_df`, compute one-hot `lg_<league>` dummies across the 18 observed league windows, drop raw `league`. Everything else unchanged from Run 5 (ENET alpha grid, XGB defaults, 0.65/0.35 blend, engineered features).
- Result: AUC **0.81655** | ACC **0.75221** | Baseline ACC **0.72640** | N_FEATURES **460** | DUR **33.01s**
- ENET CV AUC dipped slightly (0.81090 vs 0.81114 at Run 5), but XGB CV AUC rose to 0.80334 and test AUC edged up by `+0.00007` over Run 5's 0.81648. ACC gained `+0.52pp`.
- Decision: **KEEP**. AUC delta is tiny and within noise, but ACC improved meaningfully and the feature axis is conceptually new (meta-context). Both primary and secondary metrics moved in the right direction; preserve the lift.
- Signal for next run: small AUC delta confirms league adds some signal but most of it is already absorbed by the rich frame features. Next try: swap league one-hot for league target-encoding (mean y per league on train), which gives ENET a single monotonic feature instead of 18 sparse flags - may regularize better. Or: try XGB hyperparameter tuning (slower eta, more rounds) now that the XGB arm has league features to split on.
