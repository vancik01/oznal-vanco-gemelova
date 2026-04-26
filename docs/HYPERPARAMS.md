# Hyperparameter Tuning - Per-Model Reference

**Date:** 2026-04-26
**Scope:** What each tuned hyperparameter means in `analysis.Rmd`, why current values were chosen, and alternative methods to find better values.

---

## Logistic Regression (`s1-logistic`)

- **Tuned: nothing.** `glm(family = binomial)` solves IRLS to maximum likelihood; there are no hyperparameters in the unpenalized form.
- **Why nothing**: by design - LR is the interpretable, no-knob baseline. If you wanted regularization, that's exactly what S3's LASSO/Elastic Net add.
- **If you wanted to tune**: switch `method = "glm"` → `method = "glmnet"` and tune (alpha, lambda) jointly. That collapses LR + L1/L2 into one model (this is essentially what S3 does).

---

## Random Forest (`s1-rf`, `s1-h2-rf`)

- **`mtry`**: number of features sampled at each split. Low mtry → trees decorrelate, lower variance, slightly higher bias. High mtry → trees converge to the same greedy splits (less ensemble benefit). Conventional default for classification: `floor(sqrt(p))` ≈ 4 for p=20, ≈ 7 for the H2 set of 50.
- **`ntree`**: number of bagged trees. Variance of the ensemble drops as 1/ntree, with diminishing returns past ~300-500 once OOB error plateaus.
- **Current grid**: `mtry ∈ {5, 7, 10, 15, 20}`, `ntree = 500`.
  - The grid spans from "near sqrt(p)" to bagging (mtry=p). Pragmatic and standard.
  - 500 trees is a safe stability point - doesn't need to be tuned if you check OOB error visually.
- **Better**:
  - Add `mtry ∈ {3, 4}` so the canonical sqrt(p) value is included.
  - Replace fixed ntree with `plot(rf_model$finalModel)` to verify OOB plateaued, or sweep ntree ∈ {200, 500, 1000}.
  - Random search over (mtry, ntree, nodesize, maxnodes) - often beats grid for RF (Bergstra & Bengio 2012).
  - Bayesian optimization via `mlrMBO` or `ParBayesianOptimization` - fewer evaluations needed than grid for RF's noisy CV surface.

---

## Naive Bayes (`s1-nb`)

- **`usekernel`**: FALSE = Gaussian density per feature per class; TRUE = nonparametric KDE. KDE wins when features are skewed/multimodal (gold/CS distributions are mildly skewed).
- **`adjust`**: KDE bandwidth multiplier on top of Silverman's rule. < 1 = bumpier, > 1 = smoother. Only meaningful when `usekernel = TRUE`.
- **`laplace`**: additive smoothing for zero-count categorical features. Irrelevant here - all features are numeric.
- **Current grid**: `usekernel ∈ {F, T}` × `adjust ∈ {0.5, 1.0, 1.5}` = 6 combos, `laplace = 0`.
  - 1.0 is the Silverman default; 0.5 and 1.5 bracket it.
- **Better**:
  - Wider `adjust` grid (e.g. 0.25, 0.5, 1, 2, 4) - optimal bandwidth varies more than ±50%.
  - Use a principled bandwidth selector (`bw.SJ`) instead of multiplying Silverman.
  - Power-transform skewed features (Box-Cox/Yeo-Johnson) before fitting Gaussian NB instead of switching to KDE - usually cleaner.

---

## KNN (`s1-knn`)

- **`k`**: number of neighbors voting. Low k = high variance, jagged boundary. High k = smoother, can underfit (k → n collapses to majority-class predictor). Odd k avoids ties in binary classification.
- **Implicit choices not tuned**: distance metric (default Euclidean), weighting (default uniform).
- **Current grid**: log-spaced k from 5 to `4 * sqrt(n) ≈ 360`, 12 values, forced odd.
  - sqrt(n) is the textbook midpoint (Hassanat 2014); log-spacing samples small + large k evenly; capping at 4·sqrt(n) avoids wasting fits at degenerate k.
- **Better**:
  - Switch to `method = "kknn"` (weighted KNN) and tune (k, distance, weighting) jointly - distance-weighted votes usually beat uniform.
  - Try Mahalanobis or correlation distance for correlated features (gold/XP/CS move together).
  - Random search over the same range (12 random k's often finds the optimum faster than 12 fixed points).

---

## CART (`s1-cart`)

- **`cp`** (complexity parameter): minimum relative improvement in Gini required for a split. Tree growth halts when no candidate split lowers impurity by ≥ cp · (root impurity). High cp → aggressive pruning, shallow tree. Low cp → keeps splits, deep tree.
- **Implicit, not tuned**: `maxdepth`, `minsplit`, `minbucket`.
- **Current grid**: `cp ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}`.
  - All below rpart's default of 0.01 - the grid is one-sided. The CV will pick the deepest tree allowed.
- **Better**:
  - Extend grid up to 0.05 so the CV can actually choose a pruned tree (the current grid forbids this).
  - Use rpart's built-in cost-complexity pruning: fit once with `cp = 0` then `printcp` / `prune` at the CV-optimal cp - faster than caret's grid.
  - Tune (`cp`, `maxdepth`, `minsplit`) jointly: a small `cp` with `maxdepth = 6` gives a different model from small `cp` unconstrained.
  - Replace with `ctree` (conditional inference trees, partykit) - splits chosen by permutation test, fewer tuning knobs.

---

## RFE (`s3-rfe`)

- **`sizes`**: candidate subset sizes to evaluate. RFE trains the full model, ranks features by `lrFuncs` importance (LR coefficient magnitude on standardized features), drops the weakest, repeats; CV picks the size that maximizes accuracy.
- **Selection metric**: hardcoded to **accuracy** by `rfeControl` default - inconsistent with S1, which uses ROC.
- **Current sizes**: `unique(c(5, 10, 15, 20, 25, 30, n_features))`. With `n_features = 20`, sizes 25 and 30 are above the total - rfe caps them, so effective grid is {5, 10, 15, 20}.
- **Better**:
  - `sizes = 2:n_features` for a complete curve; cheap with LR.
  - Switch the selection metric to ROC: `rfeControl(functions = lrFuncs, method = "cv", number = 5)` and override `lrFuncs$summary <- twoClassSummary`.
  - Replace LR-coef ranking with permutation importance for nonlinear consistency.
  - Use stability selection on top of RFE - bootstrap, keep features selected ≥ 60% of the time.

---

## Forward Stepwise (`s3-forward`)

- **Criterion**: AIC (penalty 2 per parameter). Adds the feature that most lowers AIC each step; stops when no addition helps.
- **No CV** - in-sample log-likelihood with the AIC penalty doing the regularization.
- **Why AIC**: classic stepwise default; fast.
- **Better**:
  - BIC instead of AIC: penalty `log(n) ≈ 9` for n ≈ 8000, much stronger pruning. Often gives a model closer to LASSO at lambda.1se.
  - Bidirectional stepwise (`direction = "both"`) - lets features be removed after being added.
  - CV-based forward (`leaps::regsubsets` followed by CV on each subset size) - selection metric matches the rest of S3.
  - Honestly: drop it. LASSO covers the same ground with shrinkage and avoids stepwise's known instability problems (Harrell 2015 §4.3).

---

## LASSO (`s3-lasso`)

- **`lambda`**: L1 penalty strength. λ=0 → full LR; λ → ∞ → intercept only. Coefficients hit zero exactly above their feature-specific threshold.
- **`alpha = 1`**: fixed (pure L1).
- **`lambda.1se` vs `lambda.min`**: 1se is the largest λ within 1 standard error of CV-min - sparser model, similar accuracy. `min` is the actual CV optimum.
- **Current**: `cv.glmnet` auto-generates 100-value lambda path (geometrically spaced), 5-fold CV, picks `lambda.1se`.
- **Better**:
  - `nfolds = 10` for lower-variance λ estimate (current 5-fold has visible noise in the choice).
  - Repeated CV: average λ across 5-10 repeats of the CV split.
  - Stability selection (`stabs` package): bootstrap LASSO at multiple λ, keep features selected ≥ π_thr of the time. Gives a formal error control.
  - Adaptive LASSO: weight each coefficient by `1/|β_init|` from a ridge fit - less biased than vanilla LASSO.

---

## Elastic Net (`s3-elasticnet`)

- **`alpha`**: L1/L2 mix. 0 = ridge (no sparsity, all coefs shrunk), 1 = LASSO. Mid-values keep correlated features at reduced coefficients instead of zeroing all but one.
- **`lambda`**: same as LASSO.
- **Current**: outer alpha grid `seq(0, 1, by = 0.05)` = 21 values; inner `cv.glmnet` per alpha picks `lambda.1se`; outer arg-min CV deviance picks alpha. Already includes a "1 SE flat curve" diagnostic that tells you whether the alpha pick is meaningful or noise.
- **Better**:
  - Use `caret::train(method = "glmnet", tuneLength = N)` for joint (alpha, lambda) CV in one pass - cleaner and reuses S1's `ctrl` object so you get ROC selection.
  - Repeated CV outer loop: alpha selection currently has the high variance of any 21-point CV pick.
  - Bayesian optimization over (alpha, log(lambda)) - fewer evaluations than 21 × 100, often hits a sharper optimum.

---

## RF Importance (`s3-rf-importance`)

- **Threshold**: cutoff above which a feature is kept. Currently `mean(importance)`.
- **Importance type**: Gini-based (`varImp` default for RF) - measures total Gini reduction attributable to each feature across all trees.
- **Why mean threshold**: data-adaptive (no extra hyperparameter), simple to defend.
- **Better**:
  - **Permutation importance** instead of Gini: `randomForest(..., importance = TRUE)` then `importance(rf, type = 1)`. Gini-importance is biased toward high-cardinality and continuous features (Strobl et al. 2007); permutation importance corrects this.
  - **Boruta** (`Boruta` package): adds shadow features (random permutations of original features) as a noise floor, formal test of "is feature X better than its shadow?". No threshold to set.
  - **Top-k as a tuning param**: treat the number of kept features as `k`, sweep k via CV, pick by ROC.

---

## Cross-cutting upgrades

If you want to tighten the whole tuning pipeline at once, the best returns are:

1. **Random search instead of grid** for RF/CART/KNN: caret's `trainControl(search = "random")` + `tuneLength = 30`. Bergstra & Bengio (2012) showed random beats grid when only a few hyperparameters matter (which is true for all the S1 models here).
2. **Bayesian optimization** (`ParBayesianOptimization`, `mlrMBO`) for RF and Elastic Net - fewer evaluations than grid/random, finds better optima on noisy CV surfaces. Worth it once your search space goes beyond 2 dimensions.
3. **Repeated k-fold CV** (`method = "repeatedcv", number = 5, repeats = 3`) reduces CV-noise in the tuning decision - currently a single 5-fold has visible variance in chosen `mtry`/`alpha`/`k`.
4. **Set explicit seeds** in `trainControl(seeds = ...)` so the tuning paths are reproducible (this also fixes the reproducibility flag from earlier).
5. **Nested CV** for honest reporting: outer CV evaluates the *full pipeline including tuning*, inner CV tunes. Currently each model reports a single 80/20 test number, which carries ±0.5-1 pp of split-noise that isn't quantified.

---

## Summary table

| Model | Tuned params | Grid size | CV | Selection metric |
|---|---|---|---|---|
| Logistic Regression | none | 1 | 5-fold | ROC |
| Random Forest (H1) | mtry | 5 | 5-fold | ROC |
| Random Forest (H2) | mtry | 4 | 5-fold | ROC |
| Naive Bayes | usekernel × adjust | 6 | 5-fold | ROC |
| KNN | k | 12 (log-spaced) | 5-fold | ROC |
| CART | cp | 5 | 5-fold | ROC |
| RFE | subset size | 4 effective | 5-fold | **accuracy** (inconsistent) |
| Forward Stepwise | none | 1 | none | **AIC** (in-sample) |
| LASSO | lambda | 100 (auto) | 5-fold | binomial deviance |
| Elastic Net | alpha × lambda | 21 × 100 | 5-fold | binomial deviance |
| RF Importance | none | 1 | none (reuses S1) | mean threshold |
