---
title: "Study Material - OZNAL Capstone Defence"
format: html
---

# Study Material - OZNAL Capstone Defence

Structured to follow `analysis.qmd` section by section. For each section: what we did, why, what methods are involved, what we didn't try, potential issues, and likely exam questions.



## Part 0: Vocabulary & Key Concepts

Read this first. Every term appears throughout the project and defence.



### Core Machine Learning Concepts

**Feature** - A single measurable property used for prediction. We have 14 features (e.g., `golddiffat15`). Other names: variable, predictor, column.

**Target (label)** - What the model predicts. Ours is `blue_win` (1 = Blue won, 0 = Red won). Binary target = two possible outcomes.

**Classification** - Predicting a category (Win/Loss), not a number. Our entire project is binary classification.

**Training set / Test set** - 80% for learning, 20% for evaluation. The model never sees the test set during training.

**Overfitting** - Model memorises training data noise, performs poorly on new data. Like memorising exam answers without understanding.

**Underfitting** - Model too simple to capture real patterns. Like studying one chapter for a ten-chapter exam.

**Bias** - How far off a model's average prediction is from truth. High bias = wrong assumptions (e.g., assuming linearity when it's curved).

**Variance** - How much predictions change when trained on different data samples. High variance = unstable model. CART is the clearest example in our project: change a few training games and the entire tree structure changes.

**Bias-Variance Tradeoff** - Simple models (Logistic Regression): high bias, low variance. Complex models (single decision tree): low bias, high variance. Best model balances both. In our project, Logistic Regression wins (84.2% AUC) because its "bias" (assuming linearity) happens to be correct.

**Hyperparameter** - A setting chosen before training (not learned from data). Examples: `k` in K-Nearest Neighbors (KNN), `mtry` in Random Forest (RF).

**Cross-Validation (CV)** - Technique to estimate performance without touching the test set. We use 5-fold CV: split training data into 5 parts, train on 4, evaluate on the 5th, rotate. Average of 5 rounds is the estimate.

**Standardisation (z-score)** - Transform features to mean=0, sd=1. Formula: `z = (x - mean) / sd`. Required for distance-based models (KNN) and coefficient-comparable models (Logistic Regression (LR)). Fit on training set only, applied identically to test set.



### Metrics

**Accuracy** - Correct predictions / total. Simple but misleading if classes are imbalanced.

**Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** - Our primary metric. Measures how well the model ranks wins above losses across ALL thresholds. AUC=0.5 is random guessing, 1.0 is perfect. Threshold-invariant.

**Receiver Operating Characteristic (ROC) Curve** - Plot of True Positive Rate vs False Positive Rate at every threshold. Good models bow toward top-left.

**Precision** - Of predicted wins, how many were actual wins?

**Recall (Sensitivity)** - Of actual wins, how many did we catch?

**Specificity** - Of actual losses, how many did we catch? Loss-side Recall.

**F1 Score** - Harmonic mean of Precision and Recall. Balances both.

**Confusion Matrix** - 2x2 table of True Positives, True Negatives, False Positives, False Negatives. All metrics above derive from it.



### Statistical Concepts

**Pearson Correlation (r)** - The default "correlation." Measures linear relationship between two variables. Range: -1 to +1. r=0 means no linear relationship. When anyone says "correlation" without a qualifier, they mean Pearson.

**Multicollinearity** - When features are correlated with each other (not just the target). Example: `golddiffat15` and `kill_diff_15` (kills give gold). Problems for LR (unstable coefficients) and Naive Bayes (NB) (violated independence). Tree models are immune.

**Variance Inflation Factor (VIF)** - Measures feature overlap with all others. VIF=1 means independent. VIF>10 is conventionally problematic. Our `golddiffat15` has VIF≈23.

**p-value** - Probability of seeing the observed result if there were no real effect. p<0.05 = "statistically significant." Statistical significance ≠ practical significance (with 9,200 games, even tiny effects are "significant").

**Chi-squared test** - Tests whether two variables are independent. We use it to test: "Is securing first tower independent of winning?" It compares observed counts against expected counts (what we'd see if the variables were unrelated). Large chi-squared = unlikely to be independent.

**Akaike Information Criterion (AIC)** - Balances model fit against complexity. Lower = better. Formula: `AIC = -2*log-likelihood + 2*k`. Used in forward stepwise to decide when to stop adding features.



### Regularisation

**Regularisation** - Penalty added to the loss function to prevent overfitting by discouraging large coefficients.

**L1 (Least Absolute Shrinkage and Selection Operator (LASSO))** - Penalty: `lambda * sum(|coefficients|)`. Drives weak coefficients to exactly zero → feature selection.

**L2 (Ridge)** - Penalty: `lambda * sum(coefficients²)`. Shrinks toward zero but never reaches it → no feature selection.

**Lambda (λ)** - Regularisation strength. 0 = no penalty. Large = strong penalty.

**lambda.1se** - Largest lambda within 1 Standard Error (SE) of the best. Sparsest model that is statistically indistinguishable from the best.

**Alpha (Elastic Net)** - Mix between L1 and L2. Alpha=0 is Ridge, alpha=1 is LASSO.



### Abbreviations

| Short | Full |
|---|---|
| AIC | Akaike Information Criterion |
| AUC | Area Under the Curve |
| CART | Classification and Regression Trees |
| CS | Creep Score (minions killed) |
| CV | Cross-Validation |
| EDA | Exploratory Data Analysis |
| GLM | Generalised Linear Model |
| KDE | Kernel Density Estimation |
| KNN | K-Nearest Neighbors |
| LASSO | Least Absolute Shrinkage and Selection Operator |
| LR | Logistic Regression |
| NB | Naive Bayes |
| PCA | Principal Component Analysis |
| pp | percentage points |
| RF | Random Forest |
| ROC | Receiver Operating Characteristic |
| SE | Standard Error |
| SMOTE | Synthetic Minority Over-sampling Technique |
| VIF | Variance Inflation Factor |
| XP | Experience Points |




## Section 1: Introduction (lines 25-82)

### What we do here
Define the problem, state two hypotheses, motivate the project.

### Key content
- **Problem:** Can we predict the winner of a professional LoL match from a 15-minute snapshot?
- **Two scenarios chosen:** Scenario 1 (Model Comparison - six classifiers) and Scenario 3 (Feature Selection - forward stepwise, LASSO, Elastic Net)
- **Two hypotheses:**
  - H1: Early-game metrics can reliably predict match outcome
  - H2: Bot lane (ADC) gold advantage is the most impactful role-specific factor

### What to know for defence
- Why these two scenarios? S1 tests whether early-game data has predictive signal and which model captures it best. S3 tests which features actually matter vs which are redundant. They complement each other: S1 finds the signal, S3 distills it.
- Why not Scenario 2, 4, or 5? S2 (parametric vs non-parametric) is partially covered within S1 (LR/NB are parametric, RF/CART/KNN are non-parametric). S4 (heatmap/tree visualisation) and S5 (dimensionality reduction) don't fit our data well - 14 features don't need PCA, and the classification boundary is linear.

### Potential question
**Q: "You say Scenario 1 needs 'three statistical/ML methods and two partitioning approaches.' How do you classify your six models?"**
- Statistical/Parametric: LR, NB (and arguably XGBoost as a statistical/ML method)
- Partitioning/Non-parametric: KNN, CART (and RF as an ensemble of partitioning trees)
- The six models exceed the minimum 5 (3+2) required



## Section 2: Dataset (lines 44-50)

### What we do here
Describe the data source, size, and structure.

### Key content
- Source: Oracle's Elixir 2025 season export
- Size: 120,636 rows x 165 columns, 10,053 unique matches, 45 leagues
- Structure: 12 rows per game (5 players + 1 team aggregate per side)
- 91.9% marked "complete" (have @10/@15/@20/@25 snapshots)

### What to know for defence
- Oracle's Elixir is NOT Kaggle. No pre-built solutions, no notebooks, no leaderboard. It's a journalist-maintained statistics site publishing raw CSV exports. The rubric bans "public data repositories with integrated solutions" - this has none.
- The 12-row structure required substantial reshaping (pivot_wider). This is genuine raw data work.

### Potential question
**Q: "Why did you drop 8.1% of rows as incomplete?"**
These rows have no early-game snapshots (@10, @15 columns are all NA). Without the 15-minute data, we cannot compute any of our features. There's no way to impute game-state data that was never recorded. The remaining 91.9% is still ~9,200 games - more than enough.



## Section 3: Related Work (lines 64-82)

### What we do here
Survey published @15-minute prediction studies to set expectations.

### Key content
- Published accuracy at 15 min: 76-78% across models and datasets
- Gold differential is universally the strongest feature
- Algorithm choice matters little in the published work - all land in 77±1% on pro data
- We note a framing gap: the naive "gold leader wins" rule already gives ~73%, so models only add 3-5 pp

### What to know for defence
- Our 75.5% accuracy falls within the published 76-78% band. This validates our approach.
- Our 84.2% AUC is higher than accuracy suggests because AUC measures the full probability ranking, not just the 0.5-threshold classification.
- We add two features absent from prior work: turret plate differential and void grubs (new 2025 objective).

### Potential question
**Q: "Your accuracy is lower than Spaargaren's 77.99%. Why?"**
Different season (2025 vs 2021), different meta. Tsang (2025) documents that gold diff's predictive coefficient has declined from >1.0 to 0.83 in 2025 - the season is genuinely harder to predict. Our AUC (0.842) is a better comparison metric than accuracy.



## Section 4: Data Loading & Restructuring (lines 85-222)

### What we do here
Load the CSV, filter to complete games, reshape from 2-rows-per-game to 1-row-per-game, remove mirrored columns.

### Key methods
- **`pivot_wider()`** - Turns two rows per game (Blue + Red) into one row with `blue_*` and `red_*` prefixed columns. Arguments: `id_cols = gameid` (identifies each game), `names_from = side` (blue/red becomes the prefix), `values_from = game_cols` (which columns to spread), `names_glue = "{side}_{.value}"` (naming pattern).
- **Mirror removal** - Signed differentials (`red_golddiffat15 = -blue_golddiffat15`) and binary flags (`red_firstblood = 1 - blue_firstblood`) are exact mathematical mirrors. We drop the Red copies. Independent counts (`turretplates`, `void_grubs`, `kills`) stay on both sides because they are NOT mirrors.
- **`stopifnot()`** - Sanity check that Blue diffs really are the exact negative of Red diffs. If this assertion fails, the code stops immediately.
- **`prop.test()`** - Tests whether Blue's win rate differs from 50%. Uses chi-squared internally.

### What to know for defence
- The proportion test confirms Blue wins ~53% (p < 0.001). This is **statistically** significant but **practically** small. The 3 pp edge comes from Blue having first draft pick and minor map asymmetries.
- We chose Blue perspective by convention (Blue picks first, published work uses Blue perspective). Flipping to Red perspective would produce identical model results with inverted signs - all features are signed differentials.
- At 53/47, no resampling is needed. The minority class (Red wins) has ~4,300 examples - plenty for any model. Our confusion matrices confirm balanced precision/recall across all 6 models.

### Potential questions
**Q: "The chi-squared test says the imbalance is significant. Why no resampling?"**
Statistical significance ≠ practical significance. With 9,200 games, even a 50.5/49.5 split would test as significant. The test has enormous power at this sample size. The practical magnitude (3 pp from 50%) is negligible. Resampling techniques like Synthetic Minority Over-sampling Technique (SMOTE) are designed for 90/10 or 95/5 ratios, not 53/47.

**Q: "Would predicting red_win instead of blue_win change results?"**
No. Every feature sign flips, every label flips. The model learns the same boundary, mirrored. The "minority class" would swap but both have thousands of examples. We chose Blue by convention because Blue has first draft pick, making it the natural reference side in LoL analytics.

**Q: "How does the proportion test work? Why chi-squared?"**
`prop.test()` tests whether Blue's observed win rate (~53%) differs from the expected 50% (fair coin). It uses chi-squared internally: compare observed wins/losses against expected wins/losses under the null (50/50), compute `χ² = Σ(observed - expected)² / expected`. Our chi² = 39.63, df = 1, p = 3.07e-10 - meaning a deviation this large is virtually impossible by chance. The 95% CI [52.3%, 54.3%] excludes 50%, confirming the advantage is real. But "real" doesn't mean "big enough to require resampling."

**Q: "What does pivot_wider do in your code?"**
It turns rows into columns. We had two rows per game (one Blue, one Red). `pivot_wider(id_cols = gameid, names_from = side, values_from = game_cols, names_glue = "{side}_{.value}")` merges them into one row per game with `blue_golddiffat15`, `red_golddiffat15`, etc. as separate columns. `id_cols` identifies each game, `names_from` provides the prefix (blue/red), `names_glue` defines the naming pattern.

**Q: "At what point does class imbalance become a problem requiring resampling?"**
There's no hard rule, but practical guidelines: 50-60/40 = no action needed. 60-70/30 = start monitoring, consider stratified splits. 70-90/10 = resampling worth trying. 90/10+ = strongly recommended. The real test is empirical: check if the model degenerates to always-predict-majority (our confusion matrices show it doesn't) and whether the minority class has enough absolute examples (ours has ~4,300).


## Section 5: Exploratory Data Analysis (lines 226-666)

### What we do here
Examine the data before modelling. Every plot has a purpose that feeds into a modelling decision.

### Sub-sections and their purpose

**5.1 Target distribution** - Shows 53/47 Blue/Red split. Confirms mild imbalance, no resampling needed, accuracy is valid.

**5.2 Descriptive statistics** - Features sorted by standard deviation. Gold/XP have sd in thousands, kills in single digits, binary flags at ~0.5. This directly motivates standardisation for KNN and LR.

**5.3 Correlation matrix & correlation with blue_win** - Uses Pearson correlation (the standard/default correlation). Shows `golddiffat15` is strongest (r≈0.43). Also reveals multicollinearity patterns (gold vs kills, Blue vs Red paired counts). This motivates LASSO/Elastic Net in Scenario 3.

**5.4 Per-outcome means** - For each feature, compare the mean when Blue wins vs when Blue loses. Gold diff at 15: +1500 in wins, -1500 in losses. First tower: 0.73 in wins, 0.33 in losses.

**5.5 Gold differential distribution** - Density plot showing clear Win/Loss separation but overlap around zero. The naive rule "gold leader wins" is scored here: 72.9% accuracy. This is the baseline every model must beat.

**5.6 First objective win rates** - Chi-squared tests of independence between each objective and winning. All reject independence (p << 0.001). First tower has the biggest effect (~73% win rate when secured), first blood the smallest (~57%).

Chi-squared works by comparing observed counts against expected counts (what we'd see if the objective were independent of winning). Formula: `χ² = Σ(observed - expected)² / expected`. Large chi-squared → the difference is too large to be random chance.

**5.7 Void grubs** - Near-monotonic win rate gradient from 0 grubs (~40%) to 6 grubs (~70%). Justifies engineering `grub_diff`.

**5.8 Side advantage** - Proportion test confirms Blue wins ~53% (p < 0.001, 95% CI excludes 50%).

**5.9 Role-specific gold differentials (H2 preview)** - Bot lane has highest correlation with `blue_win` (r=0.374). Preliminary evidence for H2.

**5.10 Feature distributions / scaling check** - Histograms showing 1000x scale differences between features. Directly motivates z-score standardisation.

**5.11 EDA Summary** - 12-point summary connecting every finding to a modelling decision.

### What we didn't try
- **Scatter plot matrix (pairs plot)** for all 14 features - could reveal non-linear relationships but would be 14x14 = 196 plots. Not practical.
- **Spearman correlation** instead of Pearson - Spearman measures monotonic (not just linear) relationships using ranks. Would be more robust to outliers. We used Pearson because it's the standard and our features are approximately linear with the target.
- **Formal normality tests** (Shapiro-Wilk) on feature distributions - we checked visually via histograms. Formal tests would reject normality at this sample size (they always do with 9k+ observations), so visual inspection is more informative.

### Potential questions
**Q: "What is Pearson correlation and is it the default?"**
Yes, Pearson is the default. `cor(x, y)` in R defaults to Pearson. It measures linear relationship between two variables, outputting r in [-1, +1]. When someone says "correlation" without specifying, they mean Pearson.

**Q: "You show both `blue_turretplates` and `red_turretplates` in the EDA but said you removed mirrors. Why?"**
Turret plates and void grubs are NOT mirrors. Each team has its own turret plates independently. Blue can have 3 plates and Red can have 7. The mirrors we removed were signed differentials (`red_golddiffat15 = -blue_golddiffat15`) and binary flags (`red_firstblood = 1 - blue_firstblood`) which are exact mathematical negations. In feature engineering later, the independent counts get collapsed into differentials (`plate_diff = blue - red`).

**Q: "What does the chi-squared test of independence against outcome mean?"**
It tests: "Is securing this objective independent of winning?" If independent, the win rate should be ~53% regardless of who got it. If not independent, the win rate changes depending on who secured it. Chi-squared compares the observed pattern against the pattern we'd expect if there were no relationship. All four objectives reject independence (p << 0.001), confirming they are genuine predictors.

**Q: "Your EDA is quite long. Isn't that 'bloated, unfocused EDA' the rubric warns against?"**
No - every plot connects to a specific modelling decision. The target distribution justifies no resampling. The scale check justifies standardisation. The correlation matrix motivates regularisation. The objective tests justify keeping the binary flags. The 12-point EDA summary makes these connections explicit.

**Q: "How does the chi-squared test work under the hood?"**
It compares observed counts vs expected counts (what we'd see if two variables were independent). Formula: `χ² = Σ(observed - expected)² / expected` across all cells. Large χ² means the observed pattern is too far from the expected "no relationship" pattern to be explained by chance. The chi-squared distribution (a known mathematical curve, shaped by degrees of freedom) then tells us the probability (p-value) of getting a χ² that large by pure luck.

**Q: "You didn't run formal normality tests like Shapiro-Wilk. Why?"**
Shapiro-Wilk and similar tests are unreliable at large sample sizes. With 9,200 observations, they reject normality for even trivially small deviations from a perfect bell curve - the test has too much statistical power. Every feature would test as "non-normal," making the result uninformative. Instead, the histograms on page 31 of the PDF show that the three key differentials (gold, XP, CS) are approximately symmetric and bell-shaped, while kills are right-skewed and void grubs are discrete. More importantly, normality of features is not an assumption of any model we use - LR assumes linearity in log-odds, not normality. For NB, we set `usekernel = TRUE` to avoid assuming normality altogether.

**Q: "Are the feature distributions actually normal?"**
Partially. The three continuous differentials (`golddiffat15`, `xpdiffat15`, `csdiffat15`) are approximately symmetric and unimodal (one peak) - close enough to a bell curve. But `blue_killsat15` is clearly right-skewed (starts at 0, long tail), and `blue_void_grubs` is discrete with peaks at 0 and 6. This doesn't matter because LR doesn't assume feature normality, tree-based models don't care about distributions at all, KNN uses distances not distributions, and we handled NB by selecting kernel density estimation instead of Gaussian.

**Q: "What does 'unimodal' mean?"**
One peak. The distribution has a single hump - most values cluster around one central value with a smooth falloff in both directions. Gold diff at 15 is unimodal (one peak near zero). If it were bimodal (two peaks), it would suggest the data has two distinct subpopulations that might need separate treatment.



## Section 6: Feature Engineering (lines 668-796)

### What we do here
Transform raw data into the 14-feature modelling matrix. Engineer derived features. Build per-role features for H2.

### Key decisions
- **Keep signed differentials, drop raw stats** - `golddiffat15` (the gap) carries the signal, not `goldat15` (the absolute total). This controls multicollinearity and focuses the model on what matters: who is ahead.
- **Drop @10 snapshots** - Any signal at @10 is captured at @15 (leads tend to grow). Adding both inflates VIF without raising AUC.
- **Engineer new features:**
  - `kill_diff_15 = blue_kills - red_kills` (combat advantage)
  - `assist_diff_15` (team fight participation)
  - `gold_efficiency_15 = gold / CS` (gold per minion killed - proxy for draft/macro quality)
  - `kda_15 = (kills + assists) / max(deaths, 1)` (combat efficiency ratio)
  - `plate_diff = blue_plates - red_plates` (turret economy, plates fall at 14:00)
  - `grub_diff = blue_grubs - red_grubs` (new 2025 objective)
  - `winrate_diff = blue_prior_WR - red_prior_WR` (pre-game team strength)
- **Rolling win rate** - Computed from prior games only: `wins_before / games_before`. Uses `cumsum(result) - result` to exclude current game. First-appearance teams default to 0.5. No data leakage - only past information used.
- **Role-specific features (H2)** - Per-role gold/XP/CS diffs at 15 for top/jng/mid/bot/sup. 30 additional columns. Blue-side player's stats minus lane opponent's stats.

### What we didn't try
- **Momentum features** (`diff@15 - diff@10` for gold/XP/CS) - tested in pilot fits but were linear combinations of features already present, adding VIF without signal.
- **Champion-specific features** - Would add ~160 binary columns. Out of scope; our model isolates in-game signal.
- **Interaction terms** (`golddiff * firsttower`) - LR already hits 84.2% AUC with main effects. XGBoost captures interactions automatically and doesn't beat LR, so interactions add no signal.
- **Time-windowed kill counts** (kills in 10-15 min window) - also linear combination of existing features.

### Potential questions
**Q: "How did you prevent data leakage in the rolling win rate?"**
`wins_before = cumsum(result) - result` subtracts the current game's result. `games_before = row_number() - 1` counts only prior games. Games are sorted by date. First-appearance teams get 0.5 (neutral prior). No future data enters the feature.

**Q: "Why drop @10 features but keep @15?"**
The @15 snapshot subsumes @10 - a team ahead at 10 is usually still ahead at 15 (leads grow, as shown in the EDA scatter plot). Adding both creates redundancy without new signal, inflating VIF. We tested this in pilot fits and confirmed no AUC improvement.



## Section 7: Train/Test Split (lines 798-826)

### What we do here
80/20 random split using `set.seed(42)`. Same game IDs used for both H1 and H2 so both hypotheses are evaluated on the same held-out games.

### What to know
- ~7,400 training games, ~1,848 test games
- `set.seed(42)` ensures reproducibility - anyone running the code gets the exact same split
- The Shiny app reconstructs this same split using the same seed

### What we didn't try - ISSUE FLAG
- **Temporal split** (train on early patches, test on late patches) - more realistic for deployment. See full discussion in earlier section.
- **Stratified split** (`createDataPartition()`) - ensures exact 53/47 ratio in both sets. At 9k+ games, random sampling achieves this naturally, but explicit stratification would be more rigorous.
- **Nested Cross-Validation** - outer CV for performance estimation, inner CV for hyperparameter tuning. More robust but 5x slower.

### Potential question
**Q: "Why random split and not temporal?"**
Random splitting for comparability with published work. Features are patch-invariant (gold, XP, CS mechanics don't change). Our results match published bands, validating the approach.



## Section 8: Scenario 1 - Model Comparison (lines 828-1720)

### What we do here
Train 6 classifiers on 14 features, compare on accuracy/AUC/precision/recall/F1. Then test H2 with enriched (44-feature) data.



### 8.1 Preprocessing (lines 843-861)

**Standardisation:** `preProcess(X_train, method = c("center", "scale"))` fits mean and sd on training data only. The same transformation is applied to test data. This prevents data leakage (test statistics cannot influence the scaler).

**Why:** KNN uses Euclidean distance (gold in thousands would dominate kills in single digits). LR needs comparable coefficients. Tree-based models don't need scaling but it's harmless.



### 8.2 VIF Check (lines 863-887)

**Method:** `VIF_j = 1 / (1 - R²_j)` where R²_j is the R-squared from regressing feature j on all others. Computed on training set only.

**Results:** `golddiffat15` VIF≈23, `kill_diff_15` VIF≈13 (high - kills translate to gold). Others 4-5 or below.

**Decision: Keep all features despite high VIF.** Reasoning in a callout box:
- VIF measures redundancy, not predictive value
- `golddiffat15` is the strongest predictor - dropping it would hurt everything
- Scenario 3's LASSO/Elastic Net will handle the redundancy properly
- Tree models are immune to multicollinearity

**Potential question:**
**Q: "VIF > 10 is the rule of thumb. Why did you violate it?"**
The rule says "investigate," not "drop." Dropping the strongest predictor because it correlates with a weaker one would be counterproductive. We explicitly defer the resolution to Scenario 3, where LASSO zeros out the redundant partner (`kill_diff_15`) while keeping the stronger one (`golddiffat15`). This is documented in the callout box.



### 8.3 CV Setup (lines 889-905)

Shared `trainControl` across all 6 models:
- 5-fold CV
- `classProbs = TRUE` (needed for AUC computation)
- `metric = "ROC"` (hyperparameters selected by AUC, not accuracy)
- `savePredictions = "final"` (stores out-of-fold predictions)

**Why AUC for training but accuracy for reporting?** AUC evaluates full probability ranking (better for hyperparameter selection). Accuracy is intuitive and comparable to published baselines (Spaargaren reports accuracy). At 53/47 balance, both metrics agree.

**Why threshold 0.5 for accuracy?** It's the natural Bayesian cutoff for binary classification with near-balanced classes. If P(Win) > 0.5, Win is more likely. At 53/47, the optimal threshold would shift by ~0.03 - negligible. Our balanced confusion matrices confirm 0.5 is appropriate.

**Potential questions:**
**Q: "You used AUC-ROC for training but defend accuracy as the main reported metric. How are they related?"**
AUC-ROC evaluates all thresholds (the full probability ranking). Accuracy uses exactly one threshold: 0.5. A model can have great AUC but poor accuracy if 0.5 is a bad threshold. In our case both agree because classes are near-balanced (53/47), making 0.5 the natural cutoff. We used AUC for training because it gives better hyperparameter choices (evaluates the whole probability output). We report accuracy because it's intuitive ("75.5% correct") and directly comparable to published baselines.

**Q: "Why is 0.5 the right threshold? Could tuning it improve accuracy?"**
0.5 is the natural cutoff for binary classification with balanced classes - predict whichever class has higher probability. At 53/47, the theoretically optimal threshold would shift to ~0.47, reclassifying maybe 20-30 borderline games in our 1,848-game test set. The accuracy change would be ~1 pp at most. Our confusion matrices show balanced precision and recall, confirming 0.5 is appropriate. Threshold tuning becomes important at 70/30 imbalance or worse.

**Q: "Why 5-fold CV and not 10-fold?"**
With ~7,400 training rows, each fold has ~1,500 rows - plenty for stable evaluation. 10-fold would give slightly less biased estimates but takes twice as long. The choice doesn't materially affect results. 5-fold is the standard default in the caret package.



### 8.4 The Six Models

#### Logistic Regression (LR)
- **What:** Linear classifier modelling log-odds of winning as weighted sum of features. Sigmoid function converts to probability.
- **In plain terms:** Draws a straight line in 14D space separating wins from losses.
- **Hyperparameters:** None (standard Generalised Linear Model (GLM))
- **Why it wins (84.2% AUC):** The signal is approximately linear. Gold/XP/CS diffs have near-linear relationship with log-odds of winning. When the assumption matches reality, LR captures it exactly with minimal variance.
- **Scaling:** Requires standardised input for comparable coefficients
- **Coefficient interpretation:** `exp(coefficient)` = odds ratio. Positive coefficient → increases P(Win).
- **Common misconception - LR does NOT assume normal features.** LR assumes linearity in log-odds, not normality of inputs. Features can be skewed (kills), discrete (void grubs), or binary (firstblood) - doesn't matter. People confuse this with Linear Regression (predicting a number), where the *residuals* (errors) are assumed normal. Logistic Regression uses maximum likelihood with a binomial distribution, not least squares with Gaussian errors.
- **What IS a problem for LR:** (1) Non-linear relationship between feature and log-odds. (2) Multicollinearity inflating standard errors (VIF issue). (3) Extreme outliers pulling the fitted line. Our data has issue #2 (addressed in S3), but not #1 or #3.
- **Why LR is not scale-sensitive in theory but we scale anyway:** LR will produce identical predictions with or without scaling. But without scaling, coefficients are on different scales (gold in thousands vs kills in single digits) and cannot be compared. Scaling makes the coefficient bar chart interpretable: "which feature has the biggest effect?"

#### Random Forest (RF)
- **What:** Ensemble of 500 decision trees, each trained on bootstrapped data with random feature subsets at each split. Averages predictions.
- **In plain terms:** 500 different trees vote; majority wins.
- **Hyperparameters:** `mtry` (features per split) tuned over {3, 5, 7, 10, 14}. `ntree=500` fixed.
- **Why `mtry = sqrt(p)` default:** sqrt(14) ≈ 3.7. The grid tests around this.
- **Result:** Matches but doesn't beat LR - confirms the signal is linear.
- **Importance metric:** Mean decrease in Gini impurity. Different from LR coefficients, so rankings can differ.

#### Naive Bayes (NB)
- **What:** Applies Bayes' theorem assuming features are independent given the class. Looks at each feature separately.
- **In plain terms:** Asks "how likely is this gold diff for a Win?" and "how likely are these kills for a Win?" independently, then multiplies.
- **Hyperparameters:** `usekernel` (TRUE = Kernel Density Estimation (KDE), FALSE = Gaussian). `adjust` (bandwidth). Best: `usekernel=TRUE, adjust=1.0`.
- **Why problematic here:** Independence assumption violated - gold and kills are correlated (VIF>10). NB double-counts their signal.
- **Why NB is the only model that improves with feature selection:** Removing redundant features makes the independence assumption more realistic.
- **Not scale-sensitive:** Each feature has its own density curve evaluated independently, so scale doesn't matter.

#### K-Nearest Neighbors (KNN)
- **What:** Finds the k most similar training games (Euclidean distance) and predicts majority class.
- **In plain terms:** Look up similar games, check if they were mostly wins or losses.
- **Hyperparameters:** `k` tuned over log-spaced grid from 5 to ~4*sqrt(n).
- **Why must standardise:** Distance-based. Without scaling, gold (range 20k) dominates kills (range 24).
- **Why large k works:** Curse of dimensionality in 14D means nearest neighbours aren't that near. Large k smooths noise.
- **No feature importance:** KNN has no native way to say which features matter.

#### Decision Tree / CART
- **What:** Recursive binary splitting. At each node, finds the best feature + threshold to separate classes. Controlled by `cp` (complexity parameter).
- **In plain terms:** A flowchart of yes/no questions learned from data.
- **Hyperparameters:** `cp` tuned over {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}. Small cp = deep tree, large cp = shallow.
- **Why weakest (79.7% AUC):** Single trees have high variance. Different data → different tree structure.
- **Why include it:** Shows the value of ensembling. Gap between CART (79.7%) and RF (83.0%) = the benefit of averaging 500 trees.

#### XGBoost (Extreme Gradient Boosting)
- **What:** Builds trees sequentially, each correcting the previous ensemble's errors. Uses gradient descent on the loss function.
- **In plain terms:** Team of students where each focuses on what the previous ones got wrong.
- **Hyperparameters:** `nrounds` {100, 300}, `max_depth` {4, 6}, `eta` {0.05, 0.1}. Fixed: `gamma=0`, `colsample_bytree=0.8`, `subsample=0.8`.
- **Why it doesn't beat LR:** The true signal is linear. XGBoost approximates a straight line using many small tree splits - each split adds variance without new signal.
- **Why include it:** Benchmark from published LoL literature (Lafrance & Grewal 2026). Makes results comparable.



### 8.5 Model Comparison (lines 1234-1441)

**Metrics table:** All 6 models on test set - accuracy, AUC, precision, recall, F1.

**Confusion matrices:** 6 side-by-side. All show balanced precision/recall - no model degenerates to always-predict-majority.

**ROC curves:** 6 overlapping curves. Top 5 cluster together (83-84% AUC). CART trails (79.7%).

**Feature importance consensus:** Cross-model rank comparison across LR (coefficients), RF (Gini), CART (Gini), NB (per-feature AUC), XGBoost (gain). KNN excluded (no native importance). `golddiffat15`, `xpdiffat15`, `csdiffat15` are top 3 in every model.

### Potential questions
**Q: "Why does LR beat XGBoost?"**
The signal is linear. LR captures a linear boundary exactly in one step. XGBoost approximates it with many tree splits, each adding variance. When the true relationship matches LR's assumption, LR wins. This is the classical bias-variance tradeoff.

**Q: "CART is much worse. Why include it?"**
CART is the single-tree analogue of RF. The 3 pp gap (79.7% vs 83.0%) directly demonstrates the value of ensembling. Without CART, we couldn't make this argument.

**Q: "Explain the bias-variance tradeoff using your project results."**
Bias = the error from the model's built-in assumptions (its theoretical ceiling). Variance = how much predictions change with different training data (its instability). Simple models have high bias, low variance. Complex models have low bias, high variance. In our project: LR has high bias (assumes linearity) but low variance (stable across samples) → wins at 84.2% AUC because the linearity assumption is correct. CART has low bias (can model any shape) but high variance (changes completely with different data) → loses at 79.7% because its flexibility is wasted and its instability hurts. RF reduces CART's variance by averaging 500 trees → 83.0%.

**Q: "What is bias exactly? Is it random?"**
No, bias is the opposite of random. It is systematic and deterministic - the same error every time, no matter what data you use. Bias is the model's built-in limitation. LR's bias is "can only learn linear boundaries." If the truth is curved, LR will miss it every single time, with any amount of data. That permanent miss is bias. Variance is the random part - it changes with each data sample.

**Q: "What is variance in the ML sense?"**
How much the model's predictions change when you train on different samples of the same data. Train CART on 100 different 80/20 splits → you get 100 different trees with different split points. That instability is variance. Train LR on 100 splits → you get nearly the same line every time. Low variance = stable and reliable.

**Q: "Which of your models is scale-sensitive and which is not? Why?"**
KNN and LR are scale-sensitive. KNN uses Euclidean distance - without scaling, features with large numeric ranges (gold: 20,000) dominate features with small ranges (kills: 0-24). LR produces identical predictions with or without scaling, but unscaled coefficients are incomparable. Tree-based models (RF, CART, XGBoost) are scale-invariant because they split on thresholds ("gold > 500?"), which works the same regardless of scale. Naive Bayes is also scale-invariant because it evaluates each feature independently against its own density curve.

**Q: "Why do different models give different feature importance rankings?"**
Each model measures importance differently. LR uses coefficient magnitude (linear weight). RF uses mean decrease in Gini impurity (how much each feature improves class purity across all splits). XGBoost uses split gain. NB uses per-feature AUC (how well each feature alone separates classes). These are fundamentally different measurements. Correlated features are especially affected: RF splits importance between `golddiffat15` and `kill_diff_15` randomly, while LR assigns most weight to gold. That's why we use the cross-model consensus rank rather than any single model's importance.


### 8.6 Enriched Models / H2 Test (lines 1443-1777)

**What:** Add 30 per-role features (gold/XP/CS diffs per role) to the 14 baseline features. Retrain all 6 models.

**VIF problem:** Role features sum to team totals (`top + jng + mid + bot + sup == team`), creating perfect collinearity. Solution: `findLinearCombos()` removes exact dependencies, then iteratively remove highest-VIF feature until all VIF ≤ 10.

**Result:** Mean AUC change across 6 models is negligible. Team-level totals already capture most signal. Bot lane ranks highest among role features, supporting H2 as an ordering but not as a strong claim that role data adds predictive value.



## Section 9: Scenario 3 - Feature Selection (lines 1779-2736)

### What we do here
Reduce 14 features to the minimal set that retains performance. Three methods compared.



### 9.1 Forward Stepwise (AIC) (lines 1783-2032)

**Method:** Start with intercept-only LR. Add one feature at a time, choosing the one that reduces AIC most. Stop when nothing helps.

**Result:** 10 of 14 features kept. Dropped: `assist_diff_15`, `firstblood`, `gold_efficiency_15`, `plate_diff`.

**Key analysis - significance evolution:** At each step k, refit LR on the first k features and track p-values. Shows how features' significance changes as partners enter. `golddiffat15` drops AIC by ~2,400 on its own - more than all others combined.

**Limitation:** Greedy - cannot undo admissions. Forward keeps `kda_15` at step 10 even though its final p-value is borderline (0.051).



### 9.2 LASSO (lines 2034-2153)

**Method:** Logistic Regression with L1 penalty. `cv.glmnet` tunes lambda with 5-fold CV, `type.measure = "auc"`.

**Lambda choice:** `lambda.1se` (sparsest model within 1 SE of best). This is more conservative than `lambda.min`.

**Result:** 7 features kept (the "consensus core"): `golddiffat15`, `xpdiffat15`, `csdiffat15`, `winrate_diff`, `firsttower`, `firstdragon`, `firstherald`. Dropped 7 features including `kill_diff_15` (absorbed by gold diff).

**Why LASSO zeros `kill_diff_15`:** Every kill gives ~300 gold. Once `golddiffat15` is in the model, the kill diff signal is redundant. L1 pressure pushes it to exactly zero.



### 9.3 Elastic Net (lines 2155-2300)

**Method:** L1 + L2 penalty. Alpha controls the mix (0=Ridge, 1=LASSO).

**Alpha tuning:** Grid from 0 to 1 by 0.05 (21 values). Fixed fold IDs so AUC values are comparable across alphas. Best: alpha=0.90 (near-LASSO).

**Result:** 8 features - LASSO's 7 plus `kill_diff_15`. The small Ridge component spares it.

**Key finding:** CV AUC is flat across all 21 alphas. The data cannot distinguish LASSO from Ridge at this sample size. The collinearity is not severe enough for Elastic Net's grouping property to matter.



### 9.4 Comparison & Validation (lines 2334-2736)

**Consensus:** 7 features selected by all 3 methods. 4 features never selected by any. `kill_diff_15` in the "2 of 3" band.

**Validation:** LR refitted on each subset. All within 0.3 pp AUC of the 14-feature baseline.

**Full grid:** All 6 S1 models x all 4 subsets (Full, Forward, LASSO, ElasticNet). NB is the only model that improves with selection (independence assumption becomes more realistic). All others are flat.

### Potential questions
**Q: "Why not Ridge as a standalone method?"**
Ridge shrinks but never zeros coefficients - it is not a feature *selection* method. The rubric requires "embedded feature-selection methods." Ridge is covered by the alpha=0 endpoint of our Elastic Net grid. It was tested and is indistinguishable from LASSO.

**Q: "Why not backward selection?"**
Forward maps to "which features carry signal?" (build up from nothing). Backward maps to "which features are redundant?" (pare down from everything). With 14 features, both converge. We chose forward for interpretability. The rubric lists "forward, backward, mixed" as options, not requirements.

**Q: "Why `lambda.1se` instead of `lambda.min`?"**
lambda.1se is the sparsest model within 1 SE of the best AUC. The AUC difference is by definition within noise. We trade ~0.3 pp AUC for halving the feature count - better interpretability and robustness.



## Section 10: Conclusions (lines 2742-2770)

### Key findings to memorise

**H1 confirmed:** Every classifier beats the 72.9% gold-rule baseline. Five of six exceed 83% AUC. LR is best at 84.2%.

**H2 qualified:** Bot lane has the highest role-specific correlation, but role-level data only adds ~0.5 pp AUC over team-level totals. H2 holds as an ordering (bot > other roles) but not as a strong claim that bot lane data improves prediction.

**S1 + S3 converge:** The same 7-feature consensus core emerges from both the S1 importance rankings and the S3 selection methods.

**Recommendation:** The 7-feature consensus model. Loses only 0.3 pp AUC, trains in seconds, produces interpretable coefficients.



## Section 11: Shiny App

### Tab-by-tab walkthrough

**Tab 1 - Data Explorer:** Load raw CSV (80MB, on demand). Browse raw data. Column overview.
- Demo: Show 12-row structure, filter to team rows, explain `datacompleteness`.

**Tab 2 - EDA:** Replicates key plots (target distribution, correlations, objectives, roles).
- Demo: Gold density plot → show Win/Loss separation. Objectives → first tower biggest swing.

**Tab 3 - Match Predictor:** Game picker (league → game → auto-fill sliders). All 6 models predict live. Win/Loss banner with actual result.
- Demo: Pick LCK game. Slide gold diff from +1500 to -1500. Watch probabilities flip. Toggle first tower. Note CART jumps discontinuously (threshold-based) while LR changes smoothly.

**Tab 4 - Hyperparameter Tuning:** Select model → adjust params → "Recompute live (5-fold CV)". RF: mtry slider. KNN: k slider with elbow curve. CART: cp input. NB: kernel toggle. XGB: 3 params. LR: odds ratio plot.
- Demo: RF mtry from 3→14, AUC barely changes. KNN k from 5→200, show elbow curve. Explain small k = high variance, large k = high bias.

**Tab 5 - Model Comparison:** Section 1: S1 models with ROC overlay and metrics matrix. Section 2: S3 feature selection with lambda slider showing features appear/disappear in real-time.
- Demo: All 6 models ROC overlay. Then LASSO with lambda slider - slide right to see features drop, AUC barely changes until 3-4 features.



## Section 12: Numbers to Know by Heart

| Metric | Value |
|--------|-------|
| Dataset (raw) | 120,636 rows x 165 columns |
| Games after filtering | ~9,200 |
| Train / test | ~7,400 / ~1,848 |
| Features (Scenario 1) | 14 |
| Features (enriched H2) | 44 |
| Features (S3 consensus) | 7 |
| Naive baseline (always Blue) | 53.3% accuracy |
| Gold-rule baseline | 72.9% accuracy |
| Best accuracy (LR) | 75.5% |
| Best AUC (LR) | 0.842 |
| Weakest AUC (CART) | 0.797 |
| Published @15 accuracy | 76-78% |
| LASSO 7-feature AUC | 0.839 |
| Blue side win rate | ~53% (p < 0.001) |
| Highest VIF | golddiffat15 (≈23) |
| CV folds | 5 |
| RF trees | 500 |
| Consensus core features | golddiffat15, xpdiffat15, csdiffat15, winrate_diff, firsttower, firstdragon, firstherald |
| Never selected features | assist_diff_15, firstblood, gold_efficiency_15, plate_diff |
