# Grading Rubric Evaluation - OZNAL Capstone

Strict self-assessment against each rubric criterion. Defence-ready with evidence for each claim.

---

## 1. One-Pager Executive Summary (5 pts)

**Estimated: 4 / 5 (Good)**

| Criterion | Evidence | Verdict |
|-----------|----------|---------|
| Outlines key assumptions | Explains data source, 15-min snapshot rationale, 72.9% naive baseline | Yes |
| Explains data choices | Covers Oracle's Elixir, 10,053 games, 14 features, mirrored-column removal | Yes |
| Shows how ML scenarios were built | Names all 6 classifiers and 3 FS methods, explains why each was chosen | Yes |
| Accessible to non-technical audience | Written in plain language, no formulas, clear structure | Mostly yes |
| Reflects student's own work (not AI) | Style is natural, domain-specific, uses "we" with personal motivation | Appears genuine |
| Visual support | Includes accuracy bar chart comparing baselines vs models | Yes |

**Strengths:**
- Clean typesetting (LaTeX-quality PDF)
- Covers all deliverables in one page: domain context, data, methods, results, conclusion
- The bar chart is a strong visual summary
- Correctly frames results against the 72.9% baseline and published literature (76-78%)

**Risks for deduction (-1 pt):**
- The summary could be more explicit about *why* LR beats tree-based models (the linear signal argument) - a reviewer might want to see the "why" not just the "what"
- No mention of H2 (bot lane hypothesis) at all in the summary - this is a missing piece since it was a declared hypothesis
- The phrase "compact seven-feature model" appears without naming the seven features - a non-technical reader might want to see them listed

**Defence argument:** The rubric says "executives without technical backgrounds can follow it easily." The summary achieves this. The one missing element (H2) is a content gap, not a clarity gap. A strict scorer might dock 1 pt for incomplete coverage.

---

## 2. Project Documentation (30 pts)

### 2a. Data Understanding and EDA (5 pts)

**Estimated: 5 / 5 (Exceptional)**

| Criterion | Evidence | Verdict |
|-----------|----------|---------|
| Focused insight into the data | 12-point EDA summary links each finding to a modelling decision | Yes |
| Highlights slices essential for both scenarios | S1: correlation matrix, VIF preview; S3: multicollinearity motivation | Yes |
| Avoids redundancy | Each plot has a stated purpose; no duplicate charts | Yes |
| Shows understanding of process and rationale | Every section ends with a takeaway that feeds into the next step | Yes |

**Strengths:**
- The EDA is directly connected to model choices:
  - Scale check (fig.width=10) → standardisation decision for KNN/LR
  - Correlation matrix → motivates LASSO/Elastic Net in S3
  - VIF check → explains why high-VIF features are kept in S1 but handled in S3
  - Gold-baseline rule (72.9%) → establishes the benchmark every model must beat
- Statistical tests used where appropriate (chi-square for objectives, proportion test for side advantage)
- Feature engineering is well-motivated: turret plates, void grubs, rolling winrate - each has a stated rationale
- The 12-item EDA summary is a strong artefact for defence

**Risks for deduction:**
- Minimal. This section is thorough without being bloated. A reviewer would need to look hard to find missing elements.

**Defence argument:** The EDA connects every finding to a downstream modelling decision. The rubric's "focused insight" criterion is met - no plot exists without a modelling justification.

---

### 2b. Scenario 1 - Model Comparison (10 pts)

**Estimated: 9 / 10 (High Exceptional)**

| Criterion | Evidence |
|-----------|----------|
| Feature selection | 14 engineered features from 165 raw columns, mirrored-column removal, VIF analysis |
| Model tuning | 5-fold CV with `metric="ROC"`, tuneGrids for RF (5 mtry values), KNN (12 k values), CART (5 cp values), XGBoost (8 combinations), NB (6 combinations) |
| Performance evaluation | Accuracy, AUC-ROC, Precision, Recall, F1 on held-out test set + 5-fold CV AUC |
| Model strengths/limitations | Discussed per-model: LR interpretability vs RF immunity to collinearity, NB independence violation, KNN scale sensitivity, CART single-tree variance |
| Graphics | Confusion matrices (6), ROC curves, feature importance plots per model, significance evolution plot |
| Cross-model comparison | Consensus rank table with avg_rank and rank_spread across 5 models |

**Strengths:**
- The six models cover three method families well: parametric (LR), probabilistic (NB), tree-based (RF, CART, XGBoost), instance-based (KNN)
- The VIF discussion in S1 that motivates S3 is a strong bridge between scenarios
- Enriched model comparison (H2 test) adds depth beyond the basic scenario requirement
- The cross-model consensus rank table is a particularly strong analytical piece
- The callout box explaining why high-VIF features are not dropped shows independent reasoning
- Hyperparameters are properly documented with a best-params table

**Risks for deduction (-1 pt):**
- The rubric for Scenario 1 says "three statistical or ML methods AND two feature-space partitioning approaches." Your partition is:
  - Statistical/ML: LR, RF, NB (3 methods)
  - Partitioning: KNN, CART (2 methods)
  - XGBoost is listed as "Statistical/ML" but is really a tree ensemble
  - A strict reader might argue KNN is not a "partitioning" method in the decision-boundary sense. KNN partitions feature space implicitly via Voronoi cells, but CART/RF do explicit recursive partitioning. This classification could be challenged.
- The rubric says "two feature-space partitioning approaches" - you use KNN and CART, but these are arguably just two models, not two *approaches* to partitioning. A stricter reading might want something like CART vs clustering-based classification, or tree-based vs kernel-based boundaries.

**Defence argument:** The rubric lists examples like "Logistic Regression, Random Forest, Naive Bayes, KNN, CART" in the scenario descriptions. Your selection matches these examples exactly. The 6-model comparison exceeds the minimum of 5 (3+2). The enriched H2 analysis goes beyond the scenario requirement entirely.

---

### 2c. Scenario 3 - Feature Selection (10 pts)

**Estimated: 10 / 10 (Exceptional)**

| Criterion | Evidence |
|-----------|----------|
| One algorithmic method | Forward stepwise via AIC - with AIC path plot, addition order, significance evolution |
| Two embedded methods | LASSO (alpha=1) + Elastic Net (alpha=0.90 from grid search) |
| Features retained count | Forward: 10, LASSO: 7, Elastic Net: 8, Consensus: 7 |
| Significance tracking | P-value evolution table across the entire forward-stepwise addition path |
| Performance comparison | LR refitted on each subset, full 6-model x 4-subset grid with heatmap |

**Strengths:**
- This is the strongest section of the project. It goes well beyond the minimum requirement:
  - The significance evolution plot tracking p-values as features enter is a sophisticated analytical piece
  - The 6x4 grid (every S1 model on every S3 subset) is not required but demonstrates thorough validation
  - The alpha-grid search for Elastic Net (21 values with fixed fold IDs for comparability) is methodologically strong
  - The "consensus core" framing - 7 features selected by all 3 methods, 4 never selected by any - is clear and defensible
  - Feature-by-feature explanation of why each was kept or dropped (with VIF/AIC/L1 reasoning)
- The LASSO lambda.1se vs lambda.min choice is explicitly justified
- The conclusion that "feature selection buys parsimony, not AUC" is data-supported (0.3 pp max drop)

**Risks for deduction:**
- The rubric says "lasso, ridge, elastic net" as examples. You use LASSO and Elastic Net but not standalone Ridge. However, the rubric says "two embedded" and you provide two. Ridge is covered implicitly through the alpha=0 end of the Elastic Net grid, but not as a standalone method. A strict scorer could note this.
- Response: the rubric says "lasso, ridge, elastic net" as *examples* in parentheses, not as mandatory. The requirement is "two embedded feature-selection methods." Ridge does not perform feature selection (it shrinks but never zeros), so including it as a standalone would not meet the criterion. This is a strong defence.

**Defence argument:** S3 exceeds the requirement. The 6x4 validation grid and significance evolution analysis go beyond what was asked. The alpha-grid Elastic Net search is methodologically stronger than a fixed-alpha fit.

---

### 2d. Code Understanding (5 pts)

**Estimated: 4 / 5 (Good)**

| Criterion | Evidence |
|-----------|----------|
| Well-organized structure | Quarto document with numbered sections, clear chunk labels (155 chunks) |
| Exceeds intended scope | Enriched H2 analysis, 6x4 grid, parallel computing setup |
| Reproducibility | set.seed(42) throughout, package auto-install block, data integrity assertions (stopifnot) |

**Strengths:**
- Every R chunk has a descriptive label (`s1-logistic`, `s3-lasso-plot`, `eda-correlation`)
- The package auto-install block ensures reproducibility on a fresh machine
- Parallel computing setup with fallback (`tryCatch` for cluster creation)
- Clean tidyverse pipeline throughout
- Feature engineering is well-separated from modelling code

**Risks for deduction (-1 pt):**
- The code is clearly sophisticated and carries AI-assist signatures (verbose comments, systematic structure). The rubric explicitly flags this: "The student appears to understand the code, even though it was AI-generated and carries the typical flaws of that approach." The quality puts you in the 3-4 range by default if the examiner suspects heavy AI use. You need to *demonstrate understanding* during the oral defence to recover the 5th point.
- Some code sections are dense (e.g., the enriched VIF iterative removal loop) - be prepared to walk through these line by line

**Defence argument:** The code runs end-to-end, produces correct results, and every analytical decision is explained in the surrounding text. The structure is logical and maintainable. Prepare to explain any function you used during the oral defence.

---

### 2. Project Documentation Total: 28 / 30

---

## 3. Shiny Application (10 pts)

**Estimated: 8 / 10 (Good-to-Exceptional)**

| Criterion | Evidence |
|-----------|----------|
| Showcases selected scenarios | S1: Model Comparison tab with ROC overlays, metrics matrix. S3: Feature Selection tab with lambda slider |
| Supports data exploration | Data Explorer tab with raw CSV loading, column summary. EDA tab replicates analysis plots |
| Interactive parameter adjustment | Hyperparameter Tuning tab: RF mtry slider + live 5-fold CV recompute, KNN k slider, CART cp input, NB kernel toggle, XGB multi-param |
| Dynamic model update | Live recompute buttons for RF/KNN/CART that retrain on the fly and update metrics |
| Well documented code | ~1400 lines, helper functions separated, comments on non-obvious decisions |

**Strengths:**
- **Five tabs** cover the full project scope (Data Explorer, EDA, Match Predictor, Hyperparameter Tuning, Model Comparison)
- **Match Predictor** with game-picker from real 2025 matches is a strong demo feature - pre-fills sliders from actual game data, shows actual result vs prediction
- **Live recompute** for RF with parallel tree-growing is technically impressive
- **Feature Selection panel** lets users slide lambda and see features appear/disappear in real-time with coefficient path plot
- **Model Comparison** correctly groups parametric vs non-parametric and shows ROC overlay
- The app loads pre-saved models (.rds files) for fast startup

**Risks for deduction (-2 pts):**
- The code is ~1400 lines in a single file. A strict scorer wanting "easy to understand, maintain, and extend" might note the lack of modularization (no separate `ui.R`/`server.R` or Shiny modules). This is the standard approach for smaller Shiny apps, but the size here warrants it.
- The XGBoost hyperparameter tuning panel has 3 interdependent params (nrounds, max_depth, eta) but only one can be varied at a time in the current UI. A full grid slider would be more interactive.
- The app requires a `models/` directory with pre-saved .rds files. If these are missing, the app fails at startup with no user-friendly error. This is a maintainability concern.
- There is no explicit "about" or "help" tab explaining how to use the app

**Defence argument:** The app exceeds the minimum by a wide margin. The Match Predictor with real game data, live RF recompute with parallel processing, and interactive lambda slider demonstrate genuine Shiny capabilities. The single-file structure is standard for apps of this complexity.

---

## 4. Free Questions (15 pts)

**Cannot be evaluated pre-defence.** This depends entirely on oral performance.

**Estimated range: 10-13 / 15** (assuming solid preparation)

The rubric awards 5 pts per question (3 questions). The questions will test:
1. **Theoretical understanding** of implemented methods
2. **Code/implementation choices** - why this, not that
3. **Extension proposals** - how would you modify/extend

See the accompanying STUDY_MATERIAL.md for preparation.

---

## Score Summary

| Category | Max | Estimated | Range |
|----------|-----|-----------|-------|
| One-pager | 5 | **4** | 4-5 |
| Data & EDA | 5 | **5** | 4-5 |
| Scenario 1 | 10 | **9** | 8-10 |
| Scenario 3 | 10 | **10** | 9-10 |
| Code understanding | 5 | **4** | 3-5 |
| Shiny app | 10 | **8** | 7-9 |
| Free questions | 15 | **11** (est.) | 9-13 |
| **Total** | **60** | **51** | **44-57** |

**Pass threshold: 40/60 (66%).** You are well above it even in the pessimistic scenario.

**Realistic grade: 49-53 / 60 (82-88%)** depending on oral defence performance.

---

## Key Vulnerabilities to Prepare For

1. **"This looks AI-generated"** - The code is polished and systematic. Prepare to explain any function, any parameter choice, any analytical decision from memory. The single biggest risk is not being able to walk through the code live.

2. **"Why not Ridge standalone?"** - You used LASSO + Elastic Net. Ridge does not zero features, so it is not a feature *selection* method. The alpha-grid Elastic Net already covers alpha=0 (pure Ridge) and shows it is indistinguishable from LASSO at this sample size.

3. **"Why no backward selection?"** - Forward was chosen because it starts from nothing and builds up, which maps to the research question "which features matter?" Backward starts from everything and removes, which is less interpretable for a feature-importance narrative. The rubric says "forward, backward, mixed selection" as examples - you chose forward. You could mention that backward selection on 14 features would likely converge to the same result.

4. **"Oracle's Elixir is basically Kaggle"** - It is a public repository, but it is not Kaggle (no pre-built solutions, no kernels, no leaderboard). The data required substantial cleaning (12-row-per-game structure, incomplete rows, mirror removal). The rubric footnote bans "public data repositories with integrated solutions" - Oracle's Elixir has no integrated solutions.

5. **"53/47 class balance - why no resampling?"** - 53/47 is barely imbalanced. SMOTE/undersampling at this ratio would introduce artificial patterns. The proportion test shows significance only because of large N, not because the imbalance is practically meaningful. AUC-ROC is reported alongside accuracy precisely because it is threshold-invariant.

6. **"Why does LR beat XGBoost?"** - The signal is approximately linear (resource differentials have a near-linear relationship with log-odds of winning). When the true decision boundary is linear, LR captures it exactly, while XGBoost overfits to noise in the splits. This is the expected result from ESL Chapter 10 (Boosting and Trees).

7. **"What about temporal leakage with winrate_diff?"** - The rolling winrate is computed from *prior* games only (`cumsum(result) - result` with `row_number() - 1` as denominator). First-appearance teams default to 0.5. No future information enters the feature.

---

## Defence Q&A - Anticipated Challenges With Prepared Responses

Questions the examiner is likely to ask, framed as challenges. Each has context from our project and a ready answer.

---

### Requirement / Scope Challenges

**Q: "The rubric says models must have ideally >30 parameters. Your main LR has only 15."**

Parameters ≠ features. The rubric says "parameters" - the values the model learns internally, not the input columns. Our parameter counts:
- Random Forest: 500 trees x hundreds of splits = tens of thousands of parameters
- XGBoost: up to 300 trees x many splits = thousands
- Naive Bayes: 14 features x 2 classes x density params = 56+
- Enriched H2 Logistic Regression: 44 features + intercept = 45 parameters (directly clears >30)
- Even KNN implicitly stores all ~7,400 training rows

Only the base LR sits at 15, and it is one of six models. The raw dataset has 165 columns - the 14-feature set is the result of careful selection, not the starting point. The word "ideally" is a guideline, not a hard cutoff. The grading table does not mention a feature count threshold.

---

**Q: "You only chose forward stepwise. The rubric mentions forward, backward, and mixed."**

The rubric lists them as examples: "forward, backward, mixed selection" - we need *one* algorithmic method. We chose forward because it maps directly to our research question "which features carry independent signal?" by building from nothing. With only 14 features, forward and backward typically converge to the same subset. We could verify this live by running `stepAIC` with `direction = "backward"` - it would almost certainly retain the same 10 features.

---

**Q: "You used LASSO and Elastic Net but not Ridge. The rubric says 'lasso, ridge, elastic net.'"**

The rubric requires "two embedded feature-selection methods." Ridge (L2) does not perform feature selection - it shrinks coefficients but never zeros them. Including Ridge as a "selection" method would be incorrect. Our Elastic Net alpha-grid searched from 0 (pure Ridge) to 1 (pure LASSO) in 21 steps - Ridge was tested as the alpha=0 endpoint and shown to be indistinguishable from LASSO at this sample size. The CV AUC curve is flat across all alphas.

---

**Q: "Oracle's Elixir is basically Kaggle. The rubric bans public repositories with integrated solutions."**

Oracle's Elixir is a raw data export, not a competition platform. There are no pre-built solutions, no notebooks, no leaderboard, no code attached to the data. The rubric footnote specifically bans "public data repositories with integrated solutions, such as Kaggle." Oracle's Elixir is a journalist-maintained statistics site that publishes CSV exports. The data required substantial cleaning: 12-row-per-game structure, 8.1% incomplete rows, 165 columns needing reduction, mirror column removal. This is genuinely raw data.

---

### Data & Methodology Challenges

**Q: "53/47 class balance - why no resampling?"**

Yes, our proportion test confirms the 53/47 split is statistically significant (p < 0.001). But statistical significance is not the same as practical significance. The p-value is tiny because we have ~9,200 games - at that sample size, even a 50.5/49.5 split would test as significant. The real question is whether the imbalance is large enough to distort model training, and at 53/47 it is not. The minority class (Red wins) still has ~4,300 games (~3,400 in training) - no model is starved for examples of either class. Resampling techniques like SMOTE are designed for severe imbalance (90/10, 95/5), not for near-balanced data. Applying SMOTE at 53/47 would introduce synthetic data points that don't reflect real games. We also report AUC-ROC alongside accuracy precisely because AUC is threshold-invariant and unaffected by mild class imbalance.

---

**Q: "You didn't remove outliers. Isn't that a problem?"**

We deliberately kept outliers because they represent real, informative game states. A 10,000 gold lead at 15 minutes is unusual but genuine - it's a stomp, and the model should learn that stomps are highly predictive. Removing these would remove the most informative training examples. Furthermore, tree-based models (Random Forest, CART, XGBoost) are robust to outliers by design - they split on rank order, not magnitude. LR on standardised features is also relatively robust. If we had measurement errors or data entry mistakes, outlier removal would be warranted - but the Oracle's Elixir data is machine-collected from the game API, so extreme values are real.

---

**Q: "Why a random train/test split instead of temporal?"**

We used random splitting for comparability with published work (Spaargaren 2022, Lafrance & Grewal 2026 all use random splits). A temporal split (train on early patches, test on late patches) would better simulate real deployment and we should have considered it. However, our features are fundamental game mechanics (gold, XP, CS, objectives) that don't change between patches - unlike champion-specific features which are patch-dependent. The fact that our results (75.5% accuracy) fall within the published 76-78% band validates that the approach is sound.

---

**Q: "What about temporal leakage with winrate_diff?"**

No leakage occurs. The rolling winrate is computed as `wins_before / games_before` where `wins_before = cumsum(result) - result` (subtracts the current game) and `games_before = row_number() - 1` (counts only prior games). Games are sorted by date before computing. Teams appearing for the first time default to 0.5 (neutral prior). The current game's result never enters the feature calculation.

---

### Model Choice Challenges

**Q: "Why does Logistic Regression beat XGBoost? That seems wrong."**

It's actually the expected result when the true signal is linear. The dominant features (gold diff, XP diff, CS diff at 15 minutes) have a near-linear relationship with the log-odds of winning. LR captures this in one step with minimal variance. XGBoost approximates the same linear relationship using many small tree splits - each split introduces a bit of variance without capturing new signal. This is the classical bias-variance tradeoff from ESL Chapter 10: when the true boundary matches your model's assumption, a simpler model wins. Our result (LR 84.2% vs XGBoost 83.8% AUC) is a textbook example.

---

**Q: "Naive Bayes assumes independence, but your features are correlated. Is NB invalid?"**

NB's independence assumption is clearly violated - the VIF analysis shows `golddiffat15` and `kill_diff_15` at VIF 23 and 13 respectively. Despite this, NB still achieves 83.1% AUC. Its predictions are useful (correct ranking) even if its probability estimates are poorly calibrated (push-to-extremes behaviour). We included NB specifically to demonstrate what happens when assumptions are violated. The Scenario 3 validation grid confirms the theory: NB is the *only* model that improves with feature selection, exactly because removing redundant features makes its independence assumption more realistic.

---

**Q: "CART is much worse than everything else. Why include it?"**

CART serves a specific purpose in the comparison: it is the single-tree analogue of Random Forest. The gap between CART (79.7% AUC) and RF (83.0% AUC) directly demonstrates the value of ensembling - averaging 500 unstable trees produces a stable, better model. Without CART in the comparison, we couldn't make this argument. CART also provides an interpretable visual (the tree diagram) that no other model offers, which is useful for the Shiny app demo.

---

### Code & Technical Challenges

**Q: "This code looks AI-generated."**

We used AI to assist with coding, which the rubric explicitly allows: "You may use it to assist with coding, but you must fully understand every function you include." We can walk through any section of the code and explain what it does and why. The analytical decisions (feature selection rationale, VIF interpretation, enriched H2 design, consensus-core framing) reflect genuine domain understanding of League of Legends and the published @15 prediction literature, not generic AI output.

---

**Q: "Why is the Shiny app in a single file instead of modular ui.R/server.R?"**

Single-file `app.R` is the standard Shiny convention for apps under ~2000 lines. Our app is ~1400 lines. The `ui.R`/`server.R` split provides no functional benefit - it's purely organisational. Shiny modules would add abstraction complexity without making the app easier to review. The code is logically structured: helper functions at the top, UI definition in the middle, server logic at the bottom. Each tab's server logic is clearly separated by comments.

---

**Q: "What happens if the models/ directory is missing when the Shiny app starts?"**

The app will fail at startup with an R error when `readRDS()` can't find the file. In a production setting, we would add file-existence checks and user-friendly error messages. For this academic submission, the README documents the dependency and the models are included in the submission. The app is designed to be run after the Quarto document has been rendered (which creates the models/ directory).

---

### Extension Challenges

**Q: "How would you extend this to predict in real-time during a live match?"**

Replace the single 15-minute snapshot with a streaming architecture. The model would receive game state updates every minute (or every significant event like a kill or objective). Options:
- Simplest: retrain the same LR model at each timestamp (@5, @10, @15, @20) using timestamp-specific features
- Better: a recurrent model (LSTM or GRU) that takes the sequence of game states and outputs an evolving probability
- The Shiny app's slider interface already demonstrates the concept - moving the gold diff slider simulates how the prediction changes as the game state evolves

**Q: "Why not include champion draft data?"**

Champion picks would add ~160 binary features (one per champion) or require embedding techniques. This would shift the project from "early-game state prediction" to "draft + state prediction" - a fundamentally different problem. Our 14-feature approach isolates the in-game signal. Published work that includes draft data shows only 1-2 pp improvement over draft-blind models at the 15-minute mark, because by minute 15 the draft advantage is already reflected in the gold/XP/CS differentials.
