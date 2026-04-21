# OZNAL Project - Work Plan

## Team
- **Martin** - setup, data pipeline, classifiers, H2 analysis
- **Adriana** - partitioning models, feature selection, Shiny app, one-pager

## Deliverables
- Jupyter/R Markdown notebook (30 pts)
- Shiny app (10 pts)
- One-pager executive summary (5 pts)
- Oral defense (15 pts)

## Scenarios
- S1: Model Comparison + Feature-Space Partitioning
- S3: Feature Selection Comparison

---

## Phase 1 - Martin (setup, everyone depends on this)

### Project setup
- Git repo, .gitignore, R environment
- R Markdown notebook skeleton

### Data preparation
- Load 2025 CSV, filter `datacompleteness == 'complete'`
- Team rows (`position == 'team'`) for H1
- Player rows for H2 (role-specific differentials by `gameid + side`)
- Drop irrelevant columns, handle missing values

### Feature engineering (~46 features)
- Raw stats @10 min (15), raw stats @15 min (15)
- First objectives (4): firstblood, firstdragon, firstherald, firsttower
- Void grubs (2): void_grubs, opp_void_grubs
- Meta (2): side, firstPick
- Engineered (~8): gold/xp/cs momentum, kill pressure, gold efficiency, void grub diff, KDA@15
- H2: per-role gold/xp/cs differentials joined to team rows

### EDA
- Target distribution check (50/50)
- Correlation heatmap (early-game features vs result)
- Gold diff @15 distribution (wins vs losses)
- First objective win rates
- Side advantage analysis
- Role-specific gold diff correlations (H2 preview)
- Train/test split (stratified 80/20)

---

## Phase 2 - parallel work (independent after Phase 1)

### Martin: S1 partial - classifiers
- Logistic Regression
- Random Forest
- Naive Bayes
- Stratified k-fold CV for all three
- Metrics: accuracy, AUC-ROC, precision, recall, F1
- Confusion matrices
- ROC curves
- Explainability: LR coefficients vs RF importance vs Naive Bayes filter importance
- H1 vs H2 comparison: both feature sets through all 3 models, compare AUC-ROC
- Conclusion: which role's early-game stats matter most

### Adriana: S1 partial - partitioning + S3 full - feature selection
- **KNN** (k-nearest neighbors) - tune k, add to ROC overlay
- **Decision Tree CART** - visualize splits, add to ROC overlay
- **RFE** (Recursive Feature Elimination) - with LR as estimator
- **Forward Stepwise Selection**
- **LASSO** (L1 regularization)
- **Elastic Net** (L1 + L2)
- **Random Forest importance** (Gini / permutation)
- Accuracy vs number-of-features curve
- Overlap table / Venn: which features survive across methods
- Consensus features vs irrelevant features discussion

---

## Phase 3 - parallel work

### Martin: notebook finalization
- Integrate Adriana's sections into the main notebook
- Final discussion: synthesize S1 + S3 findings
- Make sure notebook is reproducible end-to-end

### Adriana: Shiny app + one-pager
- **Shiny app** (4 tabs):
  - Tab 1: Data explorer (filter by league, patch, role)
  - Tab 2: Model comparison (select model, see metrics)
  - Tab 3: Feature selection (toggle features, see performance)
  - Tab 4: Prediction (input stats, get win probability)
- **One-pager**: executive summary (written manually, no LLM)

---

## Dependency diagram

```
Martin: Setup -> Data Prep -> Feature Eng -> EDA -----> Martin: LR + RF + NB + H2
                                                  \
                                                   \--> Adriana: KNN + CART + Feature Selection + Shiny + One-pager
```

S1 -> pick mode
S3 -> feature engineering
Analysis -> 
TODO martin: Introduction to LoL - basic mechanics -> basic gold collection, ....
Is it ok to go S1 -> S3?  (feature engineering should be done before adding the models?)

