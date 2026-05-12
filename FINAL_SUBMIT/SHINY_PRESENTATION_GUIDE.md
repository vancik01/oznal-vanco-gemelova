---
title: "Shiny App - Defence Presentation Guide"
format: html
---

# Shiny App - Defence Presentation Guide

## What the committee grades (10 points)

| Score | Criteria |
|-------|----------|
| **10-8** (Exceptional) | Showcases both scenarios, supports data exploration, visualizes how outputs change with parameter adjustments. Code is well-documented, easy to maintain and extend. |
| **6-7** (Good) | Does the above, but not all aspects are fully polished. |
| **3-5** (Borderline) | Works but has notable limitations, several important features missing, code disorganized. |
| **0-2** (Unacceptable) | Fails to run reliably, lacks features, code is difficult to work with. |

**Key rubric phrases to hit during your presentation:**

1. "showcases the selected scenarios" - show both S1 and S3
2. "supports data exploration" - show the Data Explorer and EDA tabs
3. "visualizes how outputs change with parameter adjustments" - this is the big one; live recompute is your best feature
4. "easy to understand, maintain, and extend" - mention code structure if asked

**Our estimated score: 8/10**

Why 8 and not higher:

- Match Predictor exposes only 9 of 14 features - the other 5 are silently filled with training means
- No live recompute for Scenario 3 (lambda slider is view-only, cannot retrain with different alpha or switch AIC to BIC)
- NB diagnostic plot is a generic calibration curve, not something NB-specific
- CSV upload expects the exact Oracle's Elixir schema - no way to load a different dataset format

Why not lower than 8:

- Both scenarios are clearly showcased
- Live 5-fold CV recompute for 5/6 models directly hits the rubric's "parameter adjustments update the model"
- Code is well-structured (~1400 lines, clean helper/server separation, parallel backend)
- EDA tab is thorough (distributions, correlations, chi-squared, role breakdowns)
- Match Predictor with real game lookup is a strong applied feature


## Presentation Order and Timing

Start the app before your time slot so it is already loaded when you begin. The startup precomputes forward stepwise, LASSO, and Elastic Net - this takes 10-15 seconds.

| Tab | Time | Priority |
|-----|------|----------|
| Data Explorer | 0:30 | Low - show briefly |
| EDA | 1:30 | Medium - hit the highlights |
| Match Predictor | 1:30 | High - interactive demo |
| Hyperparameter Tuning | 2:30 | **Highest** - live recompute is the killer feature |
| Model Comparison | 2:00 | High - shows both scenarios |
| **Total** | **~8 min** | |


## Tab 1: Data Explorer (30 seconds)

**What to show:** Click "Load CSV from current location", let the table load, scroll the columns.

**What to say:** "We start from Oracle's Elixir raw data - 80MB CSV with every pro match in 2025. Each game has 12 rows: 5 players plus 1 team aggregate per side. We can filter to team rows only."

### Possible questions

**Q: Why load on demand instead of at startup?**
A: The CSV is 80MB. Loading it lazily keeps app startup under 10 seconds - the models and feature selection precomputes are what matter for the analysis.


## Tab 2: EDA (1-2 minutes)

**What to show:**

1. Target distribution bar chart (53/47 split)
2. Correlation plot (gold diff at 15 is the strongest predictor)
3. Gold density by outcome (nice separation)
4. Objectives win rates with chi-squared p-values
5. Role correlations (bot lane gold diff matters most)

**What to say:** "The class balance is 53% blue wins, 47% red - close enough that we don't need resampling. Gold difference at 15 minutes has the highest correlation with winning. The chi-squared tests show all first objectives are statistically significant, though first blood has the smallest effect."

### Possible questions

**Q: Why not use the Shapiro-Wilk test for normality?**
A: With 9,200 observations, Shapiro-Wilk rejects everything - it is too sensitive at large sample sizes. We check normality visually from the density plots. Gold/XP/CS diffs are approximately normal.

**Q: What is the chi-squared test doing here?**
A: Testing independence between getting an objective (e.g. first dragon) and winning. Low p-value means they are dependent - getting the objective is associated with winning.

**Q: Why is 53/47 balanced enough?**
A: The common rule of thumb is that resampling helps when the minority class is under 30-40%. At 47%, both classes have plenty of representation and standard metrics like accuracy remain meaningful.


## Tab 3: Match Predictor (1-2 minutes)

**What to show:**

1. Pick a league (e.g. LCK) and select a real game
2. Show how the sliders auto-fill with actual match data
3. Point out the actual result banner vs prediction
4. Show all 6 models' probabilities + average
5. Tweak gold diff slider to show probabilities change live

**What to say:** "This is the applied side - you can pick any 2025 pro game and the app fills in the real 15-minute stats. All six models predict simultaneously. Watch what happens when I push gold diff from +1500 to -1500 - every model flips."

**This is your strongest demo moment.** Move the gold diff slider dramatically. The committee loves seeing live interactivity.

### Possible questions

**Q: Why does KNN sometimes disagree with other models?**
A: KNN classifies by distance in 14-dimensional space. A small input change can flip which neighbors are closest, making it less smooth than LR.

**Q: What is the "Average" row?**
A: Simple mean of all 6 models' probabilities. Not an ensemble - just a quick sanity check.

**Q: Why only 9 sliders when you have 14 features?**
A: We exposed the most interpretable features (gold/XP/CS diffs, objectives, grubs, winrate). The remaining 5 (kill_diff, death_diff, assist_diff, turret plates, game length) are filled with training-set averages as neutral defaults. This is in the `build_input_row` function.


## Tab 4: Hyperparameter Tuning (2-3 minutes)

**This tab directly demonstrates "parameter adjustments update the model" which is the core rubric requirement. This is the most important tab.**

**What to show:**

1. Start with **Random Forest** - show mtry slider, the pre-tested grid, then hit "Recompute live" with a different mtry value. The progress bar runs 5-fold CV in real time.
2. Switch to **KNN** - show the elbow plot (AUC vs k). Recompute at a very small k (e.g. 3) and a very large k (e.g. 50) to show the orange live dots appearing on the elbow curve.
3. Show **CART** - the decision tree visualization is visually impressive.
4. Show **Logistic Regression** - the odds ratio plot with confidence intervals.

**What to say:** "The Hyperparameter Tuning tab lets me recompute any model with different parameters, live, using parallel 5-fold cross-validation. For Random Forest, I can change mtry from 3 to 7, hit recompute, and see the new AUC in about 15 seconds. For KNN, the elbow plot shows how AUC degrades at low k (overfitting) and high k (underfitting)."

### Possible questions

**Q: What happens when you pick a value not in the pre-tested grid?**
A: The app runs a full fresh 5-fold CV with parallel workers. The new point appears as "Live" in the results table and on the plots.

**Q: Why 500 trees for RF?**
A: Standard default. More trees reduce variance but with diminishing returns. 500 is the widely-accepted sweet spot.

**Q: What does mtry control?**
A: How many features are randomly considered at each split. Lower mtry = more randomness between trees = lower correlation between trees = better ensemble diversity.

**Q: Why can't I tune Logistic Regression?**
A: Standard GLM has no regularization hyperparameter. The penalty is in Scenario 3 (LASSO/Elastic Net). In Scenario 1, LR is our interpretable baseline.

**Q: What if live recompute is slow during the demo?**
Say: "The parallel backend splits 500 trees across N workers in 5-fold CV - this is the actual training pipeline, not a shortcut."

**Q: Why does the NB plot show a calibration curve instead of something NB-specific?**
A: Naive Bayes does not have a single interpretable visualization like odds ratios or a tree. The calibration curve shows whether predicted probabilities match actual outcomes - useful for checking if NB's independence assumption causes miscalibration.


## Tab 5: Model Comparison (2-3 minutes)

### Section 1 - Scenario 1 (method comparison)

**What to show:** Check all 6 models. Show the ROC overlay - all curves bunched near AUC 0.84. Show the metrics matrix. Show the top-5 features per model faceted plot.

**What to say:** "All six models perform very similarly - AUC ranges from about 0.83 to 0.84. This is a key finding: the data is linearly separable enough that even the simplest model (Logistic Regression) matches the complex ones. The feature importance panel shows gold diff at 15 is the top feature for every single model."

### Section 2 - Scenario 3 (feature selection)

**What to show:** Switch between None/Forward/LASSO/Elastic Net. **Slide the lambda slider** from left to right - show features dropping out on the coefficient path plot. Show the retained features bar chart.

**What to say:** "Here I can slide lambda to control how aggressively features get dropped. At the default 1-SE lambda, LASSO keeps 7 features with nearly the same AUC as all 14. The path plot shows each feature's coefficient trajectory - you can see gold diff surviving longest."

**This is where you demonstrate Scenario 3 understanding.** Slide lambda slowly and narrate: "As I increase lambda (slide right), the penalty grows and features with less predictive value get zeroed out first."

### Possible questions

**Q: Why do all models perform so similarly?**
A: The relationship between early-game stats and winning is approximately linear. When the true pattern is linear, adding model complexity (trees, boosting, neighbors) does not help - it just adds variance without reducing bias.

**Q: What is the difference between LASSO and Elastic Net here?**
A: LASSO uses only L1 penalty (forces coefficients to exactly zero). Elastic Net mixes L1 and L2 (Ridge). In our data they produce nearly identical results because the collinearity (VIF 13-23 on two features) is moderate - not severe enough for the L1/L2 distinction to matter.

**Q: Why does forward stepwise keep a different number of features than LASSO?**
A: Different selection mechanisms. Forward stepwise uses AIC (information criterion - adds features until adding more does not improve the model). LASSO uses L1 penalty (shrinks small coefficients to zero). They converge on a similar core set but the exact cutoff differs.

**Q: What is the 1-SE lambda rule?**
A: After cross-validation finds the lambda with the best AUC, the 1-SE rule picks a larger lambda (more penalty) whose AUC is within 1 standard error of the best. This gives a simpler model with nearly the same performance - a parsimony principle.


## How the Shiny App Works Technically

If the committee asks "explain how this app works" or "what Shiny concepts are used here," these are the key patterns.

### The core idea

Shiny takes normal R code (the same `ggplot()`, `dplyr`, `cor()` you use in the QMD) and renders it into a browser. The R code is identical - the only difference is *when* it runs. The EDA plots, for example, use the same functions as analysis.qmd - they are just wrapped for the frontend.

| QMD (static document) | Shiny (interactive app) |
|---|---|
| Code runs once at render time | Code re-runs whenever an input changes |
| Output is a fixed PDF/HTML | Output updates live in the browser |
| `ggplot(...)` just outputs a plot | `renderPlot({ ggplot(...) })` wraps it for the browser |
| Data is hardcoded | Data can come from user upload or reactive filtering |

### Three pieces of every Shiny app

1. **UI (`ui`)** - defines what the user sees: sliders, buttons, plot placeholders, tables. Like HTML but written in R. Example: `plotOutput("eda_correlation_plot", height = "520px")` just says "put a plot here."

2. **Server (`server`)** - defines what happens. Each `renderPlot()`, `renderTable()`, `renderUI()` block contains normal R code that produces the output.

3. **Reactivity** - when you move a slider, any `render*()` block that reads that slider's value automatically re-runs. You don't write "on slider change, do X" - Shiny tracks the dependency automatically.

### Key Shiny patterns used in our app

**Reactive Values (`reactiveValues`, `reactiveVal`)** - Regular R variables don't trigger re-renders. These are special containers that do. When you write `rf_live(new_data)`, every `render*()` block that reads `rf_live()` automatically re-runs. This is how live recompute results get stored and instantly displayed.

**`observeEvent` - do something when X happens** - "When the user clicks the Recompute button, run this code." Unlike `reactive()` which produces a value, `observeEvent` produces a side effect (train a model, update a reactiveVal, show a notification). The game picker uses a chain: league changes -> update game dropdown -> game selected -> fill all sliders.

**`reactive()` - computed values that cache** - Like a formula cell in Excel. The `probs()` reactive computes all 6 model predictions but re-computes only when an input slider changes. Multiple outputs (table, plot, banner) all call `probs()` but the prediction runs only once per change.

**`withProgress` + `incProgress` - the loading bar** - Wraps slow code (model retraining) in a progress bar. Sets to 10% at start, jumps to 100% when done. It is cosmetic - does not track actual per-tree progress, just shows "working..." then "done."

**`conditionalPanel` - show/hide UI based on state** - Hides the tuning table when LR is selected (no hyperparameter grid). The condition is a JavaScript expression that runs in the browser (`input.tune_model !== 'LR'`). Also used to show/hide the lambda slider (only for LASSO/Elastic Net).

**`renderUI` - dynamic UI generation** - Unlike `conditionalPanel` which hides/shows static UI, `renderUI` generates entirely different UI elements from R. When you switch from RF to KNN, the sidebar replaces the mtry slider with a k slider, changes help text, and swaps the button. The UI itself is reactive.

**Parallel backend (`doParallel` + `foreach`)** - Not a Shiny concept, but what makes live recompute practical. The app creates a cluster of worker processes at startup. When you hit Recompute, the 5-fold CV distributes work across workers using `%dopar%`. For RF, it splits 500 trees into chunks across folds so all cores stay busy. `.packages = "randomForest"` is needed because each worker is a separate R process.

**Lazy data loading** - Models (.rds files) load at startup because every tab needs them. The raw CSV (80MB) loads only when the user clicks "Load CSV" - heavy data that only one tab needs should not block startup.

**Prediction priority chain** - When you move a slider in Hyperparameter Tuning, the app decides where to get predictions: (1) live recompute results if you ran one, (2) held-out test set if at bestTune, (3) saved out-of-fold CV predictions for other pre-tested grid points. The `source` tag drives the info message shown to the user.

### Summary answer for defence

> "The app uses Shiny's reactive framework. UI elements like sliders and buttons are inputs. Each plot and table is wrapped in a `render*()` block that automatically re-runs when its inputs change. For expensive operations like model retraining, we use `observeEvent` to trigger only on button click, `withProgress` for the loading bar, and a parallel cluster to distribute 5-fold CV across CPU cores. The UI itself is dynamic - `renderUI` swaps out the entire parameter panel when you switch models."


## Risks to Prepare For

1. **CSV not found** - If the default CSV path does not resolve, Data Explorer and EDA tabs show errors. Test this before the defence - make sure the symlink or file is in place.

2. **Startup time** - If the committee gets impatient during the 10-15 second load, say: "The app precomputes three feature selection methods at startup so the Scenario 3 tab is instant."

3. **Live recompute takes time** - RF recompute takes 10-30 seconds depending on the machine. Have the explanation ready while it runs.

4. **Upload option** - The upload CSV option exists but expects the exact Oracle's Elixir schema. Stick with "Default location" during the demo.

5. **XGBoost has 3 tunable parameters but the grid only explores a subset** - If challenged: "We used caret's default grid expansion. The live recompute lets you explore any combination beyond the pre-tested grid."
