Notes
1. Target distribution 
- Needed at 1st - checks what metric we need for model evaulation. 
- If inbalanced -> Accuracy lies -> need metrics like AUC-ROC / F1 / recall -> need class weights / SMOTE


# Algos / Tools

## SMOTE - syntetic data generator 
- Take a minority class sample
- Find its k nearest neighbors (other minority samples)
- Randomly pick one neighbor
- Generate a new point somewhere between them

-> Great to boost imbalanced classes in the sample


## Keywords

Information leakage -> seeing data in train that the model should not see (predicting winner at 15 minutes is more useful than predicting winner after the match is almost over)

Modeling unit -> 

Factor - storing categorical data.
GLM = Generalized Linear Model. It's R's general-purpose function for fitting regression-style models when the target isn't a plain continuous number.

## 5-fold CV 
get all training data, split to 5 parts -> train 4 models - each on 1/4 of data -> then validate on last. Pick best model -> fit on all data and then validate on final test set.  

caret -> "Classification And REgression Training" -> same API for all classification models. (library)

Sure — plain English version:

```r
ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final",
    verboseIter = TRUE
)
```

**`method = "cv"`** → "Use cross-validation."
Instead of one train/test split, do the rotating-folds thing.

**`number = 5`** → "Use 5 folds."
Cut the data into 5 pieces, train 5 times, each piece gets to be the test piece once.

**`classProbs = TRUE`** → "Give me probabilities, not just Win/Loss."
So instead of "Blue wins", the model says "Blue wins with 73% probability". We need this to draw ROC curves.

**`summaryFunction = twoClassSummary`** → "Score each fold using AUC, recall, specificity."
Without this, caret only reports accuracy. We want AUC, so we ask for the scoring function that produces it.
    - elternatives: prSummary (precision, recall, F1) - don't need, we have balanced dataset
**`savePredictions = "final"`** → "Remember what each model predicted for every game."
Useful later if we want to look at predictions without retraining. `"final"` = save only for the best hyperparameter setting (saves memory).

**`verboseIter = TRUE`** → "Print progress while training."
Shows lines like `Fold1: mtry= 5` so you can tell it's alive when training takes minutes.


## Youden's J (optimal cutoff) - why we skip it

Youden's J = sensitivity + specificity − 1, maximised over the ROC curve to pick a "best" probability threshold instead of the default 0.5.

We do not apply it because:
- Classes are near-balanced (~53/47 blue/red wins) → default 0.5 is already close to the Youden-optimal threshold; expected gain on accuracy/F1 is ~0.1-0.5 pp.
- Our headline metric is AUC, which is threshold-independent → tuning the cutoff does not move the number we report.
- No asymmetric cost between false positives and false negatives in this problem (predicting the wrong winner at minute 15 has the same "cost" either way), so there is no domain reason to shift the threshold.

If we had imbalance or asymmetric costs, Youden's J (or a cost-weighted threshold) would be the right move.

