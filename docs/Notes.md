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
