# OZNAL Capstone Project - LoL Esports Early-Game Prediction

Predicting professional League of Legends match outcomes from early-game performance metrics (first 15 minutes).

## Hypotheses

- **H1**: Early-game performance metrics and objective control can reliably predict match outcome
- **H2**: Bot lane (ADC) gold advantage is the most impactful role-specific early-game factor

## Dataset

- **Public source (credit)**: [Oracle's Elixir](https://oracleselixir.com/tools/downloads) - 2025 Professional LoL Esports Match Data, maintained by Tim Sevenhuysen
- **Size**: 120,636 rows, 165 columns, 10,053 games across 45 leagues
- **Download (project copy)**: [Google Drive folder](https://drive.google.com/drive/folders/1uuM-gygXmHzwMYiqkopjeESIBH1moDaR?usp=sharing) - this is a copy of the Oracle's Elixir dataset; download `2025_LoL_esports_match_data_from_OraclesElixir.csv` and place it in the same folder as the R code (alongside the app and analysis files)
- **Note**: CSV not tracked in git due to size

## Scenarios

- **S1**: Model Comparison (LR, RF, Naive Bayes, KNN, CART) + feature-space partitioning
- **S3**: Feature Selection — 1 algorithmic (Forward Stepwise) + 2 embedded (LASSO, Elastic Net)

## Setup

Install R packages:

```r
install.packages(c(
  "tidyverse", "corrplot", "caret", "glmnet",
  "randomForest", "naivebayes", "rpart", "rpart.plot",
  "class", "pROC", "shiny", "shinydashboard"
))
```

## Project Structure

```
analysis.qmd    - main Quarto notebook
data/           - dataset + data README
docs/           - assignment spec, analysis documents
visualizations/ - EDA plots
```

## Team

- Martin Vančo
- Adriana Gemeľová

## Course

The Elements of Statistical Learning (OZNAL), STU FIIT, 2026
