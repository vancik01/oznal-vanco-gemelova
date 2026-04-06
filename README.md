# OZNAL Capstone Project - LoL Esports Early-Game Prediction

Predicting professional League of Legends match outcomes from early-game performance metrics (first 15 minutes).

## Hypotheses

- **H1**: Early-game performance metrics and objective control can reliably predict match outcome
- **H2**: Bot lane (ADC) gold advantage is the most impactful role-specific early-game factor

## Dataset

- **Source**: [Oracle's Elixir](https://oracleselixir.com/tools/downloads) - 2025 Professional LoL Esports Match Data
- **Size**: 120,636 rows, 165 columns, 10,053 games across 45 leagues
- **Download**: [2025_LoL_esports_match_data_from_OraclesElixir.csv](https://oracleselixir.com/tools/downloads/2025_LoL_esports_match_data_from_OraclesElixir.csv) - place in `data/`
- **Note**: CSV not tracked in git due to size

## Scenarios

- **S1**: Model Comparison (LR, RF, XGBoost, KNN, CART) + feature-space partitioning
- **S3**: Feature Selection (RFE, Forward Stepwise, LASSO, Elastic Net, RF importance)

## Project Structure

```
data/           - dataset + data README
docs/           - assignment spec, analysis documents
visualizations/ - EDA plots
```

## Team

- Martin Vanco
- Adriana Gemeľová

## Course

The Elements of Statistical Learning (OZNAL), STU FIIT, 2026
