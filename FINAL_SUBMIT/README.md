# OZNAL Capstone - LoL Esports Early-Game Prediction

## Dataset

2025 season export from **Oracle's Elixir** - the standard public source for professional League of Legends match data. 120,636 rows across 165 columns, covering 10,053 unique matches played by 445 teams across 45 professional leagues on patches 15.01-15.24. The CSV (`2025_LoL_esports_match_data_from_OraclesElixir.csv`) sits in this folder.

- Source: <https://oracleselixir.com/tools/downloads>
- Project copy: <https://drive.google.com/drive/folders/1uuM-gygXmHzwMYiqkopjeESIBH1moDaR?usp=sharing>

Missing CRAN packages are installed automatically on first run.

## Launch the Shiny app

From this folder:

```r
shiny::runApp("shiny_app.R")
```

Or in RStudio: open `shiny_app.R` and click **Run App**.

## Re-render the analysis

```bash
quarto render analysis.Qmd
```
