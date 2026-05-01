# OZNAL Capstone - LoL Esports Early-Game Prediction

## Dataset

2025 season export from **Oracle's Elixir** - the standard public source for professional League of Legends match data. 120,636 rows across 165 columns, covering 10,053 unique matches played by 445 teams across 45 professional leagues on patches 15.01-15.24.

- Source: <https://oracleselixir.com/tools/downloads>
- Project copy: <https://drive.google.com/drive/folders/1uuM-gygXmHzwMYiqkopjeESIBH1moDaR?usp=sharing>

Download the CSV (`2025_LoL_esports_match_data_from_OraclesElixir.csv`) and drop it into this folder before running anything.

Missing CRAN packages are installed automatically on first run.

## How to run

The Shiny app depends on saved model artifacts (`models/*.rds`) that are produced by the analysis pipeline. Run the steps in order:

1. **Render the analysis to generate the models.** This knits `analysis.Qmd` end-to-end and writes the fitted models, lookup tables, and supporting `.rds` files into `./models/`. Takes ~10-15 minutes on a typical laptop.

   ```bash
   quarto render analysis.Qmd
   ```

   Or in RStudio: open `analysis.Qmd` and click **Render**.

2. **Launch the Shiny app.** Once `./models/` is populated, start the app from this folder:

   ```r
   shiny::runApp("shiny_app.R")
   ```

   Or in RStudio: open `shiny_app.R` and click **Run App**. The 80MB CSV is loaded only on demand via the "Load CSV from current location" button on the Data Explorer tab.

A pre-rendered `analysis.html` is included for reading without re-running the pipeline.
