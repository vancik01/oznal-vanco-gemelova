library(shiny)
library(bslib)
library(tidyverse)
library(caret)
library(DT)
library(pROC)
library(gridExtra)

# ── Load models ────────────────────────────────────────────────────────────────
lr_model     <- readRDS("models/lr_model.rds")
rf_model     <- readRDS("models/rf_model.rds")
nb_model     <- readRDS("models/nb_model.rds")
knn_model    <- readRDS("models/knn_model.rds")
cart_model   <- readRDS("models/cart_model.rds")
preproc      <- readRDS("models/preproc.rds")
perf_summary <- readRDS("models/perf_summary.rds")
overlap_df   <- readRDS("models/overlap_df.rds")
games_model  <- readRDS("models/games_model.rds")
X_train      <- readRDS("models/X_train.rds")
game_lookup  <- readRDS("models/game_lookup.rds")
lr_test_df   <- tryCatch(readRDS("models/lr_test_df.rds"), error = function(e) NULL)
lr_cm_obj    <- tryCatch(readRDS("models/lr_cm.rds"),      error = function(e) NULL)
games_data   <- tryCatch(readRDS("models/games.rds"),      error = function(e) NULL)

feature_names <- names(X_train)
train_means   <- colMeans(X_train)

# ── Model metadata ─────────────────────────────────────────────────────────────
ACCENT <- "#5383e8"

model_meta <- list(
    LR = list(
        name  = "Logistic Regression",
        color = ACCENT,
        param = "None",
        desc  = "Standard GLM — no hyperparameter search. All features used, trained on standardized inputs. Coefficients are directly interpretable as log-odds weights."
    ),
    RF = list(
        name  = "Random Forest",
        color = ACCENT,
        param = "mtry",
        desc  = "Ensemble of 500 decision trees. mtry controls how many features are randomly considered at each split. Tested: 5, 7, 10, 15, 20."
    ),
    NB = list(
        name  = "Naive Bayes",
        color = ACCENT,
        param = "usekernel + adjust",
        desc  = "Probabilistic classifier. Tunes density estimation (Gaussian vs. kernel) and bandwidth multiplier (adjust). Tested: 2 × 3 = 6 combinations."
    ),
    KNN = list(
        name  = "K-Nearest Neighbors",
        color = ACCENT,
        param = "k",
        desc  = "Classifies by majority vote among the k nearest neighbors in scaled feature space. Small k = noisy boundary; large k = smooth but biased."
    ),
    CART = list(
        name  = "Decision Tree (CART)",
        color = ACCENT,
        param = "cp (complexity parameter)",
        desc  = "Recursive binary splitting tree. cp penalizes tree growth — small cp allows deep trees, large cp forces early stopping. Tested on log scale."
    )
)

# ── Dark ggplot theme ──────────────────────────────────────────────────────────
dark_theme <- function(base_size = 13) {
    theme_minimal(base_size = base_size) +
    theme(
        plot.background   = element_rect(fill = "#1c1c2e", color = NA),
        panel.background  = element_rect(fill = "#1c1c2e", color = NA),
        panel.grid.major  = element_line(color = "#2a2a3e"),
        panel.grid.minor  = element_blank(),
        axis.text         = element_text(color = "#9aaccc"),
        axis.title        = element_text(color = "#9aaccc"),
        plot.title        = element_text(color = "#ffffff", face = "bold"),
        plot.subtitle     = element_text(color = "#6a7590"),
        legend.background = element_rect(fill = "#1c1c2e", color = NA),
        legend.text       = element_text(color = "#9aaccc"),
        legend.title      = element_text(color = "#9aaccc"),
        strip.text        = element_text(color = "#9aaccc")
    )
}

# ── CSS ───────────────────────────────────────────────────────────────────────
nuxt_css <- "
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

body { background: #13131e; color: #c8d0e0; }

/* Navbar */
.navbar {
    background: #1c1c2e !important;
    border-bottom: 1px solid #2a2a3e;
    box-shadow: none;
    padding: 0 24px;
}
.navbar-brand {
    font-weight: 700;
    font-size: 16px;
    color: #5383e8 !important;
    letter-spacing: -0.3px;
}
.navbar-nav > li > a {
    color: #9aaccc !important;
    font-size: 14px;
    font-weight: 500;
    padding: 18px 16px !important;
}
.navbar-nav > li.active > a,
.navbar-nav > li > a:hover {
    color: #ffffff !important;
    border-bottom: 2px solid #5383e8;
    background: transparent !important;
}

/* Cards */
.card {
    background: #1c1c2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    box-shadow: none;
    padding: 20px;
    margin-bottom: 16px;
}

/* Replace wellPanel */
.well {
    background: #1c1c2e !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    padding: 20px !important;
}

/* Probability boxes */
.prob-box {
    background: #1c1c2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.prob-box:hover { border-color: #5383e8; }
.prob-label {
    font-size: 12px;
    font-weight: 600;
    color: #6a7590;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.prob-value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin: 0;
}

/* ── Predictor sidebar ───────────────────────────────────── */
.predictor-sidebar {
    height: calc(100vh - 110px);
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: #2e3250 transparent;
}
.predictor-sidebar::-webkit-scrollbar { width: 4px; }
.predictor-sidebar::-webkit-scrollbar-thumb { background: #2e3250; border-radius: 4px; }

.sidebar-section {
    background: #1c1c2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.sidebar-section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #6a7590;
    margin: 0 0 10px 0;
}

.predictor-sidebar .shiny-input-container > label {
    font-size: 12px;
    color: #9aaccc;
    font-weight: 500;
    margin-bottom: 1px;
}
.predictor-sidebar .form-group { margin-bottom: 10px; }
.predictor-sidebar .form-group:last-child { margin-bottom: 0; }
.predictor-sidebar .checkbox label,
.predictor-sidebar .radio label { font-size: 13px; color: #9aaccc; }

.autofill-hint {
    font-size: 11px;
    color: #6a7590;
    margin: 6px 0 8px 0;
    padding: 6px 8px;
    background: #13131e;
    border-radius: 6px;
    border: 1px dashed #2a2a3e;
}

.section-divider {
    display: flex;
    align-items: center;
    margin: 6px 0 10px 0;
}
.section-divider hr {
    flex: 1;
    margin: 0;
    border: none;
    border-top: 1px solid #2a2a3e;
}
.section-divider span {
    padding: 0 10px;
    font-size: 10px;
    color: #6a7590;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    white-space: nowrap;
}

.wr-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #6a7590;
    margin-top: -6px;
    margin-bottom: 2px;
}

/* ── Hyperparameter pill selector ────────────────────────── */
.pill-radio .radio-inline {
    margin: 0 4px 0 0;
    padding: 0;
}
.pill-radio .radio-inline label {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    border: 1px solid #2a2a3e;
    background: #1c1c2e;
    color: #9aaccc;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    margin: 0;
    transition: all 0.15s;
    user-select: none;
}
.pill-radio .radio-inline label:hover {
    border-color: #5383e8;
    color: #5383e8;
}
.pill-radio .radio-inline label.pill-active {
    background: #5383e8;
    border-color: #5383e8;
    color: #ffffff;
}
.pill-radio input[type='radio'] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}

/* ── Tune metric mini-cards ──────────────────────────────── */
.tune-info-card {
    background: #1c1c2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.tune-info-card-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #6a7590;
    margin: 0 0 8px 0;
}
.tune-info-card p {
    font-size: 13px;
    color: #9aaccc;
    margin: 0;
    line-height: 1.5;
}
.tune-metric-row {
    display: flex;
    gap: 8px;
    margin-top: 4px;
}
.tune-metric-box {
    flex: 1;
    background: #13131e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}
.tune-metric-box-label {
    font-size: 10px;
    font-weight: 600;
    color: #6a7590;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 2px;
}
.tune-metric-box-value {
    font-size: 18px;
    font-weight: 700;
    color: #c8d0e0;
    line-height: 1;
}
.best-badge {
    display: inline-block;
    background: #1e3a6a;
    color: #5383e8;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 4px;
}
.best-badge-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6a7590;
    margin-bottom: 4px;
}

/* Inputs */
.form-control {
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    color: #c8d0e0 !important;
    background: #0e0e1a !important;
}
.form-control:focus {
    border-color: #5383e8 !important;
    box-shadow: 0 0 0 3px rgba(83,131,232,0.15) !important;
    background: #13131e !important;
}
.selectize-input {
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    background: #0e0e1a !important;
    color: #c8d0e0 !important;
}
.selectize-dropdown {
    background: #1c1c2e !important;
    border: 1px solid #2a2a3e !important;
    color: #c8d0e0 !important;
}
.selectize-dropdown .option { color: #c8d0e0 !important; }
.selectize-dropdown .option:hover,
.selectize-dropdown .option.active { background: #2a2a3e !important; }

/* Slider */
.irs--shiny .irs-bar { background: #5383e8; }
.irs--shiny .irs-handle { background: #5383e8; border-color: #5383e8; }
.irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single { background: #5383e8; }
.irs--shiny .irs-line { background: #2a2a3e; }
.irs--shiny .irs-grid-text { color: #6a7590; }
.irs--shiny .irs-min, .irs--shiny .irs-max { color: #6a7590; }

/* Button */
.btn-primary {
    background: #5383e8 !important;
    border-color: #5383e8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #ffffff !important;
    padding: 10px 20px !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}
.btn-primary:hover {
    background: #3f6bd4 !important;
    border-color: #3f6bd4 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(83, 131, 232, 0.5) !important;
}

/* ── Model nav (vertical sidebar) ───────────────────────── */
.model-nav .shiny-input-container > label { display: none; }
.model-nav .radio { margin: 0 0 2px 0; }
.model-nav .radio label {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #9aaccc;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    margin: 0;
    width: 100%;
}
.model-nav .radio label:hover {
    background: #13131e;
    color: #c8d0e0;
    border-left-color: #2a2a3e;
}
.model-nav .radio label.model-nav-active {
    background: #1e3a6a;
    color: #5383e8;
    border-left-color: #5383e8;
    font-weight: 600;
}
.model-nav input[type='radio'] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}
.model-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Reset to Best button ────────────────────────────────── */
.btn-reset-best {
    padding: 4px 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    background: #1e3a6a !important;
    border: 1px solid #2e3a5e !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}
.btn-reset-best:hover {
    background: #253260 !important;
    color: #ffffff !important;
    transform: none !important;
    box-shadow: 0 0 14px rgba(83, 131, 232, 0.4) !important;
}

/* Game banner */
.game-banner {
    background: #1c1c2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.banner-teams { font-size: 18px; font-weight: 700; color: #ffffff; }
.banner-meta  { font-size: 13px; color: #6a7590; margin-top: 2px; }
.badge-win  { background: #0f3a28; color: #27ae60; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
.badge-loss { background: #3a1020; color: #e84057; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
.badge-correct { background: #1e3a6a; color: #5383e8; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; margin-left: 8px; }
.badge-wrong   { background: #3a2e10; color: #e8a838; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; margin-left: 8px; }

/* Tabs */
.nav-tabs { border-bottom: 1px solid #2a2a3e; }
.nav-tabs > li > a {
    font-size: 13px; font-weight: 500; color: #6a7590;
    border: none !important; border-radius: 0 !important; padding: 10px 16px;
    background: transparent !important;
}
.nav-tabs > li > a:hover { color: #c8d0e0 !important; background: transparent !important; }
.nav-tabs > li.active > a {
    color: #ffffff !important;
    border-bottom: 2px solid #5383e8 !important;
    background: transparent !important;
}

/* Page padding */
.tab-content { padding-top: 8px; }
.container-fluid { padding: 0 24px; }

h4 { font-weight: 700; font-size: 18px; color: #ffffff; letter-spacing: -0.3px; }
h5 { font-weight: 600; font-size: 14px; color: #c8d0e0; }
h6 { font-weight: 700; font-size: 11px; color: #6a7590; text-transform: uppercase; letter-spacing: 0.6px; }

/* ── Data tab fixed layout ───────────────────────────────── */
.data-fixed-layout {
    height: calc(100vh - 110px);
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow: hidden;
    padding-top: 16px;
}
.kpi-row {
    display: flex;
    gap: 10px;
    flex-shrink: 0;
}
.kpi-card {
    flex: 1;
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 0;
    border-width: 1px;
    border-style: solid;
}
.kpi-card-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #6a7590;
    margin-bottom: 6px;
}
.kpi-card-value {
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
}
.data-content-card {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: #1c1c2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    box-shadow: none;
    padding: 20px;
    margin-bottom: 0;
}
.data-fill-area {
    flex: 1;
    min-height: 0;
    overflow: hidden;
}
.data-fill-area .shiny-plot-output { height: 100% !important; }

/* View toggle */
.view-toggle { display:inline-flex; background:#13131e; border-radius:8px; padding:3px; gap:2px; border:1px solid #2a2a3e; }
.view-toggle .radio-inline { margin:0; padding:0; }
.view-toggle .radio-inline label {
    display:inline-block; padding:5px 14px; border-radius:6px;
    font-size:12px; font-weight:500; color:#9aaccc;
    cursor:pointer; margin:0; transition:all 0.15s; user-select:none; white-space:nowrap;
}
.view-toggle .radio-inline label.vt-active {
    background:#1c1c2e; color:#5383e8; font-weight:600;
    box-shadow:0 1px 3px rgba(0,0,0,0.4);
}
.view-toggle input[type='radio'] { position:absolute; opacity:0; width:0; height:0; }

/* Data tab — real tab bar */
.data-view .nav-tabs {
    border-bottom: 2px solid #2a2a3e !important;
    background: transparent !important;
    border-radius: 0 !important;
    display: flex !important;
    padding: 0 !important;
    gap: 0 !important;
    margin-bottom: 0 !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
}
.data-view .nav-tabs > li { margin: 0 !important; }
.data-view .nav-tabs > li > a {
    border-radius: 0 !important;
    padding: 10px 24px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    color: #6a7590 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: transparent !important;
    margin-right: 0 !important;
    line-height: 1.5 !important;
    transition: color 0.15s !important;
}
.data-view .nav-tabs > li.active > a,
.data-view .nav-tabs > li.active > a:hover,
.data-view .nav-tabs > li.active > a:focus {
    background: rgba(83, 131, 232, 0.12) !important;
    color: #5383e8 !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #5383e8 !important;
    border-radius: 6px 6px 0 0 !important;
    box-shadow: none !important;
}
.data-view .nav-tabs > li > a:hover {
    background: transparent !important;
    color: #c8d0e0 !important;
}
.data-view .tab-content { padding-top: 12px !important; }

/* DT table */
.dataTables_wrapper { font-size: 13px; color: #c8d0e0; }
table.dataTable thead th {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
    color: #6a7590; border-bottom: 1px solid #2a2a3e !important;
    background: #1c1c2e !important;
}
table.dataTable tbody tr { background: #1c1c2e !important; color: #c8d0e0; }
table.dataTable tbody tr:hover { background: #22223a !important; }
table.dataTable tbody tr:nth-child(even) { background: #1a1a2e !important; }
table.dataTable tbody tr.selected td { background: #1e3a6a !important; }
table.dataTable tbody td {
    white-space: nowrap;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.dataTables_filter input,
.dataTables_length select {
    background: #0e0e1a !important;
    border: 1px solid #2a2a3e !important;
    color: #c8d0e0 !important;
    border-radius: 6px !important;
}
.dataTables_info {
    color: #6a7590 !important;
    float: left !important;
    padding-top: 7px !important;
    font-size: 12px !important;
}
.dataTables_paginate {
    color: #6a7590 !important;
    float: right !important;
}
.paginate_button { color: #9aaccc !important; }
.paginate_button.current { background: #1e3a6a !important; color: #5383e8 !important; border-color: #2e3a5e !important; }
.paginate_button:hover { background: #2a2a3e !important; color: #c8d0e0 !important; border-color: #2a2a3e !important; }

/* ── Data Explorer: fileInput styling ──────────────────── */
.input-group {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #2a2a3e !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.input-group:hover {
    border-color: #5383e8 !important;
    box-shadow: 0 0 14px rgba(83, 131, 232, 0.3) !important;
}
.input-group .btn-file {
    background: #5383e8 !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 8px 18px !important;
    border-radius: 0 !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
.input-group .btn-file:hover {
    background: #3f6bd4 !important;
}
.input-group .form-control[readonly] {
    font-size: 13px !important;
    color: #6a7590 !important;
    border: none !important;
    border-radius: 0 !important;
    background: #0e0e1a !important;
}

/* ── Data Explorer sidebar ──────────────────────────────── */
.data-or-divider {
    display: flex; align-items: center; gap: 10px;
    margin: 10px 0; color: #6a7590;
    font-size: 11px; font-weight: 600;
}
.data-or-divider::before, .data-or-divider::after {
    content: ''; flex: 1; border-top: 1px solid #2a2a3e;
}
.data-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid #2a2a3e;
}
.data-stat-row:last-child { border-bottom: none; }
.data-stat-label { font-size: 12px; color: #9aaccc; font-weight: 500; }
.data-stat-value { font-size: 15px; font-weight: 700; }
.data-main-area {
    min-height: calc(100vh - 130px);
    display: flex;
    flex-direction: column;
    padding-top: 0;
}
.data-no-data-hint {
    display: flex; align-items: center; justify-content: center;
    padding: 80px 0; flex-direction: column; text-align: center;
}

/* ── Game browse table: compact pagination ───────────── */
#game_table_wrapper {
    overflow: hidden;
}
#game_table_wrapper .dataTables_paginate {
    float: none !important;
    clear: both;
    display: block;
    text-align: center;
    padding: 8px 0 0 0;
}
#game_table_wrapper .paginate_button {
    padding: 2px 7px !important;
    font-size: 11px !important;
    border-radius: 4px !important;
    min-width: 0 !important;
    margin: 0 1px !important;
}
#game_table_wrapper table.dataTable {
    margin-bottom: 0 !important;
}

/* ── Feature collapse toggle ─────────────────────────── */
.features-toggle { cursor: pointer; }
.features-toggle:hover span { color: #c8d0e0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #13131e; }
::-webkit-scrollbar-thumb { background: #2e3250; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #5383e8; }
"

# ── JS for pill active state ───────────────────────────────────────────────────
pill_js <- "
$(document).on('shiny:connected', function() {
    function syncModelNav() {
        var val = $('input[name=tune_model]:checked').val();
        $('.model-nav label').removeClass('model-nav-active');
        $('input[name=tune_model]').each(function() {
            if ($(this).val() === val) {
                $(this).closest('label').addClass('model-nav-active');
            }
        });
    }
    $(document).on('change', 'input[name=tune_model]', syncModelNav);
    setTimeout(syncModelNav, 250);

    function syncViewToggle() {
        var val = $('input[name=data_view]:checked').val();
        $('.view-toggle label').removeClass('vt-active');
        $('input[name=data_view]').each(function() {
            if ($(this).val() === val) $(this).closest('label').addClass('vt-active');
        });
    }
    $(document).on('change', 'input[name=data_view]', syncViewToggle);
    setTimeout(syncViewToggle, 300);
});

// Feature section collapse
function toggleFeatureSection() {
    var c = $('#features_content');
    var ic = $('#features_toggle_icon');
    if (c.is(':visible')) {
        c.slideUp(200);
        ic.text('▸');
    } else {
        c.slideDown(200);
        ic.text('▾');
    }
}

// Detect manual slider/checkbox/radio interactions (mouse only, not server-side updates)
var userInteractingWithSlider = false;
$(document).on('mousedown touchstart', '.predictor-sidebar .irs', function() {
    userInteractingWithSlider = true;
});
$(document).on('mouseup touchend', function() {
    setTimeout(function() { userInteractingWithSlider = false; }, 200);
});
$(document).on('change', '#golddiffat15, #xpdiffat15, #csdiffat15, #grub_diff, #wr_diff', function() {
    if (userInteractingWithSlider && window.Shiny)
        Shiny.setInputValue('user_tweaked_feature', Date.now(), {priority: 'event'});
});
$(document).on('click', '#firstblood, #firstdragon, #firstherald, #firsttower, input[name=side]', function() {
    if (window.Shiny) Shiny.setInputValue('user_tweaked_feature', Date.now(), {priority: 'event'});
});
"

# ── UI ─────────────────────────────────────────────────────────────────────────
ui <- navbarPage(
    title = "LoL Early-Game Predictor",
    theme = bslib::bs_theme(
        bg      = "#13131e",
        fg      = "#c8d0e0",
        primary = "#5383e8",
        base_font = bslib::font_google("Inter")
    ),
    header = tags$head(
        tags$style(HTML(nuxt_css)),
        tags$script(HTML(pill_js))
    ),

    # ── Tab 0: Introduction ────────────────────────────────────────────────────
    tabPanel("Introduction",
        fluidRow(
            column(8, offset = 2,
                br(),

                # ── Hero ───────────────────────────────────────────────────────
                div(class = "card", style = "text-align:center; padding:36px 24px 28px;",
                    h2(style = "color:#ffffff; font-weight:700; margin:0 0 10px 0;",
                        "LoL Early-Game Predictor"),
                    p(style = "color:#9aaccc; font-size:15px; max-width:580px; margin:0 auto;",
                        "Can the first 15 minutes of a professional League of Legends match",
                        "predict the final winner?",
                        "We trained five classification models on real competitive match data to find out.")
                ),

                br(),

                # ── What is LoL ────────────────────────────────────────────────
                div(class = "card",
                    p(class = "sidebar-section-title", "What is League of Legends?"),

                    fluidRow(
                        column(7,
                            p(style = "color:#c8d0e0; line-height:1.8; margin-bottom:12px;",
                                tags$b("League of Legends"), " is one of the world's most popular",
                                "competitive video games with over 150 million registered accounts",
                                "and a professional esports scene comparable in viewership to traditional sports."),
                            p(style = "color:#c8d0e0; line-height:1.8; margin-bottom:12px;",
                                "Two teams of ", tags$b("5 players"), " face off on a map called",
                                tags$b("Summoner's Rift."), " Each player controls a unique character",
                                "— called a ", tags$b("champion"), " — with different abilities.",
                                "The match ends when one team destroys the enemy ", tags$b("Nexus"),
                                " (the main base structure on the opposite side of the map)."),
                            p(style = "color:#c8d0e0; line-height:1.8; margin:0;",
                                "A typical match lasts 30–45 minutes,",
                                "but the first 15 minutes — the ", tags$b("early game"), " — largely",
                                "determine who has the upper hand for the rest of the match.")
                        ),
                        column(5,
                            img(src = "map.png", style = "width:100%; border-radius:8px;")
                        )
                    ),

                    div(style = "position:relative; padding-top:56.25%; margin-top:16px; border-radius:8px; overflow:hidden;",
                        tags$iframe(
                            src = "https://www.youtube.com/embed/StDaM6Rk-wI?start=1205&end=1230",
                            style = "position:absolute; top:0; left:0; width:100%; height:100%; border:none;",
                            allowfullscreen = NA
                        )
                    )
                ),

                br(),

                # ── Key concepts ───────────────────────────────────────────────
                div(class = "card",
                    p(class = "sidebar-section-title", "Key concepts"),
                    fluidRow(
                        column(4,
                            div(class = "prob-box", style = "height:100%;",
                                img(src = "gold.svg", style = "width:100%; border-radius:6px; margin-bottom:8px;"),
                                div(class = "prob-label", "Gold & CS"),
                                p(style = "color:#c8d0e0; font-size:13px; line-height:1.6; margin:6px 0 0 0;",
                                    "Players earn gold by defeating minions (CS) and enemies.",
                                    "Gold buys items that make champions stronger.",
                                    "A gold lead at 15 min is a strong advantage indicator.")
                            )
                        ),
                        column(4,
                            div(class = "prob-box", style = "height:100%;",
                                img(src = "objectives.svg", style = "width:100%; border-radius:6px; margin-bottom:8px;"),
                                div(class = "prob-label", "Objectives"),
                                p(style = "color:#c8d0e0; font-size:13px; line-height:1.6; margin:6px 0 0 0;",
                                    "Dragons, Heralds, and Towers grant permanent bonuses.",
                                    "Securing the First Dragon or First Tower gives a measurable,",
                                    "lasting edge for the rest of the match.")
                            )
                        ),
                        column(4,
                            div(class = "prob-box", style = "height:100%;",
                                img(src = "firstblood.svg", style = "width:100%; border-radius:6px; margin-bottom:8px;"),
                                div(class = "prob-label", "First Blood"),
                                p(style = "color:#c8d0e0; font-size:13px; line-height:1.6; margin:6px 0 0 0;",
                                    "The first kill of the match. It grants bonus gold and",
                                    "signals early dominance. A simple binary event —",
                                    "yes or no — that is surprisingly predictive of the outcome.")
                            )
                        )
                    )
                ),

                br(),

                # ── Why is this a hard prediction problem ─────────────────────
                div(class = "card",
                    p(class = "sidebar-section-title", "Why is this an interesting prediction problem?"),
                    p(style = "color:#c8d0e0; line-height:1.8;",
                        "Professional LoL teams are extremely evenly matched — any team can win on a given day.",
                        "A 15-minute snapshot captures only a ", tags$b("partial picture"),
                        " of the match: teams can throw a large lead or stage a comeback.",
                        "This makes early-game prediction genuinely non-trivial and a good classification benchmark."),
                    p(style = "color:#c8d0e0; line-height:1.8; margin:0;",
                        "Additionally, Riot Games (the developer) publishes detailed match statistics via a public API,",
                        "making LoL one of the richest freely available sources of structured, time-series sports data.")
                ),

                br(),

                # ── Dataset stats ──────────────────────────────────────────────
                div(class = "card",
                    p(class = "sidebar-section-title", "The Dataset"),
                    p(style = "color:#c8d0e0; line-height:1.7; margin-bottom:14px;",
                        "Professional competitive match data from major regional leagues.",
                        "Each observation represents one team's perspective at the 15-minute mark."),
                    fluidRow(
                        column(3,
                            div(class = "prob-box",
                                div(class = "prob-label", "Matches"),
                                div(class = "prob-value", style = "color:#5383e8; font-size:26px;", "10 862")
                            )
                        ),
                        column(3,
                            div(class = "prob-box",
                                div(class = "prob-label", "Features"),
                                div(class = "prob-value", style = "color:#5383e8; font-size:26px;", "16")
                            )
                        ),
                        column(3,
                            div(class = "prob-box",
                                div(class = "prob-label", "Snapshot"),
                                div(class = "prob-value", style = "color:#5383e8; font-size:26px;", "15 min")
                            )
                        ),
                        column(3,
                            div(class = "prob-box",
                                div(class = "prob-label", "Outcome"),
                                div(class = "prob-value", style = "color:#5383e8; font-size:26px;", "Win / Loss")
                            )
                        )
                    )
                ),

                br(),

                # ── Navigation guide ───────────────────────────────────────────
                div(class = "card",
                    p(class = "sidebar-section-title", "How to use this app"),
                    tags$table(
                        style = "width:100%; border-collapse:collapse;",
                        tags$thead(
                            tags$tr(
                                tags$th(style = "color:#6a7590; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; padding:6px 12px 6px 0; border-bottom:1px solid #2a2a3e;", "Tab"),
                                tags$th(style = "color:#6a7590; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; padding:6px 0; border-bottom:1px solid #2a2a3e;", "What you'll find")
                            )
                        ),
                        tags$tbody(
                            tags$tr(
                                tags$td(style = "padding:10px 12px 10px 0; color:#5383e8; font-weight:600; font-size:13px; border-bottom:1px solid #2a2a3e;", "Match Predictor"),
                                tags$td(style = "padding:10px 0; color:#c8d0e0; font-size:13px; border-bottom:1px solid #2a2a3e;", "Enter early-game stats and get a live win-probability prediction from all five models.")
                            ),
                            tags$tr(
                                tags$td(style = "padding:10px 12px 10px 0; color:#5383e8; font-weight:600; font-size:13px; border-bottom:1px solid #2a2a3e;", "Hyperparameter Tuning"),
                                tags$td(style = "padding:10px 0; color:#c8d0e0; font-size:13px; border-bottom:1px solid #2a2a3e;", "See how each model's key hyperparameter was optimised and how it affects performance.")
                            ),
                            tags$tr(
                                tags$td(style = "padding:10px 12px 10px 0; color:#5383e8; font-weight:600; font-size:13px; border-bottom:1px solid #2a2a3e;", "Model Comparison"),
                                tags$td(style = "padding:10px 0; color:#c8d0e0; font-size:13px; border-bottom:1px solid #2a2a3e;", "Compare all five models on accuracy, AUC, precision, recall, and ROC curves.")
                            ),
                            tags$tr(
                                tags$td(style = "padding:10px 12px 10px 0; color:#5383e8; font-weight:600; font-size:13px; border-bottom:1px solid #2a2a3e;", "Feature Selection"),
                                tags$td(style = "padding:10px 0; color:#c8d0e0; font-size:13px; border-bottom:1px solid #2a2a3e;", "Explore which early-game statistics carry the most predictive signal.")
                            ),
                            tags$tr(
                                tags$td(style = "padding:10px 12px 10px 0; color:#5383e8; font-weight:600; font-size:13px;", "Data Explorer"),
                                tags$td(style = "padding:10px 0; color:#c8d0e0; font-size:13px;", "Browse the raw dataset — distributions, correlations, and win-rate breakdowns by feature.")
                            )
                        )
                    )
                ),

                br(),

                # ── Models badge row ───────────────────────────────────────────
                div(class = "card", style = "text-align:center;",
                    p(class = "sidebar-section-title", style = "margin-bottom:14px;", "Models trained"),
                    tags$span(style = "display:inline-flex; gap:10px; flex-wrap:wrap; justify-content:center;",
                        tags$span(style = "background:#1a2340; border:1px solid #5383e8; color:#5383e8; font-size:12px; font-weight:600; padding:5px 14px; border-radius:20px;", "Logistic Regression"),
                        tags$span(style = "background:#1a2340; border:1px solid #5383e8; color:#5383e8; font-size:12px; font-weight:600; padding:5px 14px; border-radius:20px;", "Random Forest"),
                        tags$span(style = "background:#1a2340; border:1px solid #5383e8; color:#5383e8; font-size:12px; font-weight:600; padding:5px 14px; border-radius:20px;", "Naive Bayes"),
                        tags$span(style = "background:#1a2340; border:1px solid #5383e8; color:#5383e8; font-size:12px; font-weight:600; padding:5px 14px; border-radius:20px;", "K-Nearest Neighbors"),
                        tags$span(style = "background:#1a2340; border:1px solid #5383e8; color:#5383e8; font-size:12px; font-weight:600; padding:5px 14px; border-radius:20px;", "Decision Tree (CART)")
                    )
                ),

                br()
            )
        )
    ),

    # ── Tab 1: Match Predictor ──────────────────────────────────────────────────
    tabPanel("Match Predictor",
        fluidRow(
            column(4,
                div(class = "predictor-sidebar",
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "Browse Game"),
                        selectInput("browse_league", "League",
                            choices = NULL, width = "100%"),
                        selectInput("browse_team", "Team",
                            choices = NULL, width = "100%"),
                        conditionalPanel(
                            condition = "input.browse_team !== ''",
                            div(class = "autofill-hint",
                                "Select a row to auto-fill the features below"),
                            div(style = "overflow: hidden;",
                                DTOutput("game_table")
                            ),
                            div(style = "clear: both;")
                        )
                    ),
                    div(class = "section-divider features-toggle",
                        onclick = "toggleFeatureSection()",
                        tags$hr(),
                        span(style = "display:flex; align-items:center; gap:6px; white-space:nowrap;",
                            "Features",
                            tags$span(id = "features_toggle_icon",
                                style = "font-size:12px; color:#6a7590;", "▾")
                        ),
                        tags$hr()
                    ),
                    div(id = "features_content",
                        div(class = "sidebar-section",
                            p(class = "sidebar-section-title", "Preset"),
                            selectInput("feature_preset", NULL,
                                choices  = c("Custom" = "__custom__"),
                                selected = "__custom__",
                                width    = "100%")
                        ),
                        div(class = "sidebar-section",
                            p(class = "sidebar-section-title", "Map Side"),
                            radioButtons("side", NULL,
                                choices  = c("Blue Side" = 1, "Red Side" = 0),
                                selected = 1, inline = TRUE)
                        ),
                        div(class = "sidebar-section",
                            p(class = "sidebar-section-title", "Economic Advantage @ 15 min"),
                            sliderInput("golddiffat15", "Gold Diff",
                                min = -10000, max = 10000, value = 1500, step = 100, width = "100%"),
                            sliderInput("xpdiffat15", "XP Diff",
                                min = -6000, max = 6000, value = 800, step = 100, width = "100%"),
                            sliderInput("csdiffat15", "CS Diff",
                                min = -80, max = 80, value = 12, step = 1, width = "100%")
                        ),
                        div(class = "sidebar-section",
                            p(class = "sidebar-section-title", "Objectives"),
                            fluidRow(
                                column(6, checkboxInput("firstblood",  "First Blood",  TRUE)),
                                column(6, checkboxInput("firstdragon", "First Dragon", TRUE))
                            ),
                            fluidRow(
                                column(6, checkboxInput("firstherald", "First Herald", FALSE)),
                                column(6, checkboxInput("firsttower",  "First Tower",  FALSE))
                            ),
                            sliderInput("grub_diff", "Void Grub Advantage",
                                min = -6, max = 6, value = 2, step = 1, width = "100%")
                        ),
                        div(class = "sidebar-section",
                            p(class = "sidebar-section-title", "Pre-game Strength"),
                            sliderInput("wr_diff", "Win Rate Advantage (%)",
                                min = -50, max = 50, value = 10, step = 1, width = "100%"),
                            div(class = "wr-labels",
                                span("Opponent stronger"), span("Team stronger"))
                        )
                    ),
                )
            ),
            column(8,
                uiOutput("game_banner"),
                fluidRow(
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Logistic Regression"),
                        div(class = "prob-value", style = "color:#c8d0e0", textOutput("prob_lr"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Random Forest"),
                        div(class = "prob-value", style = "color:#c8d0e0", textOutput("prob_rf"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Naive Bayes"),
                        div(class = "prob-value", style = "color:#c8d0e0", textOutput("prob_nb"))
                    ))
                ),
                fluidRow(
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "KNN"),
                        div(class = "prob-value", style = "color:#c8d0e0", textOutput("prob_knn"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "CART"),
                        div(class = "prob-value", style = "color:#c8d0e0", textOutput("prob_cart"))
                    )),
                    column(4, div(class = "prob-box",
                        style = "border-color:#5383e8;",
                        div(class = "prob-label", style = "color:#5383e8;", "Average"),
                        div(class = "prob-value", style = "color:#5383e8", textOutput("prob_avg"))
                    ))
                ),
                plotOutput("prob_bar", height = "200px")
            )
        )
    ),

    # ── Tab 2: Hyperparameter Tuning ────────────────────────────────────────────
    tabPanel("Hyperparameter Tuning",
        fluidRow(

            # ── Left sidebar ─────────────────────────────────────────────────────
            column(4,
                div(class = "predictor-sidebar",

                    # Model selector — vertical nav
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "Select Model"),
                        div(class = "model-nav",
                            radioButtons("tune_model", NULL, selected = "RF",
                                choiceNames = list(
                                    "Logistic Regression",
                                    "Random Forest",
                                    "Naive Bayes",
                                    "KNN",
                                    "CART"
                                ),
                                choiceValues = c("LR","RF","NB","KNN","CART"))
                        )
                    ),
                    # Model description
                    div(class = "sidebar-section",
                        uiOutput("tune_description")
                    ),

                    # Parameter controls
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "Parameter Controls"),
                        uiOutput("tune_slider")
                    ),

                    # CV best reference
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "CV Best Reference"),
                        uiOutput("tune_best")
                    )
                )
            ),

            # ── Right: output ─────────────────────────────────────────────────────
            column(8,
                br(),
                uiOutput("tune_metrics"),
                div(class = "card", style = "padding:20px;",
                    plotOutput("tune_plot", height = "340px")
                ),
                conditionalPanel(
                    condition = "input.tune_model !== 'LR'",
                    div(class = "card", style = "padding:20px;",
                        h5("All Tested Configurations",
                           style = "margin:0 0 14px 0; font-size:13px;"),
                        DTOutput("tune_table")
                    )
                ),
                conditionalPanel(
                    condition = "input.tune_model === 'LR'",
                    uiOutput("tune_lr_extra")
                )
            )
        )
    ),

    # ── Tab 3: Model Comparison ─────────────────────────────────────────────────
    tabPanel("Model Comparison",
        fluidRow(column(12,
            br(),
            h4("Model Performance Overview"),
            br(),
            DTOutput("model_table"),
            br(),
            plotOutput("roc_plot", height = "460px")
        ))
    ),

    # ── Tab 4: Feature Selection ────────────────────────────────────────────────
    tabPanel("Feature Selection",
        div(style = "padding: 0 24px;",
            br(),

            # ── Intro ─────────────────────────────────────────────────────────
            div(class = "card",
                p(class = "sidebar-section-title", "Scenario 3: Feature Selection"),
                p(style = "color:#c8d0e0; line-height:1.8; margin-bottom:16px;",
                    "The EDA revealed high multicollinearity between raw stats and their differentials",
                    " (e.g. ", tags$code("goldat15"), " vs ", tags$code("golddiffat15"),
                    ") and between @10 and @15 versions of the same metric. With ~46 candidate features",
                    " many are redundant — carrying the same signal in different forms.",
                    " We applied ", tags$b("five feature selection methods"),
                    " to find the minimal subset that retains full predictive performance."
                ),
                fluidRow(
                    column(3,
                        div(class = "prob-box",
                            div(class = "prob-label", "Input features"),
                            div(class = "prob-value", style = "color:#5383e8; font-size:26px;", "21")
                        )
                    ),
                    column(3,
                        div(class = "prob-box",
                            div(class = "prob-label", "Min. features out"),
                            div(class = "prob-value", style = "color:#27ae60; font-size:26px;", "9")
                        )
                    ),
                    column(3,
                        div(class = "prob-box",
                            div(class = "prob-label", "Methods compared"),
                            div(class = "prob-value", style = "color:#5383e8; font-size:26px;", "5")
                        )
                    ),
                    column(3,
                        div(class = "prob-box",
                            div(class = "prob-label", "AUC loss (LASSO vs full)"),
                            div(class = "prob-value", style = "color:#27ae60; font-size:26px;", "< 0.1%")
                        )
                    )
                )
            ),

            # ── Method cards ──────────────────────────────────────────────────
            div(class = "card",
                p(class = "sidebar-section-title", "Selection Methods"),
                fluidRow(
                    column(4,
                        div(style = "background:#13131e; border:1px solid #2a2a3e; border-radius:8px; padding:16px; margin-bottom:12px;",
                            div(style = "color:#5383e8; font-weight:700; font-size:14px; margin-bottom:4px;", "RFE"),
                            div(style = "color:#6a7590; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;", "Recursive Feature Elimination"),
                            p(style = "color:#9aaccc; font-size:12px; line-height:1.7; margin:0;",
                                "Trains Logistic Regression repeatedly, ranks features by coefficient magnitude,",
                                " removes the weakest, and repeats. 5-fold CV selects the optimal subset size."
                            )
                        ),
                        div(style = "background:#13131e; border:1px solid #2a2a3e; border-radius:8px; padding:16px;",
                            div(style = "color:#5383e8; font-weight:700; font-size:14px; margin-bottom:4px;", "Forward Stepwise"),
                            div(style = "color:#6a7590; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;", "AIC-Based Greedy Selection"),
                            p(style = "color:#9aaccc; font-size:12px; line-height:1.7; margin:0;",
                                "Starts with an intercept-only model, greedily adds the feature that most improves",
                                " AIC at each step. Once added a feature stays — no shrinkage of redundant coefficients."
                            )
                        )
                    ),
                    column(4,
                        div(style = "background:#13131e; border:1px solid #2a2a3e; border-radius:8px; padding:16px; margin-bottom:12px;",
                            div(style = "color:#5383e8; font-weight:700; font-size:14px; margin-bottom:4px;", "LASSO (L1)"),
                            div(style = "color:#6a7590; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;", "Least Absolute Shrinkage and Selection"),
                            p(style = "color:#9aaccc; font-size:12px; line-height:1.7; margin:0;",
                                "Logistic Regression with an L1 penalty that shrinks redundant coefficients to exactly zero.",
                                " From correlated groups, keeps the strongest signal and eliminates the rest.",
                                " Lambda tuned via 5-fold CV."
                            )
                        ),
                        div(style = "background:#13131e; border:1px solid #2a2a3e; border-radius:8px; padding:16px;",
                            div(style = "color:#5383e8; font-weight:700; font-size:14px; margin-bottom:4px;", "Elastic Net (L1+L2)"),
                            div(style = "color:#6a7590; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;", "Ridge + LASSO Combined"),
                            p(style = "color:#9aaccc; font-size:12px; line-height:1.7; margin:0;",
                                "Combines LASSO (sparsity) and Ridge (coefficient shrinkage). Unlike LASSO, can retain",
                                " several correlated features at reduced coefficients. Alpha (L1/L2 mix) searched across 0–1."
                            )
                        )
                    ),
                    column(4,
                        div(style = "background:#13131e; border:1px solid #2a2a3e; border-radius:8px; padding:16px;",
                            div(style = "color:#5383e8; font-weight:700; font-size:14px; margin-bottom:4px;", "RF Importance"),
                            div(style = "color:#6a7590; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;", "Random Forest Filter"),
                            p(style = "color:#9aaccc; font-size:12px; line-height:1.7; margin:0;",
                                "Uses importance scores from the Scenario 1 Random Forest as a filter — features above the",
                                " mean importance threshold are retained. No extra training needed.",
                                " Distributes importance among correlated features rather than zeroing them out (unlike LASSO)."
                            )
                        )
                    )
                )
            ),

            # ── Performance comparison ─────────────────────────────────────────
            div(class = "card",
                p(class = "sidebar-section-title", "Performance Comparison"),
                p(style = "color:#9aaccc; font-size:13px; margin-bottom:16px;",
                    "All five methods achieve nearly the same AUC-ROC as the full-feature baseline despite using only a subset.",
                    " The lollipop chart shows how many features each method kept (label) and where it lands relative to the LR-all baseline (dashed line)."
                ),
                DTOutput("feat_perf_table"),
                br(),
                plotOutput("lollipop_plot", height = "340px")
            ),

            # ── Feature overlap heatmap ────────────────────────────────────────
            div(class = "card",
                p(class = "sidebar-section-title", "Feature Selection Overlap"),
                p(style = "color:#9aaccc; font-size:13px; margin-bottom:4px;",
                    "Each row is a feature; each column is a selection method.",
                    " Green = selected by that method, dark = eliminated.",
                    " The number (x/5) shows how many of the five methods agreed on that feature."
                ),
                plotOutput("heatmap_plot", height = "500px")
            ),

            # ── Key findings ───────────────────────────────────────────────────
            div(class = "card",
                p(class = "sidebar-section-title", "Key Findings"),
                fluidRow(
                    column(4,
                        div(style = "background:#13131e; border:1px solid #27ae60; border-radius:8px; padding:16px; min-height:120px;",
                            div(style = "color:#27ae60; font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;",
                                "Consensus Core — selected by all 5 methods"),
                            uiOutput("fs_consensus_5")
                        )
                    ),
                    column(4,
                        div(style = "background:#13131e; border:1px solid #5383e8; border-radius:8px; padding:16px; min-height:120px;",
                            div(style = "color:#5383e8; font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;",
                                "Near-Consensus — selected by 4 / 5 methods"),
                            uiOutput("fs_consensus_4")
                        )
                    ),
                    column(4,
                        div(style = "background:#13131e; border:1px solid #e84057; border-radius:8px; padding:16px; min-height:120px;",
                            div(style = "color:#e84057; font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;",
                                "Never Selected — eliminated by all 5 methods"),
                            uiOutput("fs_consensus_0")
                        )
                    )
                ),
                br(),
                p(style = "color:#c8d0e0; line-height:1.8; margin-bottom:10px;",
                    "All five methods converge on the same answer: the predictive signal in this dataset is highly concentrated.",
                    " Starting from 21 features, every method reduces the set substantially — LASSO and RF Importance down to",
                    " just 9 — with almost ", tags$b("no loss in AUC-ROC"), " (84.3–84.4% vs. 84.4% full-feature baseline).",
                    " Roughly half the features are redundant."
                ),
                p(style = "color:#9aaccc; line-height:1.8; margin:0;",
                    tags$b("firstblood"), " was never selected by any method — consistent with published LoL research",
                    " where first blood has weak win-rate correlation compared to tower or gold leads.",
                    " Kill pressure metrics similarly add no independent signal once economic and objective features are accounted for."
                )
            ),

            br()
        )
    ),

    # ── Tab 5: Data Explorer ────────────────────────────────────────────────────
    tabPanel("Data Explorer",
        fluidRow(

            # ── Left sidebar ─────────────────────────────────────────────────────
            column(3,
                div(class = "predictor-sidebar", style = "padding-top:16px;",

                    # Load section
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "Load Data"),
                        fileInput("data_upload", NULL,
                            accept      = c(".csv", "text/csv"),
                            placeholder = "No file selected",
                            buttonLabel = "Browse",
                            width       = "100%"),
                        div(class = "data-or-divider", "or"),
                        actionButton("data_use_example", "Use Built-in Dataset",
                            class = "btn-primary",
                            style = "width:100%;")
                    ),

                    # Status + stats (NULL when no data)
                    uiOutput("data_sidebar_stats")
                )
            ),

            # ── Right main content ────────────────────────────────────────────────
            column(9,
                div(class = "data-main-area data-view",
                    uiOutput("data_no_data_hint"),
                    tabsetPanel(id = "data_inner_tabs",
                        tabPanel("Table",
                            DTOutput("data_table_full")
                        ),
                        tabPanel("Correlations",
                            uiOutput("data_corr_plot_ui"),
                            uiOutput("data_corrbar_plot_ui")
                        ),
                        tabPanel("Outcome Analysis",
                            plotOutput("data_winrate_plot", height = "700px")
                        ),
                        tabPanel("Distribution",
                            uiOutput("data_dist_controls"),
                            plotOutput("data_dist_plot", height = "400px")
                        )
                    )
                )
            )
        )
    )

)

# ── Server ─────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

    rv <- reactiveValues(input_row = NULL, game_info = NULL,
                         last_selected_game = NULL, startup_done = FALSE)

    # ── Match Predictor: Browse ─────────────────────────────────────────────────
    observe({
        updateSelectInput(session, "browse_league",
            choices  = sort(unique(game_lookup$league)),
            selected = "LCK")
    })

    observeEvent(input$browse_league, {
        teams_in_league <- game_lookup %>%
            filter(league == input$browse_league) %>%
            pull(teamname) %>% unique() %>% sort()
        is_startup <- !isolate(rv$startup_done)
        updateSelectInput(session, "browse_team",
            choices  = c("— select team —" = "", teams_in_league),
            selected = if (is_startup) "T1" else "")
        if (!is_startup) {
            rv$input_row <- NULL
            rv$game_info <- NULL
        }
    })

    # Single browse_team observer — handles startup auto-load and user changes
    observeEvent(input$browse_team, {
        if (nchar(input$browse_team) == 0) {
            rv$input_row <- NULL
            rv$game_info <- NULL
            return()
        }
        if (!isolate(rv$startup_done)) {
            # First non-empty team = startup: load T1 vs KT Rolster 2025-01-24
            rv$startup_done <- TRUE
            demo <- game_lookup %>%
                filter(teamname == "T1", league == "LCK",
                       opp_teamname == "KT Rolster",
                       as.character(date) == "2025-01-24") %>%
                slice(1)
            if (nrow(demo) == 0) return()
            rv$game_info          <- demo
            rv$last_selected_game <- demo
            rv$input_row <- demo %>%
                rename(side = side_encoded) %>%
                select(all_of(feature_names)) %>%
                as.data.frame()
            updateRadioButtons(session, "side",        selected = as.character(demo$side_encoded))
            updateSliderInput(session, "golddiffat15", value = round(demo$golddiffat15))
            updateSliderInput(session, "xpdiffat15",   value = round(demo$xpdiffat15))
            updateSliderInput(session, "csdiffat15",   value = round(demo$csdiffat15))
            updateCheckboxInput(session, "firstblood",  value = as.logical(demo$firstblood))
            updateCheckboxInput(session, "firstdragon", value = as.logical(demo$firstdragon))
            updateCheckboxInput(session, "firstherald", value = as.logical(demo$firstherald))
            updateCheckboxInput(session, "firsttower",  value = as.logical(demo$firsttower))
            updateSliderInput(session, "grub_diff", value = as.integer(demo$grub_diff))
            updateSliderInput(session, "wr_diff",   value = round(demo$winrate_diff * 100, 1))
            game_label <- sprintf("%s vs %s · %s", demo$teamname, demo$opp_teamname,
                                  format(as.Date(demo$date), "%Y-%m-%d"))
            updateSelectInput(session, "feature_preset",
                choices  = c(setNames(game_label, game_label), "Custom" = "__custom__"),
                selected = game_label)
        } else {
            # User manually changed team — clear state
            rv$input_row <- NULL
            rv$game_info <- NULL
        }
    })

    browse_games_full <- reactive({
        req(input$browse_league, nchar(input$browse_team) > 0)
        game_lookup %>%
            filter(league == input$browse_league,
                   teamname == input$browse_team) %>%
            arrange(desc(date))
    })

    output$game_table <- renderDT({
        req(nchar(input$browse_team) > 0)
        browse_games_full() %>%
            transmute(
                Date     = as.character(date),
                Opponent = opp_teamname,
                Side     = side_label,
                `Gold @15` = golddiffat15,
                Result   = result_label
            ) %>%
            datatable(selection = "single", rownames = FALSE,
                options = list(
                    dom        = "tp",
                    pageLength = 5,
                    autoWidth  = FALSE,
                    scrollX    = FALSE,
                    initComplete = JS("function(settings) {
                        var w = $(this.api().table().container());
                        w.find('.row, .row > div').css({
                            'display': 'block',
                            'float': 'none',
                            'width': '100%'
                        });
                        w.find('.dataTables_paginate').css({
                            'float': 'none',
                            'width': '100%',
                            'display': 'block',
                            'clear': 'both',
                            'text-align': 'center',
                            'padding-top': '8px'
                        });
                    }")
                )) %>%
            formatStyle("Result",
                color = styleEqual(c("WIN","LOSS"), c("#27ae60","#e84057")),
                fontWeight = "bold") %>%
            formatStyle("Gold @15",
                color = styleInterval(c(-1,0), c("#e84057","#c8d0e0","#27ae60")))
    })

    observeEvent(input$game_table_rows_selected, {
        idx <- input$game_table_rows_selected
        req(length(idx) > 0)
        selected              <- browse_games_full()[idx, ]
        rv$game_info          <- selected
        rv$last_selected_game <- selected
        rv$input_row <- selected %>%
            rename(side = side_encoded) %>%
            select(all_of(feature_names)) %>%
            as.data.frame()
        updateRadioButtons(session, "side",    selected = as.character(selected$side_encoded))
        updateSliderInput(session, "golddiffat15", value = round(selected$golddiffat15))
        updateSliderInput(session, "xpdiffat15",   value = round(selected$xpdiffat15))
        updateSliderInput(session, "csdiffat15",   value = round(selected$csdiffat15))
        updateCheckboxInput(session, "firstblood",  value = as.logical(selected$firstblood))
        updateCheckboxInput(session, "firstdragon", value = as.logical(selected$firstdragon))
        updateCheckboxInput(session, "firstherald", value = as.logical(selected$firstherald))
        updateCheckboxInput(session, "firsttower",  value = as.logical(selected$firsttower))
        updateSliderInput(session, "grub_diff", value = as.integer(selected$grub_diff))
        updateSliderInput(session, "wr_diff",   value = round(selected$winrate_diff * 100, 1))
        game_label <- sprintf("%s vs %s · %s",
            selected$teamname, selected$opp_teamname,
            format(as.Date(selected$date), "%Y-%m-%d"))
        updateSelectInput(session, "feature_preset",
            choices  = c(setNames(game_label, game_label), "Custom" = "__custom__"),
            selected = game_label)
    })

    # Live prediction — updates automatically whenever any feature input changes
    observe({
        row <- as.data.frame(t(train_means))
        row$golddiffat15 <- input$golddiffat15
        row$xpdiffat15   <- input$xpdiffat15
        row$csdiffat15   <- input$csdiffat15
        row$firsttower   <- as.integer(input$firsttower)
        row$firstherald  <- as.integer(input$firstherald)
        row$firstdragon  <- as.integer(input$firstdragon)
        row$firstblood   <- as.integer(input$firstblood)
        row$grub_diff    <- as.integer(input$grub_diff)
        row$side         <- as.numeric(input$side)
        row$winrate_diff <- input$wr_diff / 100
        rv$input_row     <- row
    })

    # Reset preset to Custom and hide game banner when user manually tweaks a feature
    observeEvent(input$user_tweaked_feature, {
        updateSelectInput(session, "feature_preset", selected = "__custom__")
        rv$game_info <- NULL
    })

    # Re-apply game features when user picks the game preset from dropdown
    observeEvent(input$feature_preset, {
        req(input$feature_preset != "__custom__")
        gi <- rv$last_selected_game
        req(!is.null(gi))
        updateRadioButtons(session, "side",        selected = as.character(gi$side_encoded))
        updateSliderInput(session, "golddiffat15", value = round(gi$golddiffat15))
        updateSliderInput(session, "xpdiffat15",   value = round(gi$xpdiffat15))
        updateSliderInput(session, "csdiffat15",   value = round(gi$csdiffat15))
        updateCheckboxInput(session, "firstblood",  value = as.logical(gi$firstblood))
        updateCheckboxInput(session, "firstdragon", value = as.logical(gi$firstdragon))
        updateCheckboxInput(session, "firstherald", value = as.logical(gi$firstherald))
        updateCheckboxInput(session, "firsttower",  value = as.logical(gi$firsttower))
        updateSliderInput(session, "grub_diff", value = as.integer(gi$grub_diff))
        updateSliderInput(session, "wr_diff",   value = round(gi$winrate_diff * 100, 1))
        rv$game_info <- gi
        rv$input_row <- gi %>%
            rename(side = side_encoded) %>%
            select(all_of(feature_names)) %>%
            as.data.frame()
    }, ignoreInit = TRUE)

    # ── Prediction ──────────────────────────────────────────────────────────────
    probs <- reactive({
        req(rv$input_row)
        raw <- rv$input_row
        scaled <- predict(preproc, raw)
        list(
            lr   = round(predict(lr_model,   scaled, type = "prob")[,"Win"] * 100, 1),
            rf   = round(predict(rf_model,   raw,    type = "prob")[,"Win"] * 100, 1),
            nb   = round(predict(nb_model,   scaled, type = "prob")[,"Win"] * 100, 1),
            knn  = round(predict(knn_model,  scaled, type = "prob")[,"Win"] * 100, 1),
            cart = round(predict(cart_model, raw,    type = "prob")[,"Win"] * 100, 1)
        )
    })

    fmt <- function(p) paste0(p, "%")
    output$prob_lr   <- renderText({ fmt(probs()$lr)   })
    output$prob_rf   <- renderText({ fmt(probs()$rf)   })
    output$prob_nb   <- renderText({ fmt(probs()$nb)   })
    output$prob_knn  <- renderText({ fmt(probs()$knn)  })
    output$prob_cart <- renderText({ fmt(probs()$cart) })
    output$prob_avg  <- renderText({ fmt(round(mean(unlist(probs())), 1)) })

    output$game_banner <- renderUI({
        if (is.null(rv$game_info)) {
            return(div(class = "game-banner",
                div(class = "banner-teams", "Custom Scenario"),
                div(class = "banner-meta",
                    "Manually configured — no real match selected")
            ))
        }
        g       <- rv$game_info
        correct <- (g$result == 1 && probs()$lr >= 50) ||
                   (g$result == 0 && probs()$lr <  50)
        div(class = "game-banner",
            div(class = "banner-teams",
                sprintf("%s vs %s", g$teamname, g$opp_teamname)),
            div(class = "banner-meta",
                sprintf("%s  ·  %s Side  ·  %s", g$league, g$side_label, g$date)),
            div(style = "margin-top:10px;",
                span(class = if (g$result_label == "WIN") "badge-win" else "badge-loss",
                     g$result_label),
                span(class = if (correct) "badge-correct" else "badge-wrong",
                     if (correct) "LR: Correct" else "LR: Wrong")
            )
        )
    })

    output$prob_bar <- renderPlot(bg = "#1c1c2e", {
        req(rv$input_row)
        p <- probs()
        tibble(
            Model = c("LR","RF","NB","KNN","CART"),
            Prob  = c(p$lr, p$rf, p$nb, p$knn, p$cart)
        ) %>%
            mutate(Model = reorder(Model, Prob)) %>%
            ggplot(aes(x = Prob, y = Model)) +
            geom_col(width = 0.5, fill = "#1e3a6a") +
            geom_col(aes(x = pmin(Prob, Prob)), width = 0.5, fill = ACCENT) +
            geom_text(aes(label = paste0(Prob, "%")), hjust = -0.2, size = 3.8,
                      color = "#9aaccc", fontface = "bold") +
            geom_vline(xintercept = 50, linetype = "dashed", color = "#2a2a3e", linewidth = 0.8) +
            annotate("text", x = 50.5, y = 0.5, label = "50%", color = "#6a7590",
                     size = 3, hjust = 0) +
            scale_x_continuous(limits = c(0, 115), expand = c(0, 0)) +
            labs(x = "Win Probability (%)", y = NULL) +
            dark_theme() +
            theme(panel.grid.major.y = element_blank(),
                  axis.text.y        = element_text(color = "#9aaccc", face = "bold"))
    })

    # ── Hyperparameter Tuning tab ───────────────────────────────────────────────

    snap_to <- function(val, tested) tested[which.min(abs(tested - val))]

    # Reset slider/select to CV best value
    observeEvent(input$tune_reset, {
        m <- input$tune_model
        if (m == "RF")   updateSliderInput(session, "rf_mtry",   value = rf_model$bestTune$mtry)
        if (m == "KNN")  updateSliderInput(session, "knn_k",     value = knn_model$bestTune$k)
        if (m == "CART") updateSelectInput(session, "cart_cp",   selected = as.character(cart_model$bestTune$cp))
        if (m == "NB") {
            updateRadioButtons(session, "nb_kernel", selected = as.character(nb_model$bestTune$usekernel))
            updateSliderInput(session,  "nb_adjust", value    = nb_model$bestTune$adjust)
        }
    })

    # Left panel: model description
    output$tune_description <- renderUI({
        meta <- model_meta[[input$tune_model]]
        tagList(
            p(meta$name,
              style = "font-size:14px; font-weight:600; color:#ffffff; margin:0 0 4px 0;"),
            p(meta$desc,
              style = "font-size:12px; color:#9aaccc; margin:0; line-height:1.5;")
        )
    })

    # Left panel: overall CV best badge
    output$tune_best <- renderUI({
        m   <- input$tune_model
        col <- model_meta[[m]]$color

        best_line <- function(label, value) {
            div(style = "display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #2a2a3e;",
                span(style = "font-size:12px; color:#6a7590; font-weight:500;", label),
                span(style = paste0("font-size:13px; font-weight:700; color:", col), value)
            )
        }

        if (m == "LR") return(
            div(style = "font-size:12px; color:#9aaccc;",
                "Standard GLM — no hyperparameter search performed.")
        )
        if (m == "RF")   return(best_line("mtry", rf_model$bestTune$mtry))
        if (m == "KNN")  return(best_line("k", knn_model$bestTune$k))
        if (m == "CART") return(best_line("cp", formatC(cart_model$bestTune$cp, format="g")))
        if (m == "NB")   return(tagList(
            best_line("Density", if (nb_model$bestTune$usekernel) "Kernel density" else "Gaussian"),
            best_line("Adjust",  nb_model$bestTune$adjust)
        ))
    })

    # ── Interactive slider rendered per model ────────────────────────────────────
    output$tune_slider <- renderUI({
        m   <- input$tune_model
        col <- model_meta[[m]]$color

        reset_btn <- div(
            style = "display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;",
            p(class = "sidebar-section-title", style = "margin:0;", "Adjust Parameter"),
            actionButton("tune_reset", "Reset to Best", class = "btn-reset-best")
        )

        if (m == "LR") {
            return(div(style = "color:#6a7590; font-size:13px; font-style:italic; padding:4px 0;",
                "Standard GLM — no hyperparameter to tune. Showing model coefficients below."))
        }

        slider_css <- sprintf(
            ".irs--shiny .irs-bar{background:%s}.irs--shiny .irs-handle{background:%s;border-color:%s}.irs--shiny .irs-single{background:%s}",
            col, col, col, col)

        if (m == "RF") {
            tested <- rf_model$results$mtry
            tagList(
                tags$style(slider_css), reset_btn,
                div(style = "display:flex; justify-content:space-between; align-items:baseline;",
                    tags$label(style = "font-size:12px; color:#9aaccc; font-weight:500;",
                               "mtry — features per split"),
                    span(style = paste0("font-size:13px; font-weight:700; color:", col),
                         textOutput("rf_mtry_label", inline = TRUE))
                ),
                sliderInput("rf_mtry", NULL,
                    min = min(tested), max = max(tested),
                    value = rf_model$bestTune$mtry, step = 1, width = "100%"),
                div(style = "font-size:11px; color:#6a7590; margin-top:-4px;",
                    paste("Tested:", paste(tested, collapse = ", ")))
            )
        } else if (m == "KNN") {
            tested <- knn_model$results$k
            tagList(
                tags$style(slider_css), reset_btn,
                div(style = "display:flex; justify-content:space-between; align-items:baseline;",
                    tags$label(style = "font-size:12px; color:#9aaccc; font-weight:500;",
                               "k — number of neighbors"),
                    span(style = paste0("font-size:13px; font-weight:700; color:", col),
                         textOutput("knn_k_label", inline = TRUE))
                ),
                sliderInput("knn_k", NULL,
                    min = min(tested), max = max(tested),
                    value = knn_model$bestTune$k, step = 2, width = "100%"),
                div(style = "font-size:11px; color:#6a7590; margin-top:-4px;",
                    paste("Tested:", paste(tested, collapse = ", ")))
            )
        } else if (m == "CART") {
            tested  <- cart_model$results$cp
            choices <- setNames(as.character(tested), formatC(tested, format = "g", digits = 2))
            tagList(
                reset_btn,
                tags$label(style = "font-size:12px; color:#9aaccc; font-weight:500;",
                           "cp — complexity parameter"),
                selectInput("cart_cp", NULL,
                    choices  = choices,
                    selected = as.character(cart_model$bestTune$cp),
                    width    = "100%"),
                div(style = "font-size:11px; color:#6a7590; margin-top:2px;",
                    "Smaller = deeper tree · Larger = more pruning")
            )
        } else if (m == "NB") {
            tagList(
                tags$style(slider_css), reset_btn,
                tags$label(style = "font-size:12px; color:#9aaccc; font-weight:500; margin-bottom:4px; display:block;",
                           "Density estimation"),
                radioButtons("nb_kernel", NULL,
                    choices  = c("Gaussian (parametric)" = "FALSE",
                                 "Kernel density (non-parametric)" = "TRUE"),
                    selected = as.character(nb_model$bestTune$usekernel),
                    inline   = FALSE),
                div(style = "display:flex; justify-content:space-between; align-items:baseline; margin-top:10px;",
                    tags$label(style = "font-size:12px; color:#9aaccc; font-weight:500;",
                               "Bandwidth adjust"),
                    span(style = paste0("font-size:13px; font-weight:700; color:", col),
                         textOutput("nb_adj_label", inline = TRUE))
                ),
                sliderInput("nb_adjust", NULL,
                    min = 0.5, max = 1.5, value = nb_model$bestTune$adjust,
                    step = 0.5, width = "100%")
            )
        }
    })

    # Slider value labels (show snapped tested value)
    output$rf_mtry_label <- renderText({
        req(input$rf_mtry)
        paste0("mtry = ", snap_to(input$rf_mtry, rf_model$results$mtry))
    })
    output$knn_k_label <- renderText({
        req(input$knn_k)
        paste0("k = ", snap_to(input$knn_k, knn_model$results$k))
    })
    output$nb_adj_label <- renderText({
        req(input$nb_adjust)
        paste0("adjust = ", input$nb_adjust)
    })

    # ── Reactive: results row for currently selected parameter value ─────────────
    selected_row <- reactive({
        m <- input$tune_model
        if (m == "LR") return(lr_model$results[1, ])
        if (m == "RF") {
            val <- if (is.null(input$rf_mtry)) rf_model$bestTune$mtry else input$rf_mtry
            sv  <- snap_to(val, rf_model$results$mtry)
            return(rf_model$results %>% filter(mtry == sv))
        }
        if (m == "KNN") {
            val <- if (is.null(input$knn_k)) knn_model$bestTune$k else input$knn_k
            sv  <- snap_to(val, knn_model$results$k)
            return(knn_model$results %>% filter(k == sv))
        }
        if (m == "CART") {
            val <- if (is.null(input$cart_cp)) as.character(cart_model$bestTune$cp) else input$cart_cp
            return(cart_model$results %>% filter(abs(cp - as.numeric(val)) < 1e-10))
        }
        if (m == "NB") {
            uk  <- as.logical(if (is.null(input$nb_kernel)) as.character(nb_model$bestTune$usekernel) else input$nb_kernel)
            adj <- if (is.null(input$nb_adjust)) nb_model$bestTune$adjust else as.numeric(input$nb_adjust)
            return(nb_model$results %>% filter(usekernel == uk, abs(adjust - adj) < 0.01))
        }
    })

    # ── Live metrics for selected parameter value ────────────────────────────────
    output$tune_metrics <- renderUI({
        m   <- input$tune_model
        col <- model_meta[[m]]$color
        sr  <- selected_row()
        if (is.null(sr) || nrow(sr) == 0) return(NULL)

        fluidRow(style = "margin-bottom:16px;",
            column(4, style = "padding-right:6px;",
                div(style = paste0("background:#1c1c2e; border:1px solid #2a2a3e; border-top:3px solid ", col,
                                   "; border-radius:10px; padding:14px 18px;"),
                    div(style = "font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; color:#6a7590; margin-bottom:4px;", "AUC-ROC"),
                    div(style = paste0("font-size:28px; font-weight:700; color:", col, "; letter-spacing:-0.5px;"),
                        sprintf("%.2f%%", sr$ROC[1] * 100))
                )
            ),
            column(4, style = "padding:0 3px;",
                div(style = "background:#1c1c2e; border:1px solid #2a2a3e; border-top:3px solid #2a2a3e; border-radius:10px; padding:14px 18px;",
                    div(style = "font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; color:#6a7590; margin-bottom:4px;", "Sensitivity"),
                    div(style = "font-size:28px; font-weight:700; color:#c8d0e0; letter-spacing:-0.5px;",
                        sprintf("%.1f%%", sr$Sens[1] * 100))
                )
            ),
            column(4, style = "padding-left:6px;",
                div(style = "background:#1c1c2e; border:1px solid #2a2a3e; border-top:3px solid #2a2a3e; border-radius:10px; padding:14px 18px;",
                    div(style = "font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; color:#6a7590; margin-bottom:4px;", "Specificity"),
                    div(style = "font-size:28px; font-weight:700; color:#c8d0e0; letter-spacing:-0.5px;",
                        sprintf("%.1f%%", sr$Spec[1] * 100))
                )
            )
        )
    })

    # ── Tuning curve with moveable selected point ────────────────────────────────
    output$tune_plot <- renderPlot(bg = "#1c1c2e", {
        m   <- input$tune_model
        col <- model_meta[[m]]$color
        sr  <- selected_row()

        if (m == "LR") {
            sm   <- summary(lr_model$finalModel)$coefficients
            or_df <- data.frame(
                feature  = rownames(sm),
                estimate = sm[, "Estimate"],
                se       = sm[, "Std. Error"],
                p        = sm[, "Pr(>|z|)"]
            ) %>%
                filter(feature != "(Intercept)") %>%
                mutate(
                    OR      = exp(estimate),
                    CI_low  = exp(estimate - 1.96 * se),
                    CI_high = exp(estimate + 1.96 * se),
                    sig     = p < 0.05
                ) %>%
                arrange(OR) %>%
                slice_max(abs(log(OR)), n = 20) %>%
                arrange(OR)

            return(
                ggplot(or_df, aes(x = OR, y = reorder(feature, OR), color = sig)) +
                geom_vline(xintercept = 1, linetype = "dashed", color = "#2a2a3e", linewidth = 0.8) +
                geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                               height = 0.35, linewidth = 0.7) +
                geom_point(size = 3.5) +
                scale_x_log10(labels = function(x) sprintf("%.1fx", x)) +
                scale_color_manual(
                    values = c("FALSE" = "#2e3a5e", "TRUE" = ACCENT),
                    labels = c("FALSE" = "p >= 0.05  (not significant)",
                               "TRUE"  = "p < 0.05  (significant)"),
                    name = NULL
                ) +
                labs(
                    title    = "Logistic Regression — Odds Ratios",
                    subtitle = "Each point = how much the feature multiplies the odds of winning · bars = 95% CI · log scale",
                    x = "Odds Ratio (log scale)", y = NULL
                ) +
                dark_theme() +
                theme(legend.position    = "top",
                      panel.grid.major.y = element_line(color = "#2a2a3e"))
            )
        }

        if (m == "NB") {
            sel_uk  <- as.logical(if (is.null(input$nb_kernel)) as.character(nb_model$bestTune$usekernel) else input$nb_kernel)
            sel_adj <- if (is.null(input$nb_adjust)) nb_model$bestTune$adjust else as.numeric(input$nb_adjust)
            best_uk  <- nb_model$bestTune$usekernel
            best_adj <- nb_model$bestTune$adjust

            return(
                nb_model$results %>%
                mutate(
                    Density  = ifelse(usekernel, "Kernel density", "Gaussian"),
                    is_sel   = usekernel == sel_uk  & abs(adjust - sel_adj)  < 0.01,
                    is_best  = usekernel == best_uk & abs(adjust - best_adj) < 0.01
                ) %>%
                ggplot(aes(x = factor(adjust), y = ROC * 100, color = Density, group = Density)) +
                geom_line(linewidth = 1.2, alpha = 0.6) +
                geom_point(size = 3.5, alpha = 0.6) +
                geom_point(data = . %>% filter(is_sel),
                           size = 10, shape = 21, fill = col, color = "white", stroke = 2) +
                geom_point(data = . %>% filter(is_best & !is_sel),
                           size = 10, shape = 21, fill = "white", color = "#e8a838", stroke = 2) +
                scale_color_manual(values = c("Gaussian" = col, "Kernel density" = "#e8a838"), name = NULL) +
                labs(title = "Naive Bayes — CV AUC-ROC",
                     subtitle = "Filled = selected · Gold ring = CV best",
                     x = "Bandwidth adjust", y = "CV AUC-ROC (%)") +
                dark_theme() +
                theme(legend.position = "top")
            )
        }

        # RF, KNN, CART
        d <- switch(m,
            RF   = rf_model$results   %>% mutate(param = mtry, is_best = mtry == rf_model$bestTune$mtry),
            KNN  = knn_model$results  %>% mutate(param = k,    is_best = k    == knn_model$bestTune$k),
            CART = cart_model$results %>% mutate(param = cp,   is_best = cp   == cart_model$bestTune$cp)
        )
        sel_param <- switch(m,
            RF   = snap_to(if(is.null(input$rf_mtry)) rf_model$bestTune$mtry else input$rf_mtry, d$param),
            KNN  = snap_to(if(is.null(input$knn_k))   knn_model$bestTune$k   else input$knn_k,  d$param),
            CART = as.numeric(if(is.null(input$cart_cp)) as.character(cart_model$bestTune$cp) else input$cart_cp)
        )
        sel_row  <- d %>% filter(abs(param - sel_param) < 1e-10)
        best_row <- d %>% filter(is_best)
        titles   <- c(RF="Random Forest — mtry", KNN="KNN — k (neighbors)", CART="Decision Tree — cp")
        x_labels <- c(RF="mtry", KNN="k", CART="cp (log scale)")

        p <- ggplot(d, aes(x = param, y = ROC * 100)) +
            geom_line(color = col, linewidth = 1.2, alpha = 0.5) +
            geom_point(color = col, size = 3.5, alpha = 0.5) +
            geom_point(data = sel_row, aes(x = param, y = ROC * 100),
                       size = 11, shape = 21, fill = col, color = "white", stroke = 2.5,
                       inherit.aes = FALSE) +
            geom_text(data = sel_row, aes(x = param, y = ROC * 100,
                       label = sprintf("%.2f%%", ROC * 100)),
                      vjust = -1.6, fontface = "bold", color = col, size = 4,
                      inherit.aes = FALSE)

        if (nrow(best_row) > 0 && abs(best_row$param[1] - sel_param) > 1e-10) {
            p <- p +
                geom_point(data = best_row, aes(x = param, y = ROC * 100),
                           size = 11, shape = 21, fill = "white",
                           color = "#e8a838", stroke = 2.5, inherit.aes = FALSE) +
                annotate("text", x = best_row$param[1],
                         y = best_row$ROC[1] * 100 - diff(range(d$ROC * 100)) * 0.18,
                         label = "* best", color = "#e8a838", size = 3.2, fontface = "bold")
        }

        p <- p +
            labs(title    = titles[m],
                 subtitle = "Filled circle = selected · Gold ring = CV best (if different)",
                 x = x_labels[m], y = "CV AUC-ROC (%)") +
            dark_theme()

        if (m == "CART") p <- p + scale_x_log10()
        p
    })

    # CV results table (highlight selected row)
    output$tune_table <- renderDT({
        m  <- input$tune_model
        sr <- selected_row()

        tbl <- switch(m,
            LR = lr_model$results %>%
                transmute(`CV AUC-ROC` = sprintf("%.2f%%", ROC*100),
                          Sensitivity  = sprintf("%.2f%%", Sens*100),
                          Specificity  = sprintf("%.2f%%", Spec*100)),
            RF = rf_model$results %>%
                mutate(
                    Selected = ifelse(mtry == sr$mtry[1], "▶", ""),
                    Best     = ifelse(mtry == rf_model$bestTune$mtry, "★", "")
                ) %>%
                transmute(mtry, `CV AUC-ROC`=sprintf("%.2f%%",ROC*100),
                          Sensitivity=sprintf("%.2f%%",Sens*100),
                          Specificity=sprintf("%.2f%%",Spec*100), Selected, Best),
            KNN = knn_model$results %>%
                mutate(
                    Selected = ifelse(k == sr$k[1], "▶", ""),
                    Best     = ifelse(k == knn_model$bestTune$k, "★", "")
                ) %>%
                transmute(k, `CV AUC-ROC`=sprintf("%.2f%%",ROC*100),
                          Sensitivity=sprintf("%.2f%%",Sens*100),
                          Specificity=sprintf("%.2f%%",Spec*100), Selected, Best),
            CART = cart_model$results %>%
                mutate(
                    Selected = ifelse(abs(cp - sr$cp[1]) < 1e-10, "▶", ""),
                    Best     = ifelse(cp == cart_model$bestTune$cp, "★", "")
                ) %>%
                transmute(cp, `CV AUC-ROC`=sprintf("%.2f%%",ROC*100),
                          Sensitivity=sprintf("%.2f%%",Sens*100),
                          Specificity=sprintf("%.2f%%",Spec*100), Selected, Best),
            NB = nb_model$results %>%
                mutate(
                    Density  = ifelse(usekernel, "Kernel", "Gaussian"),
                    Selected = ifelse(usekernel == sr$usekernel[1] & abs(adjust-sr$adjust[1])<0.01, "▶",""),
                    Best     = ifelse(usekernel == nb_model$bestTune$usekernel &
                                      abs(adjust - nb_model$bestTune$adjust) < 0.01, "★","")
                ) %>%
                transmute(Density, Adjust=adjust,
                          `CV AUC-ROC`=sprintf("%.2f%%",ROC*100),
                          Sensitivity=sprintf("%.2f%%",Sens*100),
                          Specificity=sprintf("%.2f%%",Spec*100), Selected, Best)
        )
        col <- model_meta[[m]]$color
        dt  <- datatable(tbl, rownames = FALSE,
            options = list(dom = "t", ordering = FALSE, pageLength = 20))
        if ("Selected" %in% names(tbl)) {
            dt <- dt %>%
                formatStyle("Selected", color = col,      fontWeight = "bold", fontSize = "14px", textAlign = "center") %>%
                formatStyle("Best",     color = "#e8a838", fontWeight = "bold", fontSize = "16px", textAlign = "center")
        }
        dt
    })

    # ── LR extras: confusion matrix + probability distribution ──────────────────
    output$tune_lr_extra <- renderUI({
        missing <- is.null(lr_cm_obj) || is.null(lr_test_df)
        if (missing) {
            return(div(class = "card", style = "padding:20px;",
                p("Re-run analysis.Rmd to generate lr_cm.rds and lr_test_df.rds",
                  style = "color:#6a7590; font-size:13px;")))
        }
        fluidRow(
            column(6,
                div(class = "card", style = "padding:20px;",
                    h5("Confusion Matrix — Test Set",
                       style = "margin:0 0 14px 0; font-size:13px;"),
                    plotOutput("lr_confmat", height = "300px")
                )
            ),
            column(6,
                div(class = "card", style = "padding:20px;",
                    h5("Predicted Probability Distribution",
                       style = "margin:0 0 14px 0; font-size:13px;"),
                    plotOutput("lr_prob_dist", height = "300px")
                )
            )
        )
    })

    output$lr_confmat <- renderPlot(bg = "#1c1c2e", {
        req(!is.null(lr_cm_obj))
        cm_tbl <- as.data.frame(lr_cm_obj$table) %>%
            mutate(
                Prediction = factor(Prediction, levels = rev(levels(Prediction))),
                Reference  = factor(Reference,  levels = levels(Reference))
            )
        acc  <- sprintf("%.1f%%", lr_cm_obj$overall["Accuracy"]  * 100)
        sens <- sprintf("%.1f%%", lr_cm_obj$byClass["Sensitivity"] * 100)
        spec <- sprintf("%.1f%%", lr_cm_obj$byClass["Specificity"] * 100)

        ggplot(cm_tbl, aes(x = Reference, y = Prediction, fill = Freq)) +
            geom_tile(color = "#13131e", linewidth = 1.5) +
            geom_text(aes(label = Freq), size = 9, fontface = "bold",
                      color = "white") +
            scale_fill_gradient(low = "#1e3a6a", high = ACCENT) +
            labs(
                subtitle = sprintf("Accuracy %s  ·  Sensitivity %s  ·  Specificity %s",
                                   acc, sens, spec),
                x = "Actual", y = "Predicted"
            ) +
            dark_theme() +
            theme(panel.grid     = element_blank(),
                  legend.position = "none",
                  axis.text       = element_text(size = 12, face = "bold"))
    })

    output$lr_prob_dist <- renderPlot(bg = "#1c1c2e", {
        req(!is.null(lr_test_df))
        ggplot(lr_test_df, aes(x = prob_win, fill = actual, color = actual)) +
            geom_density(alpha = 0.30, linewidth = 1.1) +
            geom_vline(xintercept = 0.5, linetype = "dashed",
                       color = "#2a2a3e", linewidth = 0.8) +
            annotate("text", x = 0.52, y = Inf, label = "threshold",
                     color = "#6a7590", size = 3.2, hjust = 0, vjust = 1.8) +
            scale_fill_manual(values  = c("Win" = ACCENT, "Loss" = "#2e3a5e"),
                              name = NULL) +
            scale_color_manual(values = c("Win" = ACCENT, "Loss" = "#6a7590"),
                               name = NULL) +
            scale_x_continuous(limits = c(0, 1),
                               labels = function(x) paste0(round(x * 100), "%")) +
            labs(
                subtitle = "Model confidence by actual outcome — good separation = well-calibrated",
                x = "Predicted Win Probability", y = "Density"
            ) +
            dark_theme() +
            theme(legend.position  = "top",
                  panel.grid.minor = element_blank())
    })

    # ── Model Comparison tab ────────────────────────────────────────────────────
    output$model_table <- renderDT({
        perf_summary %>%
            filter(Method != "LR_all") %>%
            mutate(Accuracy = round(Accuracy, 1), AUC_ROC = round(AUC_ROC, 1)) %>%
            rename(`# Features` = n_features,
                   `Accuracy (%)` = Accuracy, `AUC-ROC (%)` = AUC_ROC) %>%
            datatable(rownames = FALSE,
                      options  = list(dom = "t", ordering = TRUE)) %>%
            formatStyle("AUC-ROC (%)",
                background = styleColorBar(c(80, 86), "#5383e8"),
                backgroundSize = "100% 80%", backgroundRepeat = "no-repeat",
                backgroundPosition = "center")
    })

    output$roc_plot <- renderPlot(bg = "#1c1c2e", {
        roc_path <- "models/roc_list.rds"
        if (!file.exists(roc_path)) {
            par(bg = "#1c1c2e", col.main = "#ffffff", col.axis = "#9aaccc",
                col.lab = "#9aaccc", fg = "#2a2a3e")
            plot(1, type = "n", xlim = c(1,0), ylim = c(0,1),
                 xlab = "1 - Specificity", ylab = "Sensitivity",
                 main = "ROC Curves — re-run analysis.Rmd to generate")
            abline(a = 1, b = -1, lty = 2, col = "#2a2a3e")
            return(invisible(NULL))
        }
        rocs <- readRDS(roc_path)

        models <- list(
            list(roc = rocs$lr,   label = "Logistic Regression", lty = 1,  lwd = 2.2, col = "#ffffff"),
            list(roc = rocs$rf,   label = "Random Forest",        lty = 1,  lwd = 2.2, col = "#9aaccc"),
            list(roc = rocs$nb,   label = "Naive Bayes",          lty = 2,  lwd = 1.8, col = ACCENT),
            list(roc = rocs$knn,  label = "KNN",                  lty = 2,  lwd = 1.8, col = "#e8a838"),
            list(roc = rocs$cart, label = "CART",                 lty = 3,  lwd = 1.6, col = "#e84057")
        )

        par(mar = c(5, 5, 4, 2), family = "sans", bg = "#1c1c2e",
            col.main = "#ffffff", col.axis = "#9aaccc", col.lab = "#9aaccc",
            fg = "#2a2a3e")
        plot(0, type = "n", xlim = c(1, 0), ylim = c(0, 1),
             xlab = "1 - Specificity (False Positive Rate)",
             ylab = "Sensitivity (True Positive Rate)",
             main = "ROC Curves — All Models",
             cex.main = 1.3, cex.lab = 1.05,
             las = 1, bty = "l")
        abline(a = 1, b = -1, lty = 3, col = "#2a2a3e", lwd = 1.2)

        for (m in models) {
            lines(1 - m$roc$specificities, m$roc$sensitivities,
                  col = m$col, lty = m$lty, lwd = m$lwd)
        }

        legend("bottomright",
               legend = sapply(models, function(m)
                   sprintf("%s  (AUC = %.3f)", m$label, as.numeric(pROC::auc(m$roc)))),
               col    = sapply(models, `[[`, "col"),
               lty    = sapply(models, `[[`, "lty"),
               lwd    = sapply(models, `[[`, "lwd"),
               bty    = "n", cex = 0.92, pt.cex = 1,
               text.col = "#9aaccc", bg = "#1c1c2e")
    })

    # ── Feature Selection tab ───────────────────────────────────────────────────
    output$feat_perf_table <- DT::renderDT(server = FALSE, {
        perf_summary %>%
            arrange(desc(AUC_ROC)) %>%
            mutate(
                Accuracy = sprintf("%.1f%%", Accuracy),
                AUC_ROC  = sprintf("%.1f%%", AUC_ROC)
            ) %>%
            rename(`# Features` = n_features, `Accuracy` = Accuracy, `AUC-ROC` = AUC_ROC) %>%
            datatable(rownames = FALSE,
                      options  = list(dom = "t", ordering = FALSE, pageLength = 10)) %>%
            formatStyle("Method", fontWeight = "bold", color = "#5383e8") %>%
            formatStyle(columns = c("Method", "# Features", "Accuracy", "AUC-ROC"),
                        color = "#c8d0e0", backgroundColor = "#1c1c2e")
    })

    output$fs_consensus_5 <- renderUI({
        feats <- overlap_df %>% filter(n_methods == 5) %>% pull(feature)
        if (length(feats) == 0) return(p(style = "color:#9aaccc; font-size:13px;", "None"))
        tags$ul(style = "color:#c8d0e0; font-size:13px; margin:0; padding-left:16px; line-height:1.9;",
            lapply(feats, tags$li))
    })

    output$fs_consensus_4 <- renderUI({
        feats <- overlap_df %>% filter(n_methods == 4) %>% pull(feature)
        if (length(feats) == 0) return(p(style = "color:#9aaccc; font-size:13px;", "None"))
        tags$ul(style = "color:#c8d0e0; font-size:13px; margin:0; padding-left:16px; line-height:1.9;",
            lapply(feats, tags$li))
    })

    output$fs_consensus_0 <- renderUI({
        feats <- overlap_df %>% filter(n_methods == 0) %>% pull(feature)
        if (length(feats) == 0) return(p(style = "color:#9aaccc; font-size:13px;", "None"))
        tags$ul(style = "color:#c8d0e0; font-size:13px; margin:0; padding-left:16px; line-height:1.9;",
            lapply(feats, tags$li))
    })

    output$lollipop_plot <- renderPlot(bg = "#1c1c2e", {
        baseline_auc <- perf_summary$AUC_ROC[perf_summary$Method == "LR_all"]
        perf_summary %>%
            filter(Method != "LR_all") %>%
            mutate(Method = reorder(Method, AUC_ROC)) %>%
            ggplot(aes(x = AUC_ROC, y = Method)) +
            geom_segment(aes(x = baseline_auc - 1.5, xend = AUC_ROC, yend = Method),
                         color = "#2a2a3e", linewidth = 1) +
            geom_point(size = 5, color = "#27ae60") +
            geom_text(aes(label = sprintf("%d features", n_features)),
                      nudge_x = 0.15, hjust = 0, size = 3.5, color = "#9aaccc") +
            geom_vline(xintercept = baseline_auc, linetype = "dashed",
                       color = "#e84057", linewidth = 0.8) +
            annotate("text", x = baseline_auc + 0.05, y = 0.6,
                     label = sprintf("Baseline\n(%d feat.)", max(perf_summary$n_features)),
                     color = "#e84057", size = 3, hjust = 0) +
            scale_x_continuous(limits = c(baseline_auc - 1.5, baseline_auc + 1.5)) +
            labs(title    = "Feature Selection: AUC-ROC vs. Method",
                 subtitle = "Red dashed = LR on all features  ·  Labels = features retained",
                 x = "AUC-ROC (%)", y = NULL) +
            dark_theme()
    })

    output$heatmap_plot <- renderPlot(bg = "#1c1c2e", {
        overlap_df %>%
            filter(n_methods > 0) %>%
            pivot_longer(c(RFE, Forward, LASSO, ElasticNet, RF_Imp),
                         names_to = "method", values_to = "selected") %>%
            mutate(
                feature_label = paste0(feature, "  (", n_methods, "/5)"),
                feature_label = reorder(feature_label, n_methods),
                method = factor(method,
                    levels = c("RFE","Forward","LASSO","ElasticNet","RF_Imp"))
            ) %>%
            ggplot(aes(x = method, y = feature_label, fill = selected)) +
            geom_tile(color = "#13131e", linewidth = 0.5) +
            scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "#22223a"),
                              guide = "none") +
            labs(title    = "Feature Selection Overlap Across All Methods",
                 subtitle = "Green = selected  ·  (x/5) = how many methods selected this feature",
                 x = NULL, y = NULL) +
            dark_theme() +
            theme(panel.grid  = element_blank(),
                  axis.text.x = element_text(face = "bold", size = 10),
                  axis.text.y = element_text(size = 8))
    })

    # ── Data Explorer ──────────────────────────────────────────────────────────
    rv_data <- reactiveValues(df = NULL, source = "none")

    observeEvent(input$data_use_example, {
        req(!is.null(games_data))
        rv_data$df     <- as.data.frame(games_data)
        rv_data$source <- "example"
    })

    observeEvent(input$data_upload, {
        req(input$data_upload)
        tryCatch({
            df             <- read.csv(input$data_upload$datapath, stringsAsFactors = FALSE)
            rv_data$df     <- df
            rv_data$source <- input$data_upload$name
        }, error = function(e) {
            showNotification(paste("Read error:", e$message), type = "error", duration = 5)
        })
    })

    output$data_sidebar_stats <- renderUI({
        if (is.null(rv_data$df)) return(NULL)
        df    <- rv_data$df
        label <- if (rv_data$source == "example") "Built-in dataset (games)" else rv_data$source

        is_games_fmt <- "blue_win" %in% names(df)

        win_rate <- if (is_games_fmt)
            sprintf("%.1f%%", mean(df$blue_win == 1, na.rm = TRUE) * 100)
        else if ("result" %in% names(df))
            sprintf("%.1f%%", mean(df$result == 1, na.rm = TRUE) * 100)
        else "N/A"

        win_label <- if (is_games_fmt) "Blue Win Rate" else "Win Rate"

        stat_row <- function(lbl, val, col = "#c8d0e0") {
            div(class = "data-stat-row",
                span(class = "data-stat-label", lbl),
                span(class = "data-stat-value", style = paste0("color:", col), val)
            )
        }

        extra_stats <- if (is_games_fmt) {
            n_features <- sum(grepl("^blue_", names(df))) - 1  # minus blue_win
            list(stat_row("Features", as.character(n_features), "#9b59b6"))
        } else {
            n_leagues <- if ("league"   %in% names(df)) as.character(length(unique(df$league)))   else "N/A"
            n_teams   <- if ("teamname" %in% names(df)) as.character(length(unique(df$teamname))) else "N/A"
            list(
                stat_row("Leagues", n_leagues, "#9b59b6"),
                stat_row("Teams",   n_teams,   "#e84057")
            )
        }

        tagList(
            div(class = "sidebar-section",
                div(style = "display:flex; align-items:center; gap:8px; margin-bottom:8px;",
                    div(style = "width:8px; height:8px; border-radius:50%; background:#27ae60; flex-shrink:0;"),
                    span(style = "font-size:13px; font-weight:600; color:#c8d0e0;", label)
                ),
                div(style = "font-size:12px; color:#6a7590;",
                    sprintf("%s rows × %s cols", format(nrow(df), big.mark = ","), ncol(df))),
                if (is_games_fmt)
                    div(style = "font-size:11px; color:#5383e8; margin-top:4px;", "1 row per game · blue_* / red_* columns")
            ),
            div(class = "sidebar-section",
                p(class = "sidebar-section-title", "Dataset Statistics"),
                stat_row("Total Games", format(nrow(df), big.mark = ","), ACCENT),
                stat_row(win_label,     win_rate,                          "#27ae60"),
                tagList(extra_stats)
            )
        )
    })

    output$data_no_data_hint <- renderUI({
        if (!is.null(rv_data$df)) return(NULL)
        div(class = "data-no-data-hint",
            div(style = "font-size:15px; font-weight:600; color:#9aaccc; margin-bottom:6px;",
                "No dataset loaded"),
            div(style = "font-size:13px; color:#6a7590;",
                'Upload a CSV or click "Use Built-in Dataset" in the sidebar')
        )
    })

    output$data_dist_controls <- renderUI({
        req(!is.null(rv_data$df))
        num_cols <- names(rv_data$df)[sapply(rv_data$df, is.numeric)]
        if (length(num_cols) == 0) {
            return(div(style = "color:#6a7590; font-size:13px; padding:10px 0;",
                       "No numeric columns found."))
        }
        div(style = "max-width:300px; margin-bottom:20px;",
            selectInput("data_dist_col", "Variable",
                choices  = num_cols,
                selected = if      ("blue_golddiffat15" %in% num_cols) "blue_golddiffat15"
                           else if ("golddiffat15"      %in% num_cols) "golddiffat15"
                           else num_cols[1],
                width    = "100%")
        )
    })

    output$data_table_full <- renderDT({
        req(!is.null(rv_data$df))
        datatable(rv_data$df, rownames = FALSE, filter = "top",
            options = list(
                scrollX    = TRUE,
                dom        = "rtip",
                autoWidth  = TRUE,
                pageLength = 18
            )
        )
    })

    # Same column set as EDA correlation chunk (Blue's perspective only)
    eda_early_cols <- c(
        "blue_goldat10",     "blue_xpat10",     "blue_csat10",
        "blue_golddiffat10", "blue_xpdiffat10",  "blue_csdiffat10",
        "blue_killsat10",    "blue_deathsat10",  "blue_assistsat10",
        "blue_goldat15",     "blue_xpat15",     "blue_csat15",
        "blue_golddiffat15", "blue_xpdiffat15",  "blue_csdiffat15",
        "blue_killsat15",    "blue_deathsat15",  "blue_assistsat15",
        "blue_firstblood",   "blue_firstdragon", "blue_firstherald", "blue_firsttower",
        "blue_void_grubs",   "red_void_grubs",
        "blue_win"
    )

    corr_cols_for <- function(df) {
        if ("blue_win" %in% names(df))
            intersect(eda_early_cols, names(df))
        else
            names(df)[sapply(df, is.numeric)]
    }

    output$data_corr_plot_ui <- renderUI({
        req(!is.null(rv_data$df))
        cols <- corr_cols_for(rv_data$df)
        n    <- max(2, length(cols))
        px   <- max(420, n * 52)
        plotOutput("data_corr_plot", height = paste0(px, "px"))
    })

    output$data_corr_plot <- renderPlot(bg = "#1c1c2e", {
        req(!is.null(rv_data$df))
        df     <- rv_data$df
        is_eda <- "blue_win" %in% names(df)
        cols   <- corr_cols_for(df)
        num_df <- df[, cols, drop = FALSE]
        num_df <- num_df[, colSums(is.na(num_df)) < nrow(num_df), drop = FALSE]
        if (ncol(num_df) < 2) {
            return(ggplot() +
                annotate("text", x = 0.5, y = 0.5, label = "Need at least 2 numeric columns",
                         color = "#6a7590", size = 5) +
                dark_theme() + theme_void())
        }

        n      <- ncol(num_df)
        txt_sz <- if (n <= 6) 4.2 else if (n <= 10) 3.4 else 2.6

        cor_mat <- cor(num_df, use = "pairwise.complete.obs")
        cor_df  <- as.data.frame(as.table(cor_mat)) %>%
            rename(corr = Freq)

        ggplot(cor_df, aes(x = Var1, y = Var2, fill = corr)) +
            geom_tile(color = "#13131e", linewidth = 0.5) +
            geom_text(aes(label = ifelse(abs(corr) >= 0.05,
                                         sprintf("%.2f", corr), "")),
                      size = txt_sz, color = "white", fontface = "bold") +
            scale_fill_gradient2(
                low      = "#e84057",
                mid      = "#1c1c2e",
                high     = ACCENT,
                midpoint = 0,
                limits   = c(-1, 1),
                name     = "Pearson r"
            ) +
            coord_fixed() +
            labs(
                title    = "Feature Correlation Matrix",
                subtitle = if (is_eda)
                    "Early-game features · Blue's perspective · Red = negative, Blue = positive"
                else
                    "Red = negative correlation · Blue = positive correlation"
            ) +
            dark_theme() +
            theme(
                axis.text.x     = element_text(angle = 35, hjust = 1, size = 11),
                axis.text.y     = element_text(size = 11),
                axis.title      = element_blank(),
                panel.grid      = element_blank(),
                legend.position = "right"
            )
    })

    output$data_corrbar_plot_ui <- renderUI({
        req(!is.null(rv_data$df))
        plotOutput("data_corrbar_plot", height = "380px")
    })

    output$data_corrbar_plot <- renderPlot(bg = "#1c1c2e", {
        req(!is.null(rv_data$df))
        df     <- rv_data$df
        is_eda <- "blue_win" %in% names(df)

        target_col <- if (is_eda) "blue_win"
                      else if ("result" %in% names(df)) "result"
                      else return(NULL)

        candidate_cols <- if (is_eda) eda_early_cols
                          else names(df)[sapply(df, is.numeric)]
        feat_cols <- setdiff(intersect(candidate_cols, names(df)), target_col)
        feat_cols <- feat_cols[colSums(is.na(df[, feat_cols, drop = FALSE])) < nrow(df)]
        if (length(feat_cols) < 1) return(NULL)

        target_vec <- df[[target_col]]
        cors <- sapply(feat_cols, function(col) {
            cor(df[[col]], target_vec, use = "pairwise.complete.obs")
        })

        cor_df <- data.frame(feature = names(cors), correlation = unname(cors)) %>%
            filter(!is.na(correlation)) %>%
            arrange(desc(abs(correlation))) %>%
            slice_head(n = 10) %>%
            mutate(
                feature   = factor(feature, levels = rev(feature)),
                direction = ifelse(correlation > 0, "positive", "negative")
            )

        lim <- max(abs(cor_df$correlation)) * 1.3

        ggplot(cor_df, aes(x = feature, y = correlation, fill = direction)) +
            geom_col(width = 0.65) +
            geom_text(aes(
                label = sprintf("%.3f", correlation),
                hjust = ifelse(correlation >= 0, -0.1, 1.1)
            ), size = 3.5, color = "#c8d0e0") +
            coord_flip() +
            geom_hline(yintercept = 0, color = "#6a7590", linewidth = 0.5) +
            scale_fill_manual(values = c(positive = ACCENT, negative = "#e84057"), guide = "none") +
            scale_y_continuous(limits = c(-lim, lim)) +
            labs(
                title    = sprintf("Top 10 Features Correlated with %s", target_col),
                subtitle = "All features · sorted by |r|",
                x = NULL, y = "Pearson r"
            ) +
            dark_theme() +
            theme(
                axis.text.y        = element_text(size = 11),
                panel.grid.major.y = element_blank()
            )
    })

    output$data_winrate_plot <- renderPlot(bg = "#1c1c2e", {
        req(!is.null(rv_data$df))
        df <- rv_data$df

        is_games_fmt <- "blue_win"  %in% names(df)
        is_old_fmt   <- "result"    %in% names(df)

        if (!is_games_fmt && !is_old_fmt) {
            return(ggplot() +
                annotate("text", x = 0.5, y = 0.5,
                         label = "Dataset needs a 'blue_win' or 'result' column",
                         color = "#6a7590", size = 5) +
                dark_theme() + theme_void())
        }

        # ── Normalise to a common interface ──────────────────────────────────────
        if (is_games_fmt) {
            target_col  <- "blue_win"
            win_label   <- "Blue won"
            loss_label  <- "Red won"
            win_col     <- "#3498db"
            loss_col    <- "#e84057"
            obj_cols    <- intersect(
                c("blue_firstblood","blue_firstdragon","blue_firstherald","blue_firsttower"),
                names(df))
            obj_labels  <- c(blue_firstblood  = "First Blood",
                             blue_firstdragon = "First Dragon",
                             blue_firstherald = "First Herald",
                             blue_firsttower  = "First Tower")
            cont_cols   <- intersect(
                c("blue_golddiffat15","blue_xpdiffat15","blue_csdiffat15"),
                names(df))
            cont_labels <- c(blue_golddiffat15 = "Gold Diff @15",
                             blue_xpdiffat15   = "XP Diff @15",
                             blue_csdiffat15   = "CS Diff @15")
        } else {
            target_col  <- "result"
            win_label   <- "Win"
            loss_label  <- "Loss"
            win_col     <- ACCENT
            loss_col    <- "#e84057"
            obj_cols    <- intersect(
                c("firstblood","firstdragon","firstherald","firsttower"),
                names(df))
            obj_labels  <- c(firstblood  = "First Blood",  firstdragon = "First Dragon",
                             firstherald = "First Herald", firsttower  = "First Tower")
            cont_cols   <- intersect(
                c("golddiffat15","xpdiffat15","csdiffat15","grub_diff","winrate_diff"),
                names(df))
            cont_labels <- c(golddiffat15 = "Gold Diff @15", xpdiffat15   = "XP Diff @15",
                             csdiffat15   = "CS Diff @15",   grub_diff    = "Grub Diff",
                             winrate_diff = "WR Diff")
        }

        target_vals <- df[[target_col]]
        n_win       <- sum(target_vals == 1, na.rm = TRUE)
        n_loss      <- sum(target_vals == 0, na.rm = TRUE)
        n_total     <- n_win + n_loss

        # ── Panel A: target distribution ─────────────────────────────────────────
        dist_df <- data.frame(
            Outcome = factor(c(win_label, loss_label), levels = c(win_label, loss_label)),
            n       = c(n_win, n_loss)
        )
        p_dist <- ggplot(dist_df, aes(x = Outcome, y = n, fill = Outcome)) +
            geom_col(width = 0.5) +
            geom_text(aes(label = sprintf("%s\n%.1f%%",
                                          format(n, big.mark = ","), n / n_total * 100)),
                      vjust = -0.35, size = 4, color = "#c8d0e0", fontface = "bold") +
            geom_hline(yintercept = n_total / 2,
                       linetype = "dashed", color = "#6a7590", linewidth = 0.7) +
            scale_fill_manual(values = setNames(c(win_col, loss_col),
                                                c(win_label, loss_label))) +
            scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
            labs(title    = "Target Distribution",
                 subtitle = if (is_games_fmt)
                     sprintf("Blue WR = %.1f%%  ·  p ≪ 0.001 vs 50%%  ·  n = %s games",
                             n_win / n_total * 100, format(n_total, big.mark = ","))
                 else
                     sprintf("Win rate = %.1f%%  ·  n = %s rows",
                             n_win / n_total * 100, format(n_total, big.mark = ",")),
                 x = NULL, y = "Count") +
            dark_theme() +
            theme(legend.position    = "none",
                  panel.grid.major.x = element_blank())

        p_list <- list(dist = p_dist)

        # ── Panel B: objective win rates ─────────────────────────────────────────
        if (length(obj_cols) > 0) {
            wr_obj <- lapply(obj_cols, function(col) {
                df %>%
                    filter(!is.na(.data[[col]]), !is.na(.data[[target_col]])) %>%
                    group_by(got = factor(
                        ifelse(.data[[col]] == 1, "Got it", "Didn't get it"),
                        levels = c("Got it", "Didn't get it"))) %>%
                    summarise(
                        win_rate = mean(.data[[target_col]] == 1, na.rm = TRUE) * 100,
                        n        = n(), .groups = "drop") %>%
                    mutate(objective = obj_labels[col])
            }) %>% bind_rows()

            p_list$obj <- ggplot(wr_obj, aes(x = objective, y = win_rate, fill = got)) +
                geom_col(position = position_dodge(width = 0.6), width = 0.5) +
                geom_text(aes(label = sprintf("%.1f%%", win_rate)),
                          position = position_dodge(width = 0.6),
                          vjust = -0.5, size = 3.2, color = "#c8d0e0", fontface = "bold") +
                geom_hline(yintercept = 50,
                           linetype = "dashed", color = "#6a7590", linewidth = 0.7) +
                scale_fill_manual(
                    values = c("Got it" = ACCENT, "Didn't get it" = "#2e3a5e"),
                    name = NULL) +
                scale_y_continuous(limits = c(0, 80), expand = c(0, 0),
                                   labels = function(x) paste0(x, "%")) +
                labs(title    = if (is_games_fmt) "Blue Win Rate by Objective"
                               else "Win Rate by Objective",
                     subtitle = "Dashed line = 50% baseline",
                     x = NULL,
                     y = if (is_games_fmt) "Blue Win Rate (%)" else "Win Rate (%)") +
                dark_theme() +
                theme(legend.position    = "top",
                      panel.grid.major.x = element_blank())
        }

        # ── Panel C: key differentials by outcome ─────────────────────────────────
        if (length(cont_cols) > 0) {
            box_df <- df %>%
                select(all_of(c(target_col, cont_cols))) %>%
                filter(!is.na(.data[[target_col]])) %>%
                mutate(Outcome = factor(
                    ifelse(.data[[target_col]] == 1, win_label, loss_label),
                    levels = c(win_label, loss_label))) %>%
                pivot_longer(all_of(cont_cols),
                             names_to = "feature", values_to = "value") %>%
                mutate(feature = recode(feature, !!!cont_labels))

            p_list$box <- ggplot(box_df, aes(x = Outcome, y = value, fill = Outcome)) +
                geom_hline(yintercept = 0,
                           linetype = "dashed", color = "#6a7590", linewidth = 0.6) +
                geom_boxplot(alpha = 0.7, outlier.size = 0.8,
                             outlier.color = "#6a7590", width = 0.5) +
                scale_fill_manual(
                    values = setNames(c(win_col, loss_col), c(win_label, loss_label)),
                    guide  = "none") +
                facet_wrap(~feature, scales = "free_y", nrow = 1) +
                labs(title    = "Feature Distribution by Outcome",
                     subtitle = "Median line inside box  ·  dashed = zero",
                     x = NULL, y = NULL) +
                dark_theme() +
                theme(strip.text         = element_text(size = 9, color = "#9aaccc"),
                      panel.grid.major.x = element_blank(),
                      strip.background   = element_rect(fill = "#13131e", color = NA))
        }

        # ── Arrange panels ────────────────────────────────────────────────────────
        if (!is.null(p_list$obj) && !is.null(p_list$box)) {
            gridExtra::grid.arrange(
                p_list$dist,
                gridExtra::arrangeGrob(p_list$obj, p_list$box, ncol = 2),
                nrow = 2, heights = c(1, 1.4)
            )
        } else if (!is.null(p_list$obj)) {
            gridExtra::grid.arrange(p_list$dist, p_list$obj, nrow = 2, heights = c(1, 1.4))
        } else if (!is.null(p_list$box)) {
            gridExtra::grid.arrange(p_list$dist, p_list$box, nrow = 2, heights = c(1, 1.4))
        } else {
            print(p_list$dist)
        }
    })

    output$data_dist_plot <- renderPlot(bg = "#1c1c2e", {
        req(!is.null(rv_data$df), !is.null(input$data_dist_col))
        df  <- rv_data$df
        col <- input$data_dist_col
        req(col %in% names(df))
        vals  <- df[[col]]
        valid <- is.finite(vals)
        x     <- vals[valid]
        if (length(x) < 2) return(NULL)

        is_games_fmt <- "blue_win" %in% names(df)
        is_old_fmt   <- "result"   %in% names(df)
        is_diff_col  <- grepl("diff", col, ignore.case = TRUE)
        plot_df      <- data.frame(x = x)

        if (is_games_fmt) {
            plot_df$Outcome <- factor(
                ifelse(df$blue_win[valid] == 1, "Blue won", "Red won"),
                levels = c("Blue won", "Red won")
            )
            p <- ggplot(plot_df, aes(x = x, fill = Outcome, color = Outcome)) +
                geom_density(alpha = 0.22, linewidth = 1.1) +
                scale_fill_manual(values  = c("Blue won" = "#3498db", "Red won" = "#e84057"), name = NULL) +
                scale_color_manual(values = c("Blue won" = "#3498db", "Red won" = "#e84057"), name = NULL)
        } else if (is_old_fmt) {
            plot_df$Outcome <- factor(
                ifelse(df$result[valid] == 1, "Win", "Loss"),
                levels = c("Win", "Loss")
            )
            p <- ggplot(plot_df, aes(x = x, fill = Outcome, color = Outcome)) +
                geom_density(alpha = 0.22, linewidth = 1.1) +
                scale_fill_manual(values  = c("Win" = ACCENT, "Loss" = "#2e3a5e"), name = NULL) +
                scale_color_manual(values = c("Win" = ACCENT, "Loss" = "#6a7590"), name = NULL)
        } else {
            p <- ggplot(plot_df, aes(x = x)) +
                geom_density(fill = ACCENT, color = ACCENT, alpha = 0.22, linewidth = 1.1)
        }

        if (is_diff_col) {
            p <- p + geom_vline(xintercept = 0,
                                linetype = "dashed", color = "#6a7590", linewidth = 0.8)
        }

        p +
            labs(
                title    = col,
                subtitle = sprintf("n = %s  ·  mean = %.1f  ·  sd = %.1f  ·  range [%.1f, %.1f]",
                                   format(length(x), big.mark = ","),
                                   mean(x), sd(x), min(x), max(x)),
                x = col, y = "Density"
            ) +
            dark_theme(base_size = 14) +
            theme(
                panel.grid.minor = element_blank(),
                legend.position  = if (is_games_fmt || is_old_fmt) "top" else "none"
            )
    })

}

shinyApp(ui, server)
