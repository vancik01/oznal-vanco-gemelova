library(shiny)
library(bslib)
library(tidyverse)
library(caret)
library(DT)
library(pROC)

# ── Load models ────────────────────────────────────────────────────────────────
lr_model     <- readRDS("models/lr_model.rds")
rf_model     <- readRDS("models/rf_model.rds")
nb_model     <- readRDS("models/nb_model.rds")
knn_model    <- readRDS("models/knn_model.rds")
cart_model   <- readRDS("models/cart_model.rds")
preproc      <- readRDS("models/preproc.rds")
perf_summary <- readRDS("models/perf_summary.rds")
overlap_df   <- readRDS("models/overlap_df.rds")
teams_model  <- readRDS("models/teams_model.rds")
X_train      <- readRDS("models/X_train.rds")
game_lookup  <- readRDS("models/game_lookup.rds")

feature_names <- names(X_train)
train_means   <- colMeans(X_train)

# ── Model metadata ─────────────────────────────────────────────────────────────
model_meta <- list(
    LR = list(
        name  = "Logistic Regression",
        color = "#6366f1",
        param = "None",
        desc  = "Standard GLM — no hyperparameter search. All features used, trained on standardized inputs. Coefficients are directly interpretable as log-odds weights."
    ),
    RF = list(
        name  = "Random Forest",
        color = "#0ea5e9",
        param = "mtry",
        desc  = "Ensemble of 500 decision trees. mtry controls how many features are randomly considered at each split. Tested: 5, 7, 10, 15, 20."
    ),
    NB = list(
        name  = "Naive Bayes",
        color = "#8b5cf6",
        param = "usekernel + adjust",
        desc  = "Probabilistic classifier. Tunes density estimation (Gaussian vs. kernel) and bandwidth multiplier (adjust). Tested: 2 × 3 = 6 combinations."
    ),
    KNN = list(
        name  = "K-Nearest Neighbors",
        color = "#f59e0b",
        param = "k",
        desc  = "Classifies by majority vote among the k nearest neighbors in scaled feature space. Small k = noisy boundary; large k = smooth but biased."
    ),
    CART = list(
        name  = "Decision Tree (CART)",
        color = "#ef4444",
        param = "cp (complexity parameter)",
        desc  = "Recursive binary splitting tree. cp penalizes tree growth — small cp allows deep trees, large cp forces early stopping. Tested on log scale."
    )
)

# ── CSS ───────────────────────────────────────────────────────────────────────
nuxt_css <- "
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

body { background: #f8fafc; color: #1e293b; }

/* Navbar */
.navbar {
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    padding: 0 24px;
}
.navbar-brand {
    font-weight: 700;
    font-size: 16px;
    color: #1e293b !important;
    letter-spacing: -0.3px;
}
.navbar-nav > li > a {
    color: #64748b !important;
    font-size: 14px;
    font-weight: 500;
    padding: 18px 16px !important;
}
.navbar-nav > li.active > a,
.navbar-nav > li > a:hover {
    color: #1e293b !important;
    border-bottom: 2px solid #6366f1;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    padding: 20px;
    margin-bottom: 16px;
}

/* Replace wellPanel */
.well {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    padding: 20px !important;
}

/* Probability boxes */
.prob-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 12px;
    transition: box-shadow 0.2s;
}
.prob-box:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.prob-label {
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
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
    scrollbar-color: #e2e8f0 transparent;
}
.predictor-sidebar::-webkit-scrollbar { width: 4px; }
.predictor-sidebar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

.sidebar-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.sidebar-section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #94a3b8;
    margin: 0 0 10px 0;
}

.predictor-sidebar .shiny-input-container > label {
    font-size: 12px;
    color: #475569;
    font-weight: 500;
    margin-bottom: 1px;
}
.predictor-sidebar .form-group { margin-bottom: 10px; }
.predictor-sidebar .form-group:last-child { margin-bottom: 0; }
.predictor-sidebar .checkbox label,
.predictor-sidebar .radio label { font-size: 13px; color: #475569; }

.autofill-hint {
    font-size: 11px;
    color: #94a3b8;
    margin: 6px 0 8px 0;
    padding: 6px 8px;
    background: #f8fafc;
    border-radius: 6px;
    border: 1px dashed #e2e8f0;
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
    border-top: 1px solid #e2e8f0;
}
.section-divider span {
    padding: 0 10px;
    font-size: 10px;
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    white-space: nowrap;
}

.wr-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #94a3b8;
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
    border: 1px solid #e2e8f0;
    background: #ffffff;
    color: #64748b;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    margin: 0;
    transition: all 0.15s;
    user-select: none;
}
.pill-radio .radio-inline label:hover {
    border-color: #6366f1;
    color: #6366f1;
}
.pill-radio .radio-inline label.pill-active {
    background: #6366f1;
    border-color: #6366f1;
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
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.tune-info-card-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #94a3b8;
    margin: 0 0 8px 0;
}
.tune-info-card p {
    font-size: 13px;
    color: #475569;
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
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}
.tune-metric-box-label {
    font-size: 10px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 2px;
}
.tune-metric-box-value {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
}
.best-badge {
    display: inline-block;
    background: #eef2ff;
    color: #6366f1;
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
    color: #94a3b8;
    margin-bottom: 4px;
}

/* Inputs */
.form-control {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    color: #1e293b !important;
    background: #f8fafc !important;
}
.form-control:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    background: #ffffff !important;
}
.selectize-input {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

/* Slider */
.irs--shiny .irs-bar { background: #6366f1; }
.irs--shiny .irs-handle { background: #6366f1; border-color: #6366f1; }
.irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single {
    background: #6366f1;
}

/* Button */
.btn-primary {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    box-shadow: 0 1px 3px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
}
.btn-primary:hover {
    background: #4f46e5 !important;
    border-color: #4f46e5 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px);
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
    color: #64748b;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    margin: 0;
    width: 100%;
}
.model-nav .radio label:hover {
    background: #f8fafc;
    color: #475569;
    border-left-color: #cbd5e1;
}
.model-nav .radio label.model-nav-active {
    background: #eef2ff;
    color: #6366f1;
    border-left-color: #6366f1;
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
    color: #6366f1 !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
}
.btn-reset-best:hover {
    background: #e0e7ff !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Game banner */
.game-banner {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.banner-teams { font-size: 18px; font-weight: 700; color: #1e293b; }
.banner-meta  { font-size: 13px; color: #94a3b8; margin-top: 2px; }
.badge-win  { background: #dcfce7; color: #16a34a; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
.badge-loss { background: #fee2e2; color: #dc2626; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
.badge-correct { background: #dbeafe; color: #2563eb; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; margin-left: 8px; }
.badge-wrong   { background: #fef3c7; color: #d97706; border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; margin-left: 8px; }

/* Tabs */
.nav-tabs { border-bottom: 1px solid #e2e8f0; }
.nav-tabs > li > a {
    font-size: 13px; font-weight: 500; color: #64748b;
    border: none !important; border-radius: 0 !important; padding: 10px 16px;
}
.nav-tabs > li.active > a {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
    background: transparent !important;
}

/* Page padding */
.tab-content { padding-top: 20px; }
.container-fluid { padding: 0 24px; }

h4 { font-weight: 700; font-size: 18px; color: #1e293b; letter-spacing: -0.3px; }
h5 { font-weight: 600; font-size: 14px; color: #1e293b; }
h6 { font-weight: 700; font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.6px; }

/* DT table */
.dataTables_wrapper { font-size: 13px; }
table.dataTable thead th {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
    color: #94a3b8; border-bottom: 1px solid #e2e8f0 !important;
}
table.dataTable tbody tr:hover { background: #f8fafc !important; }
table.dataTable tbody tr.selected td { background: #eef2ff !important; }
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
});
"

# ── UI ─────────────────────────────────────────────────────────────────────────
ui <- navbarPage(
    title = "LoL Early-Game Predictor",
    theme = bslib::bs_theme(bootswatch = "flatly"),
    header = tags$head(
        tags$style(HTML(nuxt_css)),
        tags$script(HTML(pill_js))
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
                            DTOutput("game_table")
                        )
                    ),
                    div(class = "section-divider",
                        tags$hr(), span("Features"), tags$hr()
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
                            min = -10000, max = 10000, value = 0, step = 100, width = "100%"),
                        sliderInput("xpdiffat15", "XP Diff",
                            min = -6000, max = 6000, value = 0, step = 100, width = "100%"),
                        sliderInput("csdiffat15", "CS Diff",
                            min = -80, max = 80, value = 0, step = 1, width = "100%")
                    ),
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "Objectives"),
                        fluidRow(
                            column(6, checkboxInput("firstblood",  "First Blood",  FALSE)),
                            column(6, checkboxInput("firstdragon", "First Dragon", FALSE))
                        ),
                        fluidRow(
                            column(6, checkboxInput("firstherald", "First Herald", FALSE)),
                            column(6, checkboxInput("firsttower",  "First Tower",  FALSE))
                        ),
                        sliderInput("grub_diff", "Void Grub Advantage",
                            min = -6, max = 6, value = 0, step = 1, width = "100%")
                    ),
                    div(class = "sidebar-section",
                        p(class = "sidebar-section-title", "Pre-game Strength"),
                        sliderInput("wr_diff", "Win Rate Advantage (%)",
                            min = -50, max = 50, value = 0, step = 1, width = "100%"),
                        div(class = "wr-labels",
                            span("Opponent stronger"), span("Team stronger"))
                    ),
                    actionButton("predict_btn", "Predict Win Probability",
                        class = "btn-primary btn-lg",
                        style = "width:100%; margin-top:4px;")
                )
            ),
            column(8,
                br(),
                uiOutput("game_banner"),
                fluidRow(
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Logistic Regression"),
                        div(class = "prob-value", style = "color:#6366f1", textOutput("prob_lr"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Random Forest"),
                        div(class = "prob-value", style = "color:#0ea5e9", textOutput("prob_rf"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Naive Bayes"),
                        div(class = "prob-value", style = "color:#8b5cf6", textOutput("prob_nb"))
                    ))
                ),
                fluidRow(
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "KNN"),
                        div(class = "prob-value", style = "color:#f59e0b", textOutput("prob_knn"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "CART"),
                        div(class = "prob-value", style = "color:#ef4444", textOutput("prob_cart"))
                    )),
                    column(4, div(class = "prob-box",
                        div(class = "prob-label", "Average"),
                        div(class = "prob-value", style = "color:#1e293b", textOutput("prob_avg"))
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
                                    tagList(tags$span(class="model-dot", style="background:#6366f1;"), "Logistic Regression"),
                                    tagList(tags$span(class="model-dot", style="background:#0ea5e9;"), "Random Forest"),
                                    tagList(tags$span(class="model-dot", style="background:#8b5cf6;"), "Naive Bayes"),
                                    tagList(tags$span(class="model-dot", style="background:#f59e0b;"), "KNN"),
                                    tagList(tags$span(class="model-dot", style="background:#ef4444;"), "CART")
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
                div(class = "card", style = "padding:20px;",
                    h5("All Tested Configurations",
                       style = "margin:0 0 14px 0; font-size:13px;"),
                    DTOutput("tune_table")
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
        fluidRow(column(12,
            br(),
            h4("Performance vs. Number of Features"),
            plotOutput("lollipop_plot", height = "340px"),
            br(),
            h4("Feature Selection Overlap Across Methods"),
            plotOutput("heatmap_plot", height = "500px")
        ))
    ),

    # ── Tab 5: Data Explorer ────────────────────────────────────────────────────
    tabPanel("Data Explorer",
        sidebarLayout(
            sidebarPanel(
                h4("Filters"),
                selectInput("filter_side", "Side",
                    choices = c("All", "Blue", "Red"), selected = "All"),
                selectInput("filter_result", "Result",
                    choices = c("All", "Win", "Loss"), selected = "All"),
                sliderInput("filter_gold", "Gold Diff @15 range",
                    min = -15000, max = 15000,
                    value = c(-15000, 15000), step = 500)
            ),
            mainPanel(
                br(),
                h5(textOutput("explorer_count")),
                DTOutput("data_table")
            )
        )
    )
)

# ── Server ─────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

    rv <- reactiveValues(input_row = NULL, game_info = NULL)

    # ── Match Predictor: Browse ─────────────────────────────────────────────────
    observe({
        updateSelectInput(session, "browse_league",
            choices = sort(unique(game_lookup$league)))
    })

    observeEvent(input$browse_league, {
        teams_in_league <- game_lookup %>%
            filter(league == input$browse_league) %>%
            pull(teamname) %>% unique() %>% sort()
        updateSelectInput(session, "browse_team",
            choices = c("— select team —" = "", teams_in_league))
        rv$input_row <- NULL
        rv$game_info <- NULL
    })

    observeEvent(input$browse_team, {
        rv$input_row <- NULL
        rv$game_info <- NULL
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
                options = list(dom = "tp", pageLength = 6, scrollY = "260px")) %>%
            formatStyle("Result",
                color = styleEqual(c("WIN","LOSS"), c("#27ae60","#e74c3c")),
                fontWeight = "bold") %>%
            formatStyle("Gold @15",
                color = styleInterval(c(-1,0), c("#e74c3c","black","#27ae60")))
    })

    observeEvent(input$game_table_rows_selected, {
        idx <- input$game_table_rows_selected
        req(length(idx) > 0)
        selected     <- browse_games_full()[idx, ]
        rv$game_info <- selected
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
    })

    observeEvent(input$predict_btn, {
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
        rv$game_info     <- NULL
    })

    # ── Prediction ──────────────────────────────────────────────────────────────
    probs <- reactive({
        req(rv$input_row)
        raw    <- rv$input_row
        scaled <- predict(preproc, raw)
        list(
            lr   = round(predict(lr_model,   scaled, type = "prob")[,"Win"] * 100, 1),
            rf   = round(predict(rf_model,   raw,    type = "prob")[,"Win"] * 100, 1),
            nb   = round(predict(nb_model,   scaled, type = "prob")[,"Win"] * 100, 1),
            knn  = round(predict(knn_model,  scaled, type = "prob")[,"Win"] * 100, 1),
            cart = round(predict(cart_model, raw,    type = "prob")[,"Win"] * 100, 1)
        )
    })

    fmt <- function(p) if (is.null(rv$input_row)) "—" else paste0(p, "%")
    output$prob_lr   <- renderText({ fmt(probs()$lr)   })
    output$prob_rf   <- renderText({ fmt(probs()$rf)   })
    output$prob_nb   <- renderText({ fmt(probs()$nb)   })
    output$prob_knn  <- renderText({ fmt(probs()$knn)  })
    output$prob_cart <- renderText({ fmt(probs()$cart) })
    output$prob_avg  <- renderText({ fmt(round(mean(unlist(probs())), 1)) })

    output$game_banner <- renderUI({
        if (is.null(rv$game_info)) return(NULL)
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

    output$prob_bar <- renderPlot({
        req(rv$input_row)
        p <- probs()
        tibble(
            Model = c("LR","RF","NB","KNN","CART"),
            Prob  = c(p$lr, p$rf, p$nb, p$knn, p$cart),
            col   = c("#6366f1","#0ea5e9","#8b5cf6","#f59e0b","#ef4444")
        ) %>%
            mutate(Model = reorder(Model, Prob)) %>%
            ggplot(aes(x = Prob, y = Model, fill = col)) +
            geom_col(width = 0.55) +
            geom_text(aes(label = paste0(Prob, "%")), hjust = -0.2, size = 4) +
            geom_vline(xintercept = 50, linetype = "dashed", color = "grey60") +
            scale_fill_identity() +
            scale_x_continuous(limits = c(0, 115), expand = c(0,0)) +
            labs(x = "Win Probability (%)", y = NULL) +
            theme_minimal() +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.y = element_blank())
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
              style = "font-size:14px; font-weight:600; color:#1e293b; margin:0 0 4px 0;"),
            p(meta$desc,
              style = "font-size:12px; color:#64748b; margin:0; line-height:1.5;")
        )
    })

    # Left panel: overall CV best badge
    output$tune_best <- renderUI({
        m   <- input$tune_model
        col <- model_meta[[m]]$color

        best_line <- function(label, value) {
            div(style = "display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #f1f5f9;",
                span(style = "font-size:12px; color:#94a3b8; font-weight:500;", label),
                span(style = paste0("font-size:13px; font-weight:700; color:", col), value)
            )
        }

        if (m == "LR") return(
            div(style = "font-size:12px; color:#64748b;",
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
            return(div(style = "color:#94a3b8; font-size:13px; font-style:italic; padding:4px 0;",
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
                    tags$label(style = "font-size:12px; color:#475569; font-weight:500;",
                               "mtry — features per split"),
                    span(style = paste0("font-size:13px; font-weight:700; color:", col),
                         textOutput("rf_mtry_label", inline = TRUE))
                ),
                sliderInput("rf_mtry", NULL,
                    min = min(tested), max = max(tested),
                    value = rf_model$bestTune$mtry, step = 1, width = "100%"),
                div(style = "font-size:11px; color:#94a3b8; margin-top:-4px;",
                    paste("Tested:", paste(tested, collapse = ", ")))
            )
        } else if (m == "KNN") {
            tested <- knn_model$results$k
            tagList(
                tags$style(slider_css), reset_btn,
                div(style = "display:flex; justify-content:space-between; align-items:baseline;",
                    tags$label(style = "font-size:12px; color:#475569; font-weight:500;",
                               "k — number of neighbors"),
                    span(style = paste0("font-size:13px; font-weight:700; color:", col),
                         textOutput("knn_k_label", inline = TRUE))
                ),
                sliderInput("knn_k", NULL,
                    min = min(tested), max = max(tested),
                    value = knn_model$bestTune$k, step = 2, width = "100%"),
                div(style = "font-size:11px; color:#94a3b8; margin-top:-4px;",
                    paste("Tested:", paste(tested, collapse = ", ")))
            )
        } else if (m == "CART") {
            tested  <- cart_model$results$cp
            choices <- setNames(as.character(tested), formatC(tested, format = "g", digits = 2))
            tagList(
                reset_btn,
                tags$label(style = "font-size:12px; color:#475569; font-weight:500;",
                           "cp — complexity parameter"),
                selectInput("cart_cp", NULL,
                    choices  = choices,
                    selected = as.character(cart_model$bestTune$cp),
                    width    = "100%"),
                div(style = "font-size:11px; color:#94a3b8; margin-top:2px;",
                    "Smaller = deeper tree · Larger = more pruning")
            )
        } else if (m == "NB") {
            tagList(
                tags$style(slider_css), reset_btn,
                tags$label(style = "font-size:12px; color:#475569; font-weight:500; margin-bottom:4px; display:block;",
                           "Density estimation"),
                radioButtons("nb_kernel", NULL,
                    choices  = c("Gaussian (parametric)" = "FALSE",
                                 "Kernel density (non-parametric)" = "TRUE"),
                    selected = as.character(nb_model$bestTune$usekernel),
                    inline   = FALSE),
                div(style = "display:flex; justify-content:space-between; align-items:baseline; margin-top:10px;",
                    tags$label(style = "font-size:12px; color:#475569; font-weight:500;",
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

        is_best <- switch(m,
            LR   = TRUE,
            RF   = !is.null(input$rf_mtry) && snap_to(input$rf_mtry, rf_model$results$mtry) == rf_model$bestTune$mtry,
            KNN  = !is.null(input$knn_k)   && snap_to(input$knn_k, knn_model$results$k)     == knn_model$bestTune$k,
            CART = !is.null(input$cart_cp) && abs(as.numeric(input$cart_cp) - cart_model$bestTune$cp) < 1e-10,
            NB   = !is.null(input$nb_kernel) && !is.null(input$nb_adjust) &&
                   as.logical(input$nb_kernel) == nb_model$bestTune$usekernel &&
                   abs(as.numeric(input$nb_adjust) - nb_model$bestTune$adjust) < 0.01
        )

        fluidRow(style = "margin-bottom:16px;",
            column(4, style = "padding-right:6px;",
                div(style = paste0("background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid ", col,
                                   "; border-radius:10px; padding:14px 18px;"),
                    div(style = "font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; color:#94a3b8; margin-bottom:4px;", "AUC-ROC"),
                    div(style = paste0("font-size:28px; font-weight:700; color:", col, "; letter-spacing:-0.5px;"),
                        sprintf("%.2f%%", sr$ROC[1] * 100))
                )
            ),
            column(4, style = "padding:0 3px;",
                div(style = "background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid #e2e8f0; border-radius:10px; padding:14px 18px;",
                    div(style = "font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; color:#94a3b8; margin-bottom:4px;", "Sensitivity"),
                    div(style = "font-size:28px; font-weight:700; color:#1e293b; letter-spacing:-0.5px;",
                        sprintf("%.1f%%", sr$Sens[1] * 100))
                )
            ),
            column(4, style = "padding-left:6px;",
                div(style = "background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid #e2e8f0; border-radius:10px; padding:14px 18px;",
                    div(style = "font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; color:#94a3b8; margin-bottom:4px;", "Specificity"),
                    div(style = "font-size:28px; font-weight:700; color:#1e293b; letter-spacing:-0.5px;",
                        sprintf("%.1f%%", sr$Spec[1] * 100))
                )
            )
        )
    })

    # ── Tuning curve with moveable selected point ────────────────────────────────
    output$tune_plot <- renderPlot({
        m   <- input$tune_model
        col <- model_meta[[m]]$color
        sr  <- selected_row()

        if (m == "LR") {
            lr_coefs <- coef(lr_model$finalModel) %>%
                as.data.frame() %>% rownames_to_column("feature") %>%
                rename(coefficient = 2) %>%
                filter(feature != "(Intercept)") %>%
                arrange(desc(abs(coefficient))) %>% head(15)
            return(
                ggplot(lr_coefs, aes(x = reorder(feature, abs(coefficient)),
                                     y = coefficient, fill = coefficient > 0)) +
                geom_col(width = 0.65) + coord_flip() +
                scale_fill_manual(values = c("TRUE" = col, "FALSE" = "#f59e0b"),
                    labels = c("TRUE" = "Positive", "FALSE" = "Negative"), name = NULL) +
                labs(title = "Logistic Regression — Top 15 Coefficients",
                     subtitle = "Standard GLM · no hyperparameter search · standardized features",
                     x = NULL, y = "Coefficient") +
                theme_minimal(base_size = 13) +
                theme(legend.position = "top", panel.grid.major.y = element_blank())
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
                           size = 10, shape = 21, fill = "white", color = "#f59e0b", stroke = 2) +
                scale_color_manual(values = c("Gaussian" = col, "Kernel density" = "#f59e0b"), name = NULL) +
                labs(title = "Naive Bayes — CV AUC-ROC",
                     subtitle = "Filled = selected · Gold ring = CV best",
                     x = "Bandwidth adjust", y = "CV AUC-ROC (%)") +
                theme_minimal(base_size = 13) +
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

        # If best != selected, add gold ring for best
        if (nrow(best_row) > 0 && abs(best_row$param[1] - sel_param) > 1e-10) {
            p <- p +
                geom_point(data = best_row, aes(x = param, y = ROC * 100),
                           size = 11, shape = 21, fill = "white",
                           color = "#f59e0b", stroke = 2.5, inherit.aes = FALSE) +
                annotate("text", x = best_row$param[1],
                         y = best_row$ROC[1] * 100 - diff(range(d$ROC * 100)) * 0.18,
                         label = "★ best", color = "#f59e0b", size = 3.2, fontface = "bold")
        }

        p <- p +
            labs(title    = titles[m],
                 subtitle = "Filled circle = selected · Gold ring = CV best (if different)",
                 x = x_labels[m], y = "CV AUC-ROC (%)") +
            theme_minimal(base_size = 13)

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
                formatStyle("Selected", color = col,     fontWeight = "bold", fontSize = "14px", textAlign = "center") %>%
                formatStyle("Best",     color = "#f59e0b", fontWeight = "bold", fontSize = "16px", textAlign = "center")
        }
        dt
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
                background = styleColorBar(c(80, 86), "#2ecc71"),
                backgroundSize = "100% 80%", backgroundRepeat = "no-repeat",
                backgroundPosition = "center")
    })

    output$roc_plot <- renderPlot({
        plot(1, type = "n", xlim = c(1, 0), ylim = c(0, 1),
             xlab = "1 - Specificity", ylab = "Sensitivity", main = "ROC Curves")
        abline(a = 1, b = -1, lty = 2, col = "grey70")
        text(0.5, 0.5,
             "Add saveRDS(roc_lr, 'shiny/models/roc_lr.rds') etc. to analysis.Rmd",
             col = "grey50", cex = 1)
    })

    # ── Feature Selection tab ───────────────────────────────────────────────────
    output$lollipop_plot <- renderPlot({
        baseline_auc <- perf_summary$AUC_ROC[perf_summary$Method == "LR_all"]
        n_all        <- max(perf_summary$n_features)
        perf_summary %>%
            filter(Method != "LR_all") %>%
            mutate(Method = reorder(Method, AUC_ROC)) %>%
            ggplot(aes(x = AUC_ROC, y = Method)) +
            geom_segment(aes(x = baseline_auc - 1.5, xend = AUC_ROC, yend = Method),
                         color = "grey75", linewidth = 1) +
            geom_point(size = 5, color = "#2ecc71") +
            geom_text(aes(label = sprintf("%d features", n_features)),
                      nudge_x = 0.15, hjust = 0, size = 3.5, color = "grey30") +
            geom_vline(xintercept = baseline_auc, linetype = "dashed",
                       color = "#e74c3c", linewidth = 0.8) +
            annotate("text", x = baseline_auc + 0.05, y = 0.6,
                     label = sprintf("Baseline\n(%d feat.)", n_all),
                     color = "#e74c3c", size = 3, hjust = 0) +
            scale_x_continuous(limits = c(baseline_auc - 1.5, baseline_auc + 1.5)) +
            labs(title    = "Feature Selection: AUC-ROC vs. Method",
                 subtitle = "Red dashed = LR on all features  ·  Labels = features retained",
                 x = "AUC-ROC (%)", y = NULL) +
            theme_minimal()
    })

    output$heatmap_plot <- renderPlot({
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
            geom_tile(color = "grey85", linewidth = 0.5) +
            scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "white"),
                              guide = "none") +
            labs(title    = "Feature Selection Overlap Across All Methods",
                 subtitle = "Green = selected  ·  (x/5) = how many methods selected this feature",
                 x = NULL, y = NULL) +
            theme_minimal() +
            theme(panel.grid  = element_blank(),
                  axis.text.x = element_text(face = "bold", size = 10),
                  axis.text.y = element_text(size = 8))
    })

    # ── Data Explorer tab ───────────────────────────────────────────────────────
    filtered_data <- reactive({
        df <- teams_model %>%
            select(result, side, golddiffat15, xpdiffat15, csdiffat15,
                   firsttower, firstherald, firstdragon, firstblood, winrate_diff) %>%
            mutate(Result = ifelse(result == 1, "Win", "Loss"),
                   Side   = ifelse(side == 1, "Blue", "Red")) %>%
            select(-result, -side) %>%
            filter(golddiffat15 >= input$filter_gold[1],
                   golddiffat15 <= input$filter_gold[2])
        if (input$filter_side != "All")   df <- df %>% filter(Side   == input$filter_side)
        if (input$filter_result != "All") df <- df %>% filter(Result == input$filter_result)
        df
    })

    output$explorer_count <- renderText({
        sprintf("%d games shown", nrow(filtered_data()))
    })

    output$data_table <- renderDT({
        datatable(filtered_data(), rownames = FALSE,
                  options = list(pageLength = 15, scrollX = TRUE))
    })
}

shinyApp(ui, server)
