library(shiny)
library(tidyverse)
library(caret)
library(pROC)
library(DT)

# Load saved models and data
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

feature_names <- names(X_train)
train_means   <- colMeans(X_train)

# ── UI ────────────────────────────────────────────────────────────────────────

ui <- navbarPage(
    title = "LoL Early-Game Predictor",
    theme = bslib::bs_theme(bootswatch = "flatly"),

    # Tab 1: Prediction
    tabPanel("Prediction",
        sidebarLayout(
            sidebarPanel(
                h4("Early-game stats @ 15 min"),
                hr(),
                h5("Resources"),
                sliderInput("golddiffat15", "Gold Diff @15",
                    min = -10000, max = 10000, value = 0, step = 100),
                sliderInput("xpdiffat15", "XP Diff @15",
                    min = -8000, max = 8000, value = 0, step = 100),
                sliderInput("csdiffat15", "CS Diff @15",
                    min = -60, max = 60, value = 0, step = 1),
                hr(),
                h5("Objectives"),
                checkboxInput("firsttower",   "First Tower",   value = FALSE),
                checkboxInput("firstherald",  "First Herald",  value = FALSE),
                checkboxInput("firstdragon",  "First Dragon",  value = FALSE),
                checkboxInput("firstblood",   "First Blood",   value = FALSE),
                hr(),
                h5("Other"),
                selectInput("side", "Side", choices = c("Blue" = 1, "Red" = 0)),
                sliderInput("winrate_diff", "Team Winrate Diff (pre-game)",
                    min = -0.5, max = 0.5, value = 0, step = 0.01),
                hr(),
                actionButton("predict_btn", "Predict", class = "btn-primary btn-lg w-100")
            ),
            mainPanel(
                h3("Win Probability"),
                br(),
                fluidRow(
                    column(4, wellPanel(
                        h4("Logistic Regression", style = "text-align:center"),
                        h2(textOutput("prob_lr"), style = "text-align:center; color:#2ecc71")
                    )),
                    column(4, wellPanel(
                        h4("Random Forest", style = "text-align:center"),
                        h2(textOutput("prob_rf"), style = "text-align:center; color:#3498db")
                    )),
                    column(4, wellPanel(
                        h4("Naive Bayes", style = "text-align:center"),
                        h2(textOutput("prob_nb"), style = "text-align:center; color:#9b59b6")
                    ))
                ),
                fluidRow(
                    column(4, wellPanel(
                        h4("KNN", style = "text-align:center"),
                        h2(textOutput("prob_knn"), style = "text-align:center; color:#e67e22")
                    )),
                    column(4, wellPanel(
                        h4("CART", style = "text-align:center"),
                        h2(textOutput("prob_cart"), style = "text-align:center; color:#e74c3c")
                    )),
                    column(4, wellPanel(
                        h4("Average", style = "text-align:center"),
                        h2(textOutput("prob_avg"), style = "text-align:center; font-weight:bold")
                    ))
                ),
                br(),
                plotOutput("prob_bar")
            )
        )
    ),

    # Tab 2: Model Comparison
    tabPanel("Model Comparison",
        fluidRow(
            column(12,
                h3("Model Performance Overview"),
                DTOutput("model_table"),
                br(),
                h4("Select models to show on ROC curve:"),
                checkboxGroupInput("roc_models", NULL,
                    choices  = c("LR", "RF", "NB", "KNN", "CART"),
                    selected = c("LR", "RF", "NB", "KNN", "CART"),
                    inline   = TRUE
                ),
                plotOutput("roc_plot", height = "500px")
            )
        )
    ),

    # Tab 3: Feature Selection
    tabPanel("Feature Selection",
        fluidRow(
            column(12,
                h3("Feature Selection Comparison"),
                plotOutput("lollipop_plot", height = "350px"),
                br(),
                h3("Feature Selection Overlap"),
                plotOutput("heatmap_plot", height = "500px")
            )
        )
    ),

    # Tab 4: Data Explorer
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
                h4(textOutput("explorer_count")),
                DTOutput("data_table")
            )
        )
    )
)

# ── Server ────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

    input_row <- reactive({
        row <- as.data.frame(t(train_means))
        row$golddiffat15 <- input$golddiffat15
        row$xpdiffat15   <- input$xpdiffat15
        row$csdiffat15   <- input$csdiffat15
        row$firsttower   <- as.integer(input$firsttower)
        row$firstherald  <- as.integer(input$firstherald)
        row$firstdragon  <- as.integer(input$firstdragon)
        row$firstblood   <- as.integer(input$firstblood)
        row$side         <- as.numeric(input$side)
        row$winrate_diff <- input$winrate_diff
        row
    }) |> bindEvent(input$predict_btn, ignoreNULL = FALSE)

    scaled_row <- reactive({
        predict(preproc, input_row())
    })

    probs <- reactive({
        raw    <- input_row()
        scaled <- scaled_row()
        list(
            lr   = round(predict(lr_model,   scaled, type = "prob")[, "Win"] * 100, 1),
            rf   = round(predict(rf_model,   raw,    type = "prob")[, "Win"] * 100, 1),
            nb   = round(predict(nb_model,   scaled, type = "prob")[, "Win"] * 100, 1),
            knn  = round(predict(knn_model,  scaled, type = "prob")[, "Win"] * 100, 1),
            cart = round(predict(cart_model, raw,    type = "prob")[, "Win"] * 100, 1)
        )
    })

    fmt_prob <- function(p) paste0(p, "%")

    output$prob_lr   <- renderText({ fmt_prob(probs()$lr)   })
    output$prob_rf   <- renderText({ fmt_prob(probs()$rf)   })
    output$prob_nb   <- renderText({ fmt_prob(probs()$nb)   })
    output$prob_knn  <- renderText({ fmt_prob(probs()$knn)  })
    output$prob_cart <- renderText({ fmt_prob(probs()$cart) })
    output$prob_avg  <- renderText({
        fmt_prob(round(mean(unlist(probs())), 1))
    })

    output$prob_bar <- renderPlot({
        p <- probs()
        tibble(
            Model = c("LR", "RF", "NB", "KNN", "CART"),
            Prob  = c(p$lr, p$rf, p$nb, p$knn, p$cart),
            Color = c("#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c")
        ) %>%
            ggplot(aes(x = reorder(Model, Prob), y = Prob, fill = Model)) +
            geom_col(width = 0.6) +
            geom_text(aes(label = paste0(Prob, "%")), hjust = -0.2, size = 5) +
            geom_hline(yintercept = 50, linetype = "dashed", color = "grey50") +
            scale_fill_manual(values = setNames(
                c("#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"),
                c("LR", "RF", "NB", "KNN", "CART")
            )) +
            coord_flip() +
            scale_y_continuous(limits = c(0, 110)) +
            labs(title = "Win Probability by Model", x = NULL, y = "Win Probability (%)") +
            theme_minimal() +
            theme(legend.position = "none")
    })

    # Model comparison table
    output$model_table <- renderDT({
        perf_summary %>%
            filter(Method != "LR_all") %>%
            mutate(
                Accuracy = round(Accuracy, 1),
                AUC_ROC  = round(AUC_ROC, 1)
            ) %>%
            rename(`# Features` = n_features, `Accuracy (%)` = Accuracy, `AUC-ROC (%)` = AUC_ROC) %>%
            datatable(rownames = FALSE, options = list(dom = "t", ordering = TRUE)) %>%
            formatStyle("AUC-ROC (%)",
                background = styleColorBar(c(80, 86), "#2ecc71"),
                backgroundSize = "100% 80%", backgroundRepeat = "no-repeat",
                backgroundPosition = "center")
    })

    output$roc_plot <- renderPlot({
        plot(1, type = "n", xlab = "1 - Specificity", ylab = "Sensitivity",
             main = "ROC Curves — run notebook to generate test predictions")
        text(0.5, 0.5, "Save roc objects from analysis.Rmd\nto enable this chart", cex = 1.2, col = "grey50")
    })

    # Feature selection plots
    output$lollipop_plot <- renderPlot({
        baseline_auc <- perf_summary$AUC_ROC[perf_summary$Method == "LR_all"]
        n_all <- max(perf_summary$n_features)

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
            labs(title = "Feature Selection: AUC-ROC vs. Method",
                 subtitle = "Red dashed = LR on all features  ·  Labels = features used",
                 x = "AUC-ROC (%)", y = NULL) +
            theme_minimal()
    })

    output$heatmap_plot <- renderPlot({
        overlap_df %>%
            filter(n_methods > 0) %>%
            pivot_longer(cols = c(RFE, Forward, LASSO, ElasticNet, RF_Imp),
                         names_to = "method", values_to = "selected") %>%
            mutate(
                feature_label = paste0(feature, "  (", n_methods, "/5)"),
                feature_label = reorder(feature_label, n_methods),
                method = factor(method, levels = c("RFE", "Forward", "LASSO", "ElasticNet", "RF_Imp"))
            ) %>%
            ggplot(aes(x = method, y = feature_label, fill = selected)) +
            geom_tile(color = "grey85", linewidth = 0.5) +
            scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "white"),
                              guide = "none") +
            labs(title = "Feature Selection Overlap", x = NULL, y = NULL) +
            theme_minimal() +
            theme(panel.grid = element_blank(),
                  axis.text.x = element_text(face = "bold"))
    })

    # Data explorer
    filtered_data <- reactive({
        df <- teams_model %>%
            select(result, side, golddiffat15, xpdiffat15, csdiffat15,
                   firsttower, firstherald, firstdragon, firstblood, winrate_diff) %>%
            mutate(
                Result = ifelse(result == 1, "Win", "Loss"),
                Side   = ifelse(side == 1, "Blue", "Red")
            ) %>%
            select(-result, -side) %>%
            filter(golddiffat15 >= input$filter_gold[1],
                   golddiffat15 <= input$filter_gold[2])

        if (input$filter_side != "All")
            df <- df %>% filter(Side == input$filter_side)
        if (input$filter_result != "All")
            df <- df %>% filter(Result == input$filter_result)
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
