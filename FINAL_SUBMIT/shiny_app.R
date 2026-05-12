.required_pkgs <- c(
    "shiny", "tidyverse", "caret", "doParallel", "foreach", "pROC",
    "randomForest", "rpart", "rpart.plot", "glmnet", "DT",
    "naivebayes", "xgboost", "MASS", "corrplot", "gridExtra"
)
.missing_pkgs <- setdiff(.required_pkgs, rownames(installed.packages()))
if (length(.missing_pkgs)) {
    message("Installing missing packages: ", paste(.missing_pkgs, collapse = ", "))
    install.packages(.missing_pkgs, repos = "https://cloud.r-project.org")
}

library(shiny)
library(tidyverse)
library(caret)
library(doParallel)
library(foreach)
library(pROC)
library(randomForest)
library(rpart.plot)
library(glmnet)
library(DT)

# Use as many cores as available. The live RF retrain parallelizes over
# (fold x tree-chunk), so the useful ceiling is fold_count x chunks_per_fold.
detected_cores <- parallel::detectCores(logical = TRUE)
if (is.na(detected_cores) || detected_cores < 2) detected_cores <- 2
n_cores <- max(1, detected_cores - 1)
cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)
# Don't register on.exit - runApp() sources this in a function frame, so the
# cluster would die before the first user click. It dies with the R process.

# 5-fold CV; split each fold's 500 trees into chunks so all workers stay busy.
N_FOLDS         <- 5
N_TREES_TOTAL   <- 500
chunks_per_fold <- max(1, ceiling(n_cores / N_FOLDS))
trees_per_chunk <- ceiling(N_TREES_TOTAL / chunks_per_fold)

# Custom 5-fold CV with parallel tree-growing inside each fold. Same statistical
# protocol as caret::train(method="rf", ntree=500): 5-fold CV, predictions at
# threshold 0.5, AUC via pROC, Sens/Spec via caret::confusionMatrix.
recompute_rf_parallel <- function(x, y, mtry_val) {
    set.seed(42)
    folds <- caret::createFolds(y, k = N_FOLDS)

    n_chunks <- chunks_per_fold
    n_trees  <- trees_per_chunk

    tasks <- expand.grid(
        fold  = seq_len(N_FOLDS),
        chunk = seq_len(n_chunks)
    )

    rf_chunks <- foreach::foreach(
        i = seq_len(nrow(tasks)),
        .packages = "randomForest"
    ) %dopar% {
        f         <- tasks$fold[i]
        test_idx  <- folds[[f]]
        train_idx <- setdiff(seq_along(y), test_idx)
        randomForest::randomForest(
            x     = x[train_idx, , drop = FALSE],
            y     = y[train_idx],
            mtry  = mtry_val,
            ntree = n_trees
        )
    }

    fold_results <- lapply(seq_len(N_FOLDS), function(f) {
        chunk_idx   <- which(tasks$fold == f)
        fold_models <- rf_chunks[chunk_idx]
        rf <- if (length(fold_models) == 1) fold_models[[1]]
              else do.call(randomForest::combine, fold_models)
        test_idx <- folds[[f]]
        pred <- predict(rf, newdata = x[test_idx, , drop = FALSE], type = "prob")
        roc_obj <- pROC::roc(y[test_idx], pred[, "Win"], quiet = TRUE,
                             levels = c("Loss", "Win"), direction = "<")
        pred_class <- factor(
            ifelse(pred[, "Win"] > 0.5, "Win", "Loss"),
            levels = c("Loss", "Win")
        )
        cm <- caret::confusionMatrix(pred_class, y[test_idx], positive = "Win")
        list(
            metrics = data.frame(
                ROC  = as.numeric(pROC::auc(roc_obj)),
                Sens = as.numeric(cm$byClass["Sensitivity"]),
                Spec = as.numeric(cm$byClass["Specificity"])
            ),
            predictions = data.frame(
                prob_win = pred[, "Win"],
                actual   = y[test_idx]
            )
        )
    })

    fold_metrics <- do.call(rbind, lapply(fold_results, `[[`, "metrics"))
    all_preds    <- do.call(rbind, lapply(fold_results, `[[`, "predictions"))

    list(
        ROC         = mean(fold_metrics$ROC),
        Sens        = mean(fold_metrics$Sens),
        Spec        = mean(fold_metrics$Spec),
        predictions = all_preds
    )
}

# Generic 5-fold CV trainer for non-RF models. Uses caret with the registered
# parallel backend (parallelizes across folds -> up to 5 cores useful). Saves
# OOF predictions so we can render the diagnostic plots.
recompute_caret <- function(method, x, y, tuneGrid, ...) {
    caret::train(
        x = x, y = y,
        method   = method,
        metric   = "ROC",
        tuneGrid = tuneGrid,
        trControl = caret::trainControl(
            method          = "cv",
            number          = N_FOLDS,
            classProbs      = TRUE,
            summaryFunction = caret::twoClassSummary,
            savePredictions = "final",
            allowParallel   = TRUE
        ),
        ...
    )
}

# Pull (x, y) out of any caret model's stored trainingData.
get_xy <- function(model) {
    td <- model$trainingData
    list(
        x = td[, setdiff(names(td), ".outcome"), drop = FALSE],
        y = td$.outcome
    )
}

models_dir <- "./models"
lr_model    <- readRDS(file.path(models_dir, "lr_model.rds"))
rf_model    <- readRDS(file.path(models_dir, "rf_model.rds"))
nb_model    <- readRDS(file.path(models_dir, "nb_model.rds"))
knn_model   <- readRDS(file.path(models_dir, "knn_model.rds"))
cart_model  <- readRDS(file.path(models_dir, "cart_model.rds"))
xgb_model   <- readRDS(file.path(models_dir, "xgb_model.rds"))
preproc     <- readRDS(file.path(models_dir, "preproc.rds"))
X_train     <- readRDS(file.path(models_dir, "X_train.rds"))
game_lookup <- readRDS(file.path(models_dir, "game_lookup.rds"))
games_model <- readRDS(file.path(models_dir, "games_model.rds"))

# Raw Oracle's Elixir CSV - loaded lazily from the Data Explorer tab
# (80MB; upload path overrides per-session).
RAW_CSV_NAME <- "2025_LoL_esports_match_data_from_OraclesElixir.csv"
.raw_csv_candidates <- c(
    file.path(".", RAW_CSV_NAME)
)
default_raw_path <- .raw_csv_candidates[file.exists(.raw_csv_candidates)][1]

load_default_raw_csv <- function(path) {
    read.csv(path, stringsAsFactors = FALSE,
             check.names = FALSE, na.strings = c("", "NA"))
}

# Reconstruct the 80/20 holdout to evaluate on the same 1,848-row test set.
set.seed(42)
.train_ids <- sample(games_model$gameid, size = floor(0.8 * nrow(games_model)))
.test_data <- games_model[!games_model$gameid %in% .train_ids, ]
X_test         <- .test_data[, !names(.test_data) %in% c("gameid", "blue_win"), drop = FALSE]
y_test_factor  <- factor(.test_data$blue_win, levels = c(0, 1), labels = c("Loss", "Win"))
X_test_scaled  <- predict(preproc, X_test)

predict_test_set <- function(model, scaled = FALSE) {
    nd <- if (scaled) X_test_scaled else X_test
    p  <- predict(model, newdata = nd, type = "prob")
    data.frame(prob_win = p[, "Win"], actual = y_test_factor)
}

test_preds <- list(
    LR   = predict_test_set(lr_model,   scaled = TRUE),
    RF   = predict_test_set(rf_model,   scaled = FALSE),
    NB   = predict_test_set(nb_model,   scaled = FALSE),
    KNN  = predict_test_set(knn_model,  scaled = TRUE),
    CART = predict_test_set(cart_model, scaled = FALSE),
    XGB  = predict_test_set(xgb_model,  scaled = FALSE)
)

# Feature-selection precomputes (Scenario 3): forward stepwise, LASSO,
# Elastic Net. All three share the saved-LR scaled design matrix.
.train_data    <- games_model[games_model$gameid %in% .train_ids, ]
X_train_full   <- .train_data[, !names(.train_data) %in% c("gameid", "blue_win"),
                              drop = FALSE]
X_train_scaled <- predict(preproc, X_train_full)
y_train_factor <- factor(.train_data$blue_win, levels = c(0, 1),
                         labels = c("Loss", "Win"))
y_train_bin    <- as.integer(y_train_factor) - 1L  # Loss=0, Win=1

xs_train_mat <- as.matrix(X_train_scaled)
xs_test_mat  <- as.matrix(X_test_scaled)

# Forward stepwise on logistic regression, AIC-driven, on scaled inputs.
.fs_train_df <- data.frame(blue_win = y_train_bin, X_train_scaled,
                           check.names = FALSE)
.full_glm <- suppressWarnings(glm(blue_win ~ ., data = .fs_train_df,
                                  family = binomial()))
.null_glm <- suppressWarnings(glm(blue_win ~ 1, data = .fs_train_df,
                                  family = binomial()))
fwd_model <- suppressWarnings(MASS::stepAIC(
    .null_glm,
    scope     = list(lower = .null_glm, upper = .full_glm),
    direction = "forward",
    trace     = 0
))
fwd_features <- setdiff(names(coef(fwd_model)), "(Intercept)")

fwd_aic_path <- local({
    available <- names(coef(.full_glm))[-1]
    selected  <- character(0)
    rows      <- list(data.frame(step = 0L, feature = "intercept only",
                                 AIC = AIC(.null_glm), stringsAsFactors = FALSE))
    current   <- .null_glm
    for (s in seq_along(fwd_features)) {
        feat <- fwd_features[s]
        selected <- c(selected, feat)
        fml <- as.formula(paste("blue_win ~", paste(selected, collapse = " + ")))
        current <- suppressWarnings(glm(fml, data = .fs_train_df, family = binomial()))
        rows[[s + 1]] <- data.frame(step = s, feature = feat,
                                    AIC = AIC(current), stringsAsFactors = FALSE)
    }
    do.call(rbind, rows)
})

# LASSO: full λ path + cv.glmnet (AUC, 1se rule).
lasso_fit  <- glmnet::glmnet(xs_train_mat, y_train_bin,
                             family = "binomial", alpha = 1.0)
set.seed(42)
lasso_cv <- glmnet::cv.glmnet(xs_train_mat, y_train_bin,
    family       = "binomial",
    alpha        = 1.0,
    nfolds       = 5,
    type.measure = "auc"
)
lasso_default_lambda <- lasso_cv$lambda.1se

# Elastic Net: α-grid seq(0,1,0.05) with fixed foldids so cv_auc is comparable
# across alphas. Pick the α whose lambda.1se model maximises 5-fold CV AUC,
# then refit glmnet at that α to get the full λ path for the UI.
set.seed(42)
.enet_foldid <- sample(rep(seq(5), length.out = length(y_train_bin)))
.enet_alpha_grid <- seq(0, 1, by = 0.05)

.enet_grid_results <- lapply(.enet_alpha_grid, function(a) {
    cv_fit <- glmnet::cv.glmnet(
        x            = xs_train_mat,
        y            = y_train_bin,
        family       = "binomial",
        alpha        = a,
        foldid       = .enet_foldid,
        type.measure = "auc"
    )
    idx_1se <- cv_fit$index["1se", 1]
    list(
        alpha     = a,
        lambda_1se = cv_fit$lambda.1se,
        cv_auc    = cv_fit$cvm[idx_1se]
    )
})
.enet_grid_df <- do.call(rbind, lapply(.enet_grid_results, function(r)
    data.frame(alpha = r$alpha, lambda_1se = r$lambda_1se, cv_auc = r$cv_auc)))
enet_best_alpha <- .enet_grid_df$alpha[which.max(.enet_grid_df$cv_auc)]

elnet_fit  <- glmnet::glmnet(xs_train_mat, y_train_bin,
                             family = "binomial", alpha = enet_best_alpha)
elnet_cv_final <- glmnet::cv.glmnet(
    x            = xs_train_mat,
    y            = y_train_bin,
    family       = "binomial",
    alpha        = enet_best_alpha,
    foldid       = .enet_foldid,
    type.measure = "auc"
)
elnet_default_lambda <- elnet_cv_final$lambda.1se

# Pre-formatted α string for UI labels.
enet_alpha_label <- sprintf("α = %.2f", enet_best_alpha)

predict_glmnet_test <- function(fit, lambda) {
    p <- as.numeric(predict(fit, newx = xs_test_mat, s = lambda,
                            type = "response"))
    data.frame(prob_win = p, actual = y_test_factor)
}

predict_fwd_test <- function() {
    p <- predict(fwd_model,
                 newdata = data.frame(X_test_scaled, check.names = FALSE),
                 type    = "response")
    data.frame(prob_win = as.numeric(p), actual = y_test_factor)
}

plot_fwd_aic_path <- function(path_df) {
    path_df$label <- ifelse(path_df$step == 0, "",
                            gsub("_", " ", path_df$feature))
    ggplot(path_df, aes(step, AIC)) +
        geom_line(color = "#2c3e50", linewidth = 1) +
        geom_point(color = "#2c3e50", size = 3) +
        geom_text(aes(label = label), hjust = -0.15, vjust = -0.8,
                  size = 3.2, color = "#555555") +
        scale_x_continuous(breaks = path_df$step,
                           expand = expansion(mult = c(0.05, 0.15))) +
        labs(title = "Forward stepwise - AIC at each addition step",
             x = "Step (feature added)", y = "AIC") +
        theme_grey(base_size = 12)
}

# Full LR baseline for the FS comparison panel - same lr_model, no penalty.
fs_full_preds <- test_preds$LR

retained_at <- function(fit, lambda) {
    co <- as.matrix(coef(fit, s = lambda))
    co <- co[rownames(co) != "(Intercept)", , drop = FALSE]
    nz <- abs(co[, 1]) > 1e-10
    data.frame(feature = rownames(co)[nz],
               beta    = co[nz, 1],
               row.names = NULL)
}

feature_names <- names(X_train)
train_means   <- colMeans(X_train)

games_2025 <- game_lookup %>%
    filter(format(as.Date(date), "%Y") == "2025") %>%
    arrange(desc(date))

fill_feature_nas <- function(row) {
    for (f in feature_names) {
        if (is.na(row[[f]])) row[[f]] <- train_means[[f]]
    }
    row
}

build_input_row <- function(input) {
    row <- as.data.frame(as.list(train_means))
    row$golddiffat15  <- input$golddiffat15
    row$xpdiffat15    <- input$xpdiffat15
    row$csdiffat15    <- input$csdiffat15
    row$firstblood    <- as.numeric(input$firstblood)
    row$firstdragon   <- as.numeric(input$firstdragon)
    row$firstherald   <- as.numeric(input$firstherald)
    row$firsttower    <- as.numeric(input$firsttower)
    row$grub_diff     <- input$grub_diff
    row$winrate_diff  <- input$wr_diff / 100
    row[, feature_names, drop = FALSE]
}

predict_prob <- function(model, newdata) {
    p <- tryCatch(
        predict(model, newdata = newdata, type = "prob"),
        error = function(e) NULL
    )
    if (is.null(p)) return(NA_real_)
    if ("Win" %in% colnames(p)) p[["Win"]] else p[, ncol(p)]
}

fmt_pct <- function(x) if (is.na(x)) "-" else sprintf("%.1f%%", x * 100)
snap_to <- function(val, tested) tested[which.min(abs(tested - val))]

# Diagnostic plots (universal binary-classification charts)
filter_pred <- function(pred_df, params) {
    f <- pred_df
    for (k in names(params)) {
        v <- params[[k]]
        if (is.logical(v) || is.character(v)) f <- f[f[[k]] == v, ]
        else f <- f[!is.na(f[[k]]) & abs(f[[k]] - v) < 1e-9, ]
    }
    if (nrow(f) == 0) return(NULL)
    data.frame(prob_win = f$Win, actual = factor(f$obs, levels = c("Loss", "Win")))
}

plot_roc <- function(df) {
    if (is.null(df) || nrow(df) == 0) return(ggplot() + theme_void() +
        labs(title = "ROC curve  -  no data"))
    roc_obj <- pROC::roc(df$actual, df$prob_win,
        levels = c("Loss", "Win"), direction = "<", quiet = TRUE)
    auc_val <- as.numeric(pROC::auc(roc_obj))
    roc_df <- data.frame(
        fpr = 1 - roc_obj$specificities,
        tpr = roc_obj$sensitivities
    )[order(1 - roc_obj$specificities), ]
    ggplot(roc_df, aes(fpr, tpr)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
        geom_line(linewidth = 1.1, color = "#2c3e50") +
        scale_x_continuous(limits = c(0, 1)) +
        scale_y_continuous(limits = c(0, 1)) +
        labs(title = sprintf("ROC curve  -  AUC = %.3f", auc_val),
             x = "False Positive Rate", y = "True Positive Rate") +
        coord_equal() +
        theme_grey(base_size = 12)
}

compute_all_metrics <- function(df, threshold = 0.5) {
    if (is.null(df) || nrow(df) == 0) {
        return(data.frame(Metric = character(), Value = character()))
    }
    pred <- factor(ifelse(df$prob_win > threshold, "Win", "Loss"),
                   levels = c("Loss", "Win"))
    cm <- caret::confusionMatrix(pred, df$actual, positive = "Win")
    auc_val <- as.numeric(pROC::auc(pROC::roc(df$actual, df$prob_win,
        levels = c("Loss", "Win"), direction = "<", quiet = TRUE)))
    pct <- function(x) sprintf("%.2f%%", x * 100)
    num <- function(x) sprintf("%.3f",   x)
    data.frame(
        Metric = c("AUC-ROC",
                   "Accuracy",
                   "Sensitivity (Recall)",
                   "Specificity",
                   "Precision (PPV)",
                   "F1 score"),
        Value = c(num(auc_val),
                  pct(unname(cm$overall["Accuracy"])),
                  pct(unname(cm$byClass["Sensitivity"])),
                  pct(unname(cm$byClass["Specificity"])),
                  pct(unname(cm$byClass["Pos Pred Value"])),
                  num(unname(cm$byClass["F1"]))),
        stringsAsFactors = FALSE
    )
}

# Model-specific signature plots
plot_lr_or <- function(model) {
    co <- summary(model$finalModel)$coefficients
    co <- co[rownames(co) != "(Intercept)", , drop = FALSE]
    est <- co[, "Estimate"]
    se  <- co[, "Std. Error"]
    df <- data.frame(
        feature = rownames(co),
        or      = exp(est),
        lo      = exp(est - 1.96 * se),
        hi      = exp(est + 1.96 * se)
    )
    df$feature <- factor(df$feature, levels = df$feature[order(df$or)])
    ggplot(df, aes(or, feature)) +
        geom_vline(xintercept = 1, linetype = "dashed", color = "grey60") +
        geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.25, color = "#2c3e50") +
        geom_point(size = 3, color = "#2c3e50") +
        scale_x_log10() +
        labs(title = "Odds ratios (log scale, 95% CI)",
             subtitle = "OR > 1 raises P(Win); OR < 1 lowers it",
             x = "Odds ratio", y = NULL) +
        theme_grey(base_size = 12)
}

plot_varimp_lr <- function(model) {
    co <- coef(model$finalModel)
    co <- co[names(co) != "(Intercept)"]
    abs_co <- abs(co)
    abs_co <- abs_co / max(abs_co) * 100
    df <- data.frame(feature = names(abs_co), score = unname(abs_co))
    df$feature <- factor(df$feature, levels = df$feature[order(df$score)])
    ggplot(df, aes(score, feature)) +
        geom_col(fill = "#2c3e50") +
        labs(title = "Feature importance (LR, standardized |coefficient|, scaled 0-100)",
             x = "Importance", y = NULL) +
        theme_grey(base_size = 12)
}

plot_varimp <- function(model, title) {
    vi_obj <- tryCatch(caret::varImp(model), error = function(e) NULL)
    if (is.null(vi_obj)) return(ggplot() + theme_void() +
        labs(title = paste(title, " -  varImp unavailable")))
    vi <- if (is.data.frame(vi_obj$importance)) vi_obj$importance
          else if (is.data.frame(vi_obj)) vi_obj
          else return(ggplot() + theme_void() +
              labs(title = paste(title, " -  varImp unavailable")))
    vi$feature <- rownames(vi)
    score_col <- if ("Overall" %in% names(vi)) "Overall" else names(vi)[1]
    vi$score <- vi[[score_col]]
    vi <- vi[order(-vi$score), ]
    vi$feature <- factor(vi$feature, levels = rev(vi$feature))
    ggplot(vi, aes(score, feature)) +
        geom_col(fill = "#2c3e50") +
        labs(title = title, x = "Importance (scaled 0-100)", y = NULL) +
        theme_grey(base_size = 12)
}

plot_cart_tree <- function(model) {
    rpart.plot::rpart.plot(
        model$finalModel,
        type = 5, extra = 104, fallen.leaves = TRUE,
        box.palette = "BuGn", branch.lty = 3, shadow.col = "grey90",
        main = "Decision tree (saved bestTune)"
    )
}

plot_calibration <- function(df, n_bins = 10) {
    if (is.null(df) || nrow(df) == 0) return(ggplot() + theme_void() +
        labs(title = "Calibration  -  no data"))
    breaks <- seq(0, 1, length.out = n_bins + 1)
    df2 <- df %>%
        mutate(bin = cut(prob_win, breaks = breaks, include.lowest = TRUE)) %>%
        group_by(bin) %>%
        summarise(
            mean_pred   = mean(prob_win),
            mean_actual = mean(actual == "Win"),
            n           = dplyr::n(),
            .groups     = "drop"
        ) %>%
        filter(!is.na(bin))
    ggplot(df2, aes(mean_pred, mean_actual)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
        geom_line(color = "#3498db", linewidth = 1) +
        geom_point(aes(size = n), color = "#2c3e50") +
        scale_x_continuous(limits = c(0, 1)) +
        scale_y_continuous(limits = c(0, 1)) +
        labs(title = "Calibration curve",
             subtitle = "Diagonal = perfectly calibrated",
             x = "Mean predicted P(Win)", y = "Observed fraction of Win",
             size = "Bin n") +
        coord_equal() +
        theme_grey(base_size = 12) +
        theme(legend.position = "right")
}

# Cross-model comparison helpers
MODEL_FAMILY <- c(
    LR   = "Statistical / Parametric",
    NB   = "Statistical / Parametric",
    RF   = "Ensemble / Non-parametric",
    CART = "Partitioning / Non-parametric",
    KNN  = "Partitioning / Non-parametric",
    XGB  = "Ensemble / Non-parametric"
)

MODEL_LABEL <- c(
    LR   = "Logistic Regression",
    NB   = "Naive Bayes",
    RF   = "Random Forest",
    CART = "Decision Tree (CART)",
    KNN  = "K-Nearest Neighbors",
    XGB  = "XGBoost"
)

MODEL_COLOR <- c(
    LR   = "#2c3e50",
    NB   = "#16a085",
    RF   = "#d35400",
    CART = "#8e44ad",
    KNN  = "#2980b9",
    XGB  = "#c0392b"
)

plot_roc_overlay <- function(named_dfs) {
    if (length(named_dfs) == 0) return(ggplot() + theme_void() +
        labs(title = "ROC overlay  -  pick at least one model"))
    rows <- list()
    auc_lab <- c()
    for (nm in names(named_dfs)) {
        df <- named_dfs[[nm]]
        if (is.null(df) || nrow(df) == 0) next
        roc_obj <- pROC::roc(df$actual, df$prob_win,
            levels = c("Loss", "Win"), direction = "<", quiet = TRUE)
        auc_v <- as.numeric(pROC::auc(roc_obj))
        rows[[nm]] <- data.frame(
            fpr = 1 - roc_obj$specificities,
            tpr = roc_obj$sensitivities,
            model = nm
        )
        auc_lab[nm] <- sprintf("%s  -  AUC %.3f",
            ifelse(nm %in% names(MODEL_LABEL), MODEL_LABEL[[nm]], nm),
            auc_v)
    }
    if (length(rows) == 0) return(ggplot() + theme_void() +
        labs(title = "ROC overlay  -  no predictions"))
    big <- do.call(rbind, rows)
    big$model <- factor(big$model, levels = names(auc_lab),
                        labels = unname(auc_lab))
    color_vec <- if (all(names(named_dfs) %in% names(MODEL_COLOR)))
                     unname(MODEL_COLOR[names(named_dfs)]) else NULL
    p <- ggplot(big, aes(fpr, tpr, color = model)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                    color = "grey60") +
        geom_line(linewidth = 1.05) +
        coord_equal() +
        scale_x_continuous(limits = c(0, 1)) +
        scale_y_continuous(limits = c(0, 1)) +
        labs(title = "ROC overlay  -  1,848-row held-out test set",
             x = "False Positive Rate", y = "True Positive Rate",
             color = NULL) +
        theme_grey(base_size = 12) +
        theme(legend.position = "right")
    if (!is.null(color_vec)) p <- p + scale_color_manual(values = color_vec)
    p
}

metrics_matrix <- function(named_dfs, threshold = 0.5) {
    if (length(named_dfs) == 0) {
        return(data.frame(Model = character(), AUC = character(),
                          Acc = character(), Sens = character(),
                          Spec = character(), Prec = character(),
                          F1 = character()))
    }
    rows <- lapply(names(named_dfs), function(nm) {
        df <- named_dfs[[nm]]
        if (is.null(df) || nrow(df) == 0) return(NULL)
        pred <- factor(ifelse(df$prob_win > threshold, "Win", "Loss"),
                       levels = c("Loss", "Win"))
        cm <- caret::confusionMatrix(pred, df$actual, positive = "Win")
        auc_v <- as.numeric(pROC::auc(pROC::roc(df$actual, df$prob_win,
            levels = c("Loss", "Win"), direction = "<", quiet = TRUE)))
        data.frame(
            Model = if (nm %in% names(MODEL_LABEL)) MODEL_LABEL[[nm]] else nm,
            Family = if (nm %in% names(MODEL_FAMILY)) MODEL_FAMILY[[nm]] else "",
            AUC   = sprintf("%.3f", auc_v),
            Acc   = sprintf("%.1f%%", cm$overall["Accuracy"] * 100),
            Sens  = sprintf("%.1f%%", cm$byClass["Sensitivity"] * 100),
            Spec  = sprintf("%.1f%%", cm$byClass["Specificity"] * 100),
            Prec  = sprintf("%.1f%%", cm$byClass["Pos Pred Value"] * 100),
            F1    = sprintf("%.3f",   cm$byClass["F1"]),
            stringsAsFactors = FALSE
        )
    })
    do.call(rbind, rows)
}

# Per-model top-N feature importance, normalised to 0-100 within each model.
plot_topfeats_panel <- function(model_keys, top_n = 5) {
    if (length(model_keys) == 0) return(ggplot() + theme_void() +
        labs(title = "Top features  -  pick at least one model"))
    rows <- list()
    for (k in model_keys) {
        m <- switch(k,
            LR = lr_model, RF = rf_model, NB = nb_model,
            KNN = knn_model, CART = cart_model, XGB = xgb_model)
        if (k == "LR") {
            co <- coef(m$finalModel)
            co <- co[names(co) != "(Intercept)"]
            abs_co <- abs(co)
            abs_co <- abs_co / max(abs_co) * 100
            d <- data.frame(feature = names(abs_co), score = unname(abs_co))
        } else {
            vi <- tryCatch(caret::varImp(m)$importance, error = function(e) NULL)
            if (is.null(vi) || nrow(vi) == 0) next
            score_col <- if ("Overall" %in% names(vi)) "Overall"
                         else if ("Win" %in% names(vi)) "Win"
                         else names(vi)[1]
            d <- data.frame(feature = rownames(vi),
                            score   = vi[[score_col]])
        }
        d <- d[order(-d$score), ][seq_len(min(top_n, nrow(d))), ]
        d$model <- if (k %in% names(MODEL_LABEL)) MODEL_LABEL[[k]] else k
        d$rank  <- seq_len(nrow(d))
        rows[[k]] <- d
    }
    if (length(rows) == 0) return(ggplot() + theme_void() +
        labs(title = "Top features  -  no varImp available"))
    big <- do.call(rbind, rows)
    big$feature <- factor(big$feature)
    ggplot(big, aes(x = score, y = reorder(feature, score), fill = model)) +
        geom_col(show.legend = FALSE) +
        facet_wrap(~ model, scales = "free_y", ncol = 3) +
        scale_fill_manual(values = unname(MODEL_COLOR[
            match(unique(big$model), MODEL_LABEL)])) +
        labs(title = sprintf("Top %d features per model (caret::varImp, scaled 0-100)", top_n),
             x = "Importance", y = NULL) +
        theme_grey(base_size = 11) +
        theme(strip.text = element_text(face = "bold"))
}

# Feature-selection plots
plot_glmnet_path <- function(fit, current_lambda, title) {
    co_mat <- as.matrix(fit$beta)
    lambdas <- fit$lambda
    df <- data.frame(
        feature = rep(rownames(co_mat), ncol(co_mat)),
        beta    = as.numeric(co_mat),
        log_l   = rep(log(lambdas), each = nrow(co_mat))
    )
    last_betas <- co_mat[, 1, drop = TRUE]
    label_features <- names(sort(abs(last_betas), decreasing = TRUE)[1:5])
    df$label_at <- ifelse(df$feature %in% label_features &
                          df$log_l == min(df$log_l), df$feature, NA)
    ggplot(df, aes(log_l, beta, color = feature)) +
        geom_line(linewidth = 0.7) +
        geom_vline(xintercept = log(current_lambda),
                   linetype = "dashed", color = "grey40") +
        labs(title = title,
             subtitle = "Each line = one feature's coefficient over the λ path. Dashed line = current λ.",
             x = "log(λ)", y = "Coefficient (β, scaled inputs)",
             color = "Feature") +
        theme_grey(base_size = 12) +
        theme(legend.position = "right",
              legend.key.height = unit(0.9, "lines"))
}

plot_retained_bar <- function(retained_df, total_features, title) {
    if (nrow(retained_df) == 0) return(ggplot() + theme_void() +
        labs(title = paste(title, "- all features dropped at this λ")))
    df <- retained_df
    df <- df[order(abs(df$beta), decreasing = TRUE), ]
    df$feature <- factor(df$feature, levels = rev(df$feature))
    df$direction <- ifelse(df$beta > 0, "Raises P(Win)", "Lowers P(Win)")
    ggplot(df, aes(beta, feature, fill = direction)) +
        geom_col() +
        geom_vline(xintercept = 0, color = "grey50") +
        scale_fill_manual(values = c("Raises P(Win)" = "#27ae60",
                                     "Lowers P(Win)" = "#c0392b")) +
        labs(title = sprintf("%s  -  %d / %d features retained",
                             title, nrow(retained_df), total_features),
             x = "Coefficient (scaled β)", y = NULL, fill = NULL) +
        theme_grey(base_size = 12) +
        theme(legend.position = "top")
}

plot_knn_elbow <- function(model, live_df = NULL, current_k = NULL) {
    pre <- model$results
    pre$Source <- "Pre-tested"
    df <- pre[, c("k", "ROC", "Source")]
    if (!is.null(live_df) && nrow(live_df) > 0) {
        ld <- data.frame(k = live_df$k, ROC = live_df$ROC, Source = "Live")
        df <- rbind(df, ld)
    }
    df <- df[order(df$k), ]
    p <- ggplot(df, aes(k, ROC, color = Source, shape = Source)) +
        geom_line(data = subset(df, Source == "Pre-tested"),
                  linewidth = 1, color = "#2c3e50") +
        geom_point(size = 3) +
        scale_color_manual(values = c("Pre-tested" = "#2c3e50", "Live" = "#e67e22")) +
        labs(title = "AUC-ROC vs k (elbow)",
             x = "k (neighbors)", y = "CV AUC-ROC") +
        theme_grey(base_size = 12) +
        theme(legend.position = "top")
    if (!is.null(current_k)) {
        p <- p + geom_vline(xintercept = current_k,
                            linetype = "dashed", color = "grey50")
    }
    p
}

# EDA helpers
EDA_GAME_COLS <- c(
    "result",
    "goldat10", "xpat10", "csat10",
    "goldat15", "xpat15", "csat15",
    "golddiffat10", "xpdiffat10", "csdiffat10",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "killsat10", "deathsat10", "assistsat10",
    "killsat15", "deathsat15", "assistsat15",
    "firstblood", "firstdragon", "firstherald", "firsttower",
    "void_grubs", "turretplates"
)

EDA_EARLY_COLS <- c(
    "blue_goldat10", "blue_xpat10", "blue_csat10",
    "blue_golddiffat10", "blue_xpdiffat10", "blue_csdiffat10",
    "blue_killsat10", "blue_deathsat10", "blue_assistsat10",
    "blue_goldat15", "blue_xpat15", "blue_csat15",
    "blue_golddiffat15", "blue_xpdiffat15", "blue_csdiffat15",
    "blue_killsat15", "blue_deathsat15", "blue_assistsat15",
    "blue_firstblood", "blue_firstdragon", "blue_firstherald", "blue_firsttower",
    "blue_void_grubs", "red_void_grubs",
    "blue_turretplates", "red_turretplates",
    "blue_win"
)

empty_plot <- function(title) {
    ggplot() + theme_void() + labs(title = title)
}

prepare_eda_data <- function(raw) {
    if (is.null(raw) || nrow(raw) == 0) {
        return(list(error = "No dataset loaded.", games = NULL, players = NULL))
    }

    required <- c("gameid", "side", "position", "datacompleteness", EDA_GAME_COLS)
    missing <- setdiff(required, names(raw))
    if (length(missing) > 0) {
        return(list(
            error = paste("Missing required columns:", paste(missing, collapse = ", ")),
            games = NULL,
            players = NULL
        ))
    }

    complete <- raw %>% filter(datacompleteness == "complete")
    teams <- complete %>% filter(position == "team")
    players <- complete %>% filter(position != "team")

    games_full <- teams %>%
        select(gameid, side, all_of(EDA_GAME_COLS)) %>%
        mutate(side = tolower(side)) %>%
        pivot_wider(
            id_cols     = gameid,
            names_from  = side,
            values_from = all_of(EDA_GAME_COLS),
            names_glue  = "{side}_{.value}"
        )

    required_after_pivot <- c(
        "blue_result", "red_result",
        setdiff(EDA_EARLY_COLS, "blue_win")
    )
    missing_after_pivot <- setdiff(required_after_pivot, names(games_full))
    if (length(missing_after_pivot) > 0) {
        return(list(
            error = paste("Could not build blue/red game table. Missing after pivot:",
                          paste(missing_after_pivot, collapse = ", ")),
            games = NULL,
            players = players
        ))
    }

    redundant_mirrors <- c(
        "red_golddiffat10", "red_xpdiffat10", "red_csdiffat10",
        "red_golddiffat15", "red_xpdiffat15", "red_csdiffat15",
        "red_firstblood", "red_firstdragon", "red_firstherald", "red_firsttower"
    )

    games <- games_full %>%
        mutate(blue_win = blue_result) %>%
        select(-blue_result, -red_result, -any_of(redundant_mirrors))

    list(
        error = NULL,
        games = games,
        players = players,
        counts = list(
            raw = nrow(raw),
            complete = nrow(complete),
            teams = nrow(teams),
            players = nrow(players)
        )
    )
}

eda_numeric_summary <- function(games) {
    cols <- setdiff(EDA_EARLY_COLS, "blue_win")
    games %>%
        select(all_of(cols)) %>%
        pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
        drop_na() %>%
        group_by(feature) %>%
        summarise(
            n = n(),
            mean = mean(value),
            sd = sd(value),
            min = min(value),
            q25 = quantile(value, 0.25),
            median = median(value),
            q75 = quantile(value, 0.75),
            max = max(value),
            .groups = "drop"
        ) %>%
        mutate(across(c(mean, sd, min, q25, median, q75, max), ~ round(.x, 2))) %>%
        arrange(desc(sd))
}

eda_result_correlations <- function(games) {
    cor_data <- games %>% select(all_of(EDA_EARLY_COLS)) %>% drop_na()
    if (nrow(cor_data) == 0) {
        return(tibble(feature = character(), correlation = numeric()))
    }
    cor_data %>%
        select(-blue_win) %>%
        summarise(across(everything(), ~ cor(.x, cor_data$blue_win))) %>%
        pivot_longer(everything(), names_to = "feature", values_to = "correlation") %>%
        arrange(desc(abs(correlation)))
}

eda_outcome_means <- function(games) {
    games %>%
        select(all_of(EDA_EARLY_COLS)) %>%
        drop_na() %>%
        mutate(outcome = if_else(blue_win == 1, "blue_win", "red_win")) %>%
        select(-blue_win) %>%
        pivot_longer(-outcome, names_to = "feature", values_to = "value") %>%
        group_by(feature, outcome) %>%
        summarise(mean = mean(value), .groups = "drop") %>%
        pivot_wider(names_from = outcome, values_from = mean) %>%
        mutate(diff = blue_win - red_win) %>%
        arrange(desc(abs(diff))) %>%
        mutate(across(c(blue_win, red_win, diff), ~ round(.x, 3)))
}

eda_objective_winrates <- function(games) {
    obj_cols <- c("firstblood", "firstdragon", "firstherald", "firsttower")
    purrr::map_dfr(obj_cols, function(obj) {
        blue_col <- paste0("blue_", obj)
        df <- games %>%
            select(all_of(c(blue_col, "blue_win"))) %>%
            drop_na()
        if (nrow(df) == 0) return(NULL)
        chi <- suppressWarnings(chisq.test(table(df[[blue_col]], df$blue_win)))
        tibble(
            objective = str_replace(obj, "first", "First ") %>% str_to_title(),
            secured_by = c("Blue secured", "Red secured"),
            win_rate_blue = c(
                mean(df$blue_win[df[[blue_col]] == 1]) * 100,
                mean(df$blue_win[df[[blue_col]] == 0]) * 100
            ),
            n = c(sum(df[[blue_col]] == 1), sum(df[[blue_col]] == 0)),
            chi2 = as.numeric(chi$statistic),
            p_value = chi$p.value
        )
    })
}

eda_role_correlations <- function(games, players) {
    role_diffs_blue <- players %>%
        filter(
            position %in% c("top", "jng", "mid", "bot", "sup"),
            side == "Blue",
            !is.na(golddiffat15)
        ) %>%
        select(gameid, position, golddiffat15)

    role_diffs_blue %>%
        inner_join(games %>% select(gameid, blue_win), by = "gameid") %>%
        group_by(position) %>%
        summarise(
            correlation = cor(golddiffat15, blue_win),
            mean_diff_win = mean(golddiffat15[blue_win == 1]),
            mean_diff_loss = mean(golddiffat15[blue_win == 0]),
            .groups = "drop"
        ) %>%
        mutate(position = toupper(position))
}

# UI
ui <- navbarPage(
    title = "LoL Early-Game Win Predictor",

    # Tab: Data Explorer
    tabPanel("Data Explorer",
        sidebarLayout(
            sidebarPanel(
                h4("Data source"),
                radioButtons("data_source", NULL,
                    choices = c("Default location (preloaded)" = "default",
                                "Upload CSV file"               = "upload"),
                    selected = "default"),
                conditionalPanel(
                    condition = "input.data_source == 'upload'",
                    fileInput("data_file", "Upload Oracle's Elixir CSV",
                              accept = c(".csv", "text/csv"),
                              buttonLabel = "Browse...",
                              placeholder = "no file selected"),
                    helpText("Expects the same column layout as the 2025 Oracle's Elixir match-data CSV.")
                ),
                conditionalPanel(
                    condition = "input.data_source == 'default'",
                    uiOutput("data_default_path_note"),
                    actionButton("load_default_csv", "Load CSV from current location",
                                 icon = icon("download"), class = "btn-primary"),
                    helpText("The 80MB CSV is loaded only on demand to keep startup fast.")
                ),
                hr(),
                h4("Dataset summary"),
                uiOutput("data_summary"),
                hr(),
                h4("Display"),
                checkboxInput("data_only_team_rows",
                    "Show team rows only (drop player rows)", FALSE),
                helpText("Each game has 12 rows: 5 players + 1 team aggregate per side. Toggle on to view only the per-team rows.")
            ),
            mainPanel(
                h3("Raw Oracle's Elixir 2025 LoL esports match data"),
                p("Source (credit): ",
                  tags$a(href = "https://oracleselixir.com/tools/downloads",
                         "Oracle's Elixir", target = "_blank"),
                  " - maintained by Tim Sevenhuysen. Public dataset, free for academic use."),
                DT::dataTableOutput("data_table"),
                hr(),
                h4("Column overview"),
                p("Type and missing-value summary for every column in the loaded CSV."),
                DT::dataTableOutput("data_columns_table")
            )
        )
    ),

    # Tab: EDA
    tabPanel("EDA",
        fluidPage(
            h2("Exploratory Data Analysis"),
            uiOutput("eda_status"),
            hr(),
            h3("Basic metrics"),
            fluidRow(
                column(4, uiOutput("eda_basic_metrics")),
                column(8, plotOutput("eda_target_plot", height = "300px"))
            ),
            hr(),
            h3("Feature summaries"),
            fluidRow(
                column(6, plotOutput("eda_correlation_plot", height = "620px")),
                column(6, plotOutput("eda_outcome_means_plot", height = "520px"))
            ),
            fluidRow(
                column(6,
                    h4("Descriptive statistics"),
                    DT::dataTableOutput("eda_descriptive_table")
                ),
                column(6,
                    h4("Correlation with blue_win"),
                    DT::dataTableOutput("eda_correlation_table")
                )
            ),
            hr(),
            h3("Distributions and game-state signals"),
            fluidRow(
                column(6, plotOutput("eda_gold_density_plot", height = "360px")),
                column(6, plotOutput("eda_gold_scatter_plot", height = "360px"))
            ),
            fluidRow(
                column(6, plotOutput("eda_distribution_plot", height = "420px")),
                column(6, plotOutput("eda_grubs_plot", height = "420px"))
            ),
            hr(),
            h3("Objectives, side, and roles"),
            fluidRow(
                column(6, plotOutput("eda_objectives_plot", height = "420px")),
                column(6, plotOutput("eda_side_plot", height = "420px"))
            ),
            fluidRow(
                column(6, plotOutput("eda_roles_plot", height = "420px")),
                column(6, plotOutput("eda_roles_boxplot", height = "420px"))
            ),
            h4("Per-outcome means"),
            DT::dataTableOutput("eda_outcome_means_table")
        )
    ),

    # Tab: Match Predictor
    tabPanel("Match Predictor",
        sidebarLayout(
            sidebarPanel(
                h4("Pre-fill from a 2025 game"),
                selectInput("game_league", "League", choices = NULL),
                selectInput("game_pick", "Game", choices = NULL),
                helpText("Picking a game auto-fills the inputs below. Tweak any value to override."),
                hr(),

                h4("Match conditions"),
                helpText("All inputs are from the Blue team's perspective. The game-picker pre-fills the correct signed differentials for the chosen side."),

                strong("Objectives"),
                checkboxInput("firstblood",  "First blood",  TRUE),
                checkboxInput("firstdragon", "First dragon", TRUE),
                checkboxInput("firstherald", "First herald", FALSE),
                checkboxInput("firsttower",  "First tower",  FALSE),
                sliderInput("grub_diff", "Void grub advantage",
                    min = -6, max = 6, value = 2, step = 1),

                strong("Pre-game"),
                sliderInput("wr_diff", "Win rate advantage (%)",
                    min = -50, max = 50, value = 10, step = 1),

                strong("Economy at 15 min"),
                sliderInput("golddiffat15", "Gold difference",
                    min = -10000, max = 10000, value = 1500, step = 100),
                sliderInput("xpdiffat15", "Experience difference",
                    min = -6000, max = 6000, value = 800, step = 100),
                sliderInput("csdiffat15", "CS difference",
                    min = -80, max = 80, value = 12, step = 1)
            ),
            mainPanel(
                uiOutput("game_banner"),
                h4("Predicted win probability"),
                uiOutput("target_banner"),
                tableOutput("prob_table"),
                plotOutput("prob_plot", height = "320px"),
                helpText("A probability above 50% means that model predicts the team to win, given the early-game state on the left.")
            )
        )
    ),

    # Tab: Hyperparameter Tuning
    tabPanel("Hyperparameter Tuning",
        sidebarLayout(
            sidebarPanel(
                h4("Select model"),
                radioButtons("tune_model", NULL, selected = "RF",
                    choiceNames  = list(
                        "Logistic Regression (parametric)",
                        "Random Forest (ensemble)",
                        "Naive Bayes (probabilistic)",
                        "K-Nearest Neighbors (partitioning)",
                        "Decision Tree / CART (partitioning)",
                        "XGBoost (ensemble)"),
                    choiceValues = c("LR", "RF", "NB", "KNN", "CART", "XGB")),
                hr(),
                uiOutput("tune_description"),
                hr(),
                h4("Adjust parameters"),
                uiOutput("tune_slider"),
                hr(),
                h4("CV best"),
                uiOutput("tune_best")
            ),
            mainPanel(
                uiOutput("diag_source_note"),
                h4("Performance metrics"),
                tableOutput("metrics_table"),
                hr(),
                fluidRow(
                    column(6, plotOutput("diag_roc",      height = "360px")),
                    column(6, plotOutput("diag_specific", height = "360px"))
                ),
                conditionalPanel(
                    condition = "input.tune_model !== 'LR'",
                    hr(),
                    h4("All tested configurations"),
                    tableOutput("tune_table")
                )
            )
        )
    ),

    # Tab: Model Comparison
    tabPanel("Model Comparison",
        fluidPage(
            h2("Model comparison  -  Scenarios 1 + 3 on the same held-out test set"),
            helpText(sprintf("All metrics below come from the saved %d-row holdout (set.seed(42)). Compare like with like.",
                             nrow(test_preds[[1]]))),
            hr(),

            # Section 1: Method comparison (Scenario 1)
            h3("1. Model Comparison"),
            p("Scenario 1: six classifiers across three families - Statistical, Ensemble, and Partitioning. Pick which to overlay on the right."),
            fluidRow(
                column(3,
                    h4("Choose models"),
                    checkboxGroupInput("cmp_models", NULL,
                        choiceNames  = list(
                            "Logistic Regression",
                            "Naive Bayes",
                            "Random Forest",
                            "Decision Tree (CART)",
                            "K-Nearest Neighbors",
                            "XGBoost"
                        ),
                        choiceValues = c("LR", "NB", "RF", "CART", "KNN", "XGB"),
                        selected     = c("LR", "NB", "RF", "CART", "KNN")),
                    helpText("Top-features panel uses caret::varImp - values are scaled 0-100 within each model.")
                ),
                column(9,
                    plotOutput("cmp_roc_overlay", height = "380px"),
                    h4("Metrics matrix"),
                    tableOutput("cmp_metrics_table"),
                    h4("Top-N features per model"),
                    plotOutput("cmp_topfeats", height = "380px")
                )
            ),
            hr(),

            # Section 2: Feature selection (Scenario 3)
            h3("2. Feature selection  -  full LR vs algorithmic vs embedded"),
            p(sprintf("Scenario 3: one algorithmic (forward stepwise, AIC) and two embedded (LASSO, Elastic Net at %s - best from α-grid seq(0,1,0.05)) regularised logistic regressions. The full LR is shown as the no-FS baseline. Slide λ to see how many features survive and how AUC moves.",
                      enet_alpha_label)),
            fluidRow(
                column(3,
                    h4("FS configuration"),
                    radioButtons("fs_method", "Feature-selection method",
                        choiceNames = c(
                            "None (full LR)",
                            "Forward stepwise (AIC)",
                            "LASSO (glmnet, α = 1)",
                            sprintf("Elastic Net (%s)", enet_alpha_label)),
                        choiceValues = c("none", "forward", "lasso", "elnet"),
                        selected = "lasso"),
                    conditionalPanel(
                        condition = "input.fs_method == 'lasso' ||
                                     input.fs_method == 'elnet'",
                        sliderInput("fs_log_lambda",
                            "log10(λ) - slide left for less penalty",
                            min = -6, max = 1, value = -2.5, step = 0.05,
                            width = "100%"),
                        actionButton("fs_use_default", "Reset to cv.glmnet 1-SE λ"),
                        br(), br(),
                        uiOutput("fs_lambda_summary")
                    ),
                    conditionalPanel(
                        condition = "input.fs_method == 'forward'",
                        helpText(sprintf("Forward stepwise selected %d features by AIC (precomputed at startup).",
                                          length(fwd_features)))
                    ),
                ),
                column(9,
                    fluidRow(
                        column(6, plotOutput("fs_roc",      height = "360px")),
                        column(6, plotOutput("fs_retained", height = "360px"))
                    ),
                    conditionalPanel(
                        condition = "input.fs_method == 'lasso' ||
                                     input.fs_method == 'elnet'",
                        plotOutput("fs_path", height = "320px")
                    ),
                    conditionalPanel(
                        condition = "input.fs_method == 'forward'",
                        plotOutput("fs_aic_path", height = "320px")
                    ),
                    h4("FS vs no-FS summary"),
                    tableOutput("fs_summary_table")
                )
            )
        )
    )
)

# Server
server <- function(input, output, session) {

    rv <- reactiveValues(actual_result = NULL, game_label = NULL)

    # Game picker
    observe({
        updateSelectInput(session, "game_league",
            choices  = c("- pick a league -" = "", sort(unique(games_2025$league))))
    })

    observeEvent(input$game_league, {
        if (!nzchar(input$game_league)) {
            updateSelectInput(session, "game_pick", choices = c("- pick a game -" = ""))
            return()
        }
        gs <- games_2025 %>% filter(league == input$game_league)
        labels <- sprintf("%s - %s vs %s",
                          format(as.Date(gs$date), "%Y-%m-%d"),
                          gs$teamname, gs$opp_teamname)
        ids <- as.character(seq_len(nrow(gs)))
        updateSelectInput(session, "game_pick",
            choices = c("- pick a game -" = "", setNames(ids, labels)))
    })

    observeEvent(input$game_pick, {
        if (!nzchar(input$game_pick) || !nzchar(input$game_league)) {
            rv$actual_result <- NULL
            rv$game_label    <- NULL
            return()
        }
        gs <- games_2025 %>% filter(league == input$game_league)
        idx <- as.integer(input$game_pick)
        if (idx < 1 || idx > nrow(gs)) return()
        g <- gs[idx, ]

        updateSliderInput(session,   "golddiffat15", value = round(g$golddiffat15))
        updateSliderInput(session,   "xpdiffat15",   value = round(g$xpdiffat15))
        updateSliderInput(session,   "csdiffat15",   value = round(g$csdiffat15))
        updateCheckboxInput(session, "firstblood",   value = as.logical(g$firstblood))
        updateCheckboxInput(session, "firstdragon",  value = as.logical(g$firstdragon))
        updateCheckboxInput(session, "firstherald",  value = as.logical(g$firstherald))
        updateCheckboxInput(session, "firsttower",   value = as.logical(g$firsttower))
        updateSliderInput(session,   "grub_diff",    value = as.integer(g$grub_diff))
        updateSliderInput(session,   "wr_diff",      value = round(g$winrate_diff * 100, 1))

        rv$actual_result <- as.character(g$result_label)
        rv$game_label    <- sprintf("%s vs %s  -  %s  -  %s  -  %s side",
                                    g$teamname, g$opp_teamname,
                                    format(as.Date(g$date), "%Y-%m-%d"),
                                    g$league, g$side_label)
    })

    output$game_banner <- renderUI({
        if (is.null(rv$game_label)) return(NULL)
        wellPanel(
            tags$b(rv$game_label),
            br(),
            tags$span("Actual result: "),
            tags$b(rv$actual_result %||% "-")
        )
    })

    # Match Predictor probabilities
    # LR/KNN need the scaled design matrix (preproc); RF/NB/CART/XGB consume raw.
    probs <- reactive({
        row <- build_input_row(input)
        row_scaled <- predict(preproc, row)
        list(
            LR   = predict_prob(lr_model,   row_scaled),
            RF   = predict_prob(rf_model,   row),
            NB   = predict_prob(nb_model,   row),
            KNN  = predict_prob(knn_model,  row_scaled),
            CART = predict_prob(cart_model, row),
            XGB  = predict_prob(xgb_model,  row)
        )
    })

    output$prob_table <- renderTable({
        p <- probs()
        data.frame(
            Model = c("Logistic Regression", "Random Forest", "Naive Bayes",
                      "K-Nearest Neighbors", "Decision Tree (CART)", "XGBoost", "Average"),
            `Win probability` = c(
                fmt_pct(p$LR), fmt_pct(p$RF), fmt_pct(p$NB),
                fmt_pct(p$KNN), fmt_pct(p$CART), fmt_pct(p$XGB),
                fmt_pct(mean(unlist(p), na.rm = TRUE))
            ),
            check.names = FALSE
        )
    }, striped = TRUE, hover = TRUE, width = "100%", align = "lr")

    output$target_banner <- renderUI({
        p <- probs()
        avg_prob <- mean(unlist(p), na.rm = TRUE)
        if (is.na(avg_prob)) return(NULL)

        predicts_win <- avg_prob >= 0.5
        label <- if (predicts_win) "WIN" else "LOSS"
        bg <- if (predicts_win) "#eaf7ef" else "#fdecea"
        border <- if (predicts_win) "#27ae60" else "#c0392b"
        text <- if (predicts_win) "#1e7e46" else "#96281b"

        div(
            style = paste(
                "border-left: 8px solid", border, ";",
                "background:", bg, ";",
                "padding: 16px 18px;",
                "margin: 10px 0 18px 0;",
                "border-radius: 6px;"
            ),
            div(
                style = "font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #555;",
                "Model target"
            ),
            div(
                style = paste(
                    "font-size: 34px;",
                    "font-weight: 800;",
                    "line-height: 1.1;",
                    "color:", text, ";"
                ),
                label
            ),
            div(
                style = "font-size: 16px; color: #333; margin-top: 4px;",
                sprintf("Average predicted win probability: %.1f%%", avg_prob * 100)
            )
        )
    })

    output$prob_plot <- renderPlot({
        p <- probs()
        df <- data.frame(
            model = c("LR", "RF", "NB", "KNN", "CART", "XGB"),
            prob  = c(p$LR, p$RF, p$NB, p$KNN, p$CART, p$XGB)
        )
        df <- df[order(df$prob), ]
        df$model <- factor(df$model, levels = df$model)

        ggplot(df, aes(x = model, y = prob)) +
            geom_col(width = 0.6) +
            geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
            geom_text(aes(label = sprintf("%.0f%%", prob * 100)), hjust = -0.1, size = 4) +
            coord_flip() +
            scale_y_continuous(labels = scales::percent, limits = c(0, 1.05)) +
            labs(x = NULL, y = "Predicted win probability") +
            theme_grey(base_size = 13)
    }, res = 96)

    # Hyperparameter Tuning
    observeEvent(input$tune_reset, {
        m <- input$tune_model
        if (m == "RF")   updateSliderInput(session,  "rf_mtry",  value = rf_model$bestTune$mtry)
        if (m == "KNN")  updateSliderInput(session,  "knn_k",    value = knn_model$bestTune$k)
        if (m == "CART") updateTextInput(session, "cart_cp",
                                         value = format(cart_model$bestTune$cp, scientific = FALSE))
        if (m == "NB") {
            updateRadioButtons(session,  "nb_kernel", selected = as.character(nb_model$bestTune$usekernel))
            updateNumericInput(session,  "nb_adjust", value    = nb_model$bestTune$adjust)
        }
        if (m == "XGB") {
            updateNumericInput(session, "xgb_nrounds",   value = xgb_model$bestTune$nrounds)
            updateNumericInput(session, "xgb_max_depth", value = xgb_model$bestTune$max_depth)
            updateNumericInput(session, "xgb_eta",       value = xgb_model$bestTune$eta)
        }
    })

    output$tune_description <- renderUI({
        descs <- list(
            LR   = "Standard GLM - no hyperparameter search. All features used, trained on standardized inputs. Coefficients are interpretable as log-odds weights.",
            RF   = "Ensemble of 500 decision trees. mtry controls how many features are randomly considered at each split.",
            NB   = "Probabilistic classifier. Tunes density estimation (Gaussian vs. kernel) and bandwidth multiplier (adjust).",
            KNN  = "Classifies by majority vote among the k nearest neighbors in scaled feature space.",
            CART = "Recursive binary splitting tree. cp penalizes tree growth - small cp allows deep trees, large cp forces early stopping.",
            XGB  = "Gradient boosting. Tunes the number of boosting rounds (nrounds), tree depth (max_depth), and learning rate (eta)."
        )
        p(descs[[input$tune_model]])
    })

    output$tune_best <- renderUI({
        m <- input$tune_model
        if (m == "LR") return(p(em("Standard GLM - no hyperparameter search performed.")))
        if (m == "RF")   return(p(strong("mtry: "),  rf_model$bestTune$mtry))
        if (m == "KNN")  return(p(strong("k: "),     knn_model$bestTune$k))
        if (m == "CART") return(p(strong("cp: "),    formatC(cart_model$bestTune$cp, format = "g")))
        if (m == "NB")   return(tagList(
            p(strong("Density: "), if (nb_model$bestTune$usekernel) "Kernel density" else "Gaussian"),
            p(strong("Adjust: "),  nb_model$bestTune$adjust)
        ))
        if (m == "XGB")  return(tagList(
            p(strong("nrounds: "),   xgb_model$bestTune$nrounds),
            p(strong("max_depth: "), xgb_model$bestTune$max_depth),
            p(strong("eta: "),       xgb_model$bestTune$eta)
        ))
    })

    output$tune_slider <- renderUI({
        m <- input$tune_model

        reset_btn <- actionButton("tune_reset", "Reset to best")

        if (m == "LR") {
            return(p(em("Standard GLM - no hyperparameter to tune. The plot below shows odds ratios for the top features.")))
        }
        if (m == "RF") {
            tested <- rf_model$results$mtry
            return(tagList(
                sliderInput("rf_mtry", "mtry - features per split",
                    min = 1, max = ncol(rf_model$trainingData) - 1,
                    value = rf_model$bestTune$mtry, step = 1, width = "100%"),
                helpText(paste("Pre-tested grid:", paste(tested, collapse = ", "),
                               "- slider can pick any value 1 to",
                               ncol(rf_model$trainingData) - 1)),
                div(style = "display:flex; gap:8px; margin-top:6px;",
                    reset_btn,
                    actionButton("rf_recompute", "Recompute live (5-fold CV)",
                                 class = "btn-primary")),
                br(),
                uiOutput("rf_live_status")
            ))
        }
        if (m == "KNN") {
            tested <- knn_model$results$k
            knn_max <- max(100L, max(tested) * 2L)
            return(tagList(
                sliderInput("knn_k", "k - number of neighbors",
                    min = 1, max = knn_max,
                    value = knn_model$bestTune$k, step = 1, width = "100%"),
                helpText(paste("Pre-tested grid:", paste(tested, collapse = ", "),
                               "- slider can pick any value 1 to", knn_max)),
                div(style = "display:flex; gap:8px; margin-top:6px;",
                    reset_btn,
                    actionButton("knn_recompute", "Recompute live (5-fold CV)",
                                 class = "btn-primary")),
                br(),
                uiOutput("knn_live_status")
            ))
        }
        if (m == "CART") {
            tested <- cart_model$results$cp
            return(tagList(
                textInput("cart_cp", "cp - complexity parameter",
                    value = format(cart_model$bestTune$cp, scientific = FALSE),
                    width = "100%"),
                helpText(paste0("Pre-tested grid: ",
                                paste(format(tested, scientific = FALSE), collapse = ", "),
                                "  -  any positive value is allowed (smaller = deeper tree). Scientific notation like 1e-4 also works.")),
                div(style = "display:flex; gap:8px; margin-top:6px;",
                    reset_btn,
                    actionButton("cart_recompute", "Recompute live (5-fold CV)",
                                 class = "btn-primary")),
                br(),
                uiOutput("cart_live_status")
            ))
        }
        if (m == "NB") {
            tested_adj <- sort(unique(nb_model$results$adjust))
            return(tagList(
                radioButtons("nb_kernel", "Density estimation",
                    choices = c("Gaussian (parametric)" = "FALSE",
                                "Kernel density (non-parametric)" = "TRUE"),
                    selected = as.character(nb_model$bestTune$usekernel)),
                numericInput("nb_adjust", "Bandwidth adjust",
                    value = nb_model$bestTune$adjust,
                    min = 0.1, max = 5, step = 0.1, width = "100%"),
                helpText(paste0("Pre-tested grid: ",
                                paste(tested_adj, collapse = ", "),
                                "  -  any value in [0.1, 5] is allowed")),
                div(style = "display:flex; gap:8px; margin-top:6px;",
                    reset_btn,
                    actionButton("nb_recompute", "Recompute live (5-fold CV)",
                                 class = "btn-primary")),
                br(),
                uiOutput("nb_live_status")
            ))
        }
        if (m == "XGB") {
            tested_n   <- sort(unique(xgb_model$results$nrounds))
            tested_d   <- sort(unique(xgb_model$results$max_depth))
            tested_eta <- sort(unique(xgb_model$results$eta))
            return(tagList(
                numericInput("xgb_nrounds", "nrounds - boosting iterations",
                    value = xgb_model$bestTune$nrounds,
                    min = 1, max = 5000, step = 1, width = "100%"),
                helpText(paste("Pre-tested:", paste(tested_n, collapse = ", "))),
                numericInput("xgb_max_depth", "max_depth - tree depth",
                    value = xgb_model$bestTune$max_depth,
                    min = 1, max = 30, step = 1, width = "100%"),
                helpText(paste("Pre-tested:", paste(tested_d, collapse = ", "))),
                numericInput("xgb_eta", "eta - learning rate",
                    value = xgb_model$bestTune$eta,
                    min = 0.001, max = 1, step = 0.01, width = "100%"),
                helpText(paste("Pre-tested:", paste(tested_eta, collapse = ", "),
                               "- lower eta = slower but smoother learning")),
                div(style = "display:flex; gap:8px; margin-top:6px;",
                    reset_btn,
                    actionButton("xgb_recompute", "Recompute live (5-fold CV)",
                                 class = "btn-primary")),
                br(),
                uiOutput("xgb_live_status")
            ))
        }
    })

    # RF live retraining
    rf_live <- reactiveVal(tibble::tibble(
        mtry = integer(), ROC = double(), Sens = double(),
        Spec = double(), elapsed = double(), predictions = list()
    ))

    observeEvent(input$rf_recompute, {
        val <- isolate(input$rf_mtry)
        max_mtry <- ncol(rf_model$trainingData) - 1
        if (is.null(val) || val < 1 || val > max_mtry) return()

        existing <- rf_live()
        if (any(existing$mtry == val)) {
            showNotification(sprintf("mtry = %d already computed live (%.2f%% AUC)",
                                     val, existing$ROC[existing$mtry == val][1] * 100),
                             duration = 3, type = "message")
            return()
        }

        td <- rf_model$trainingData
        x  <- td[, setdiff(names(td), ".outcome"), drop = FALSE]
        y  <- td$.outcome

        withProgress(
            message = sprintf("Live RF  -  mtry = %d  -  5-fold CV", val),
            detail  = sprintf("training on %d workers (%d folds x %d tree-chunks)...",
                              n_cores, N_FOLDS, chunks_per_fold),
            value   = 0.1, {
                t0 <- Sys.time()
                res <- tryCatch(
                    recompute_rf_parallel(x = x, y = y, mtry_val = val),
                    error = function(e) e
                )
                if (inherits(res, "error")) {
                    showNotification(paste("Training failed:", res$message),
                                     duration = 6, type = "error")
                    return()
                }
                elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
                new_row <- tibble::tibble(
                    mtry        = val,
                    ROC         = res$ROC,
                    Sens        = res$Sens,
                    Spec        = res$Spec,
                    elapsed     = elapsed,
                    predictions = list(res$predictions)
                )
                rf_live(dplyr::bind_rows(existing, new_row))
                incProgress(0.9, detail = sprintf("done in %.1f s", elapsed))
            })
    })

    output$rf_live_status <- renderUI({
        live <- rf_live()
        if (nrow(live) == 0) {
            return(helpText(sprintf(
                "No live retrain yet. Click \"Recompute\" to run a fresh 5-fold CV (parallel on %d workers  -  %d folds x %d tree-chunks).",
                n_cores, N_FOLDS, chunks_per_fold)))
        }
        last <- live[nrow(live), ]
        tagList(
            helpText(sprintf(
                "Live points: %d   -   Last: mtry=%d, AUC=%.2f%%, %.1fs (%d workers)",
                nrow(live), last$mtry, last$ROC * 100, last$elapsed, n_cores))
        )
    })

    # KNN live retrain
    knn_live <- reactiveVal(tibble::tibble(
        k = integer(), ROC = double(), Sens = double(),
        Spec = double(), elapsed = double(), predictions = list()
    ))

    extract_caret_preds <- function(m) {
        p <- m$pred
        if (is.null(p)) return(NULL)
        data.frame(prob_win = p$Win,
                   actual   = factor(p$obs, levels = c("Loss", "Win")))
    }

    observeEvent(input$knn_recompute, {
        val <- isolate(input$knn_k)
        if (is.null(val)) return()
        existing <- knn_live()
        if (any(existing$k == val)) {
            showNotification(sprintf("k = %d already computed live (%.2f%% AUC)",
                                     val, existing$ROC[existing$k == val][1] * 100),
                             duration = 3, type = "message")
            return()
        }
        xy <- get_xy(knn_model)
        withProgress(
            message = sprintf("Live KNN  -  k = %d  -  5-fold CV", val),
            detail  = sprintf("training on %d workers...", min(n_cores, N_FOLDS)),
            value   = 0.1, {
                t0 <- Sys.time()
                m <- tryCatch(recompute_caret("knn", xy$x, xy$y, data.frame(k = val)),
                              error = function(e) e)
                if (inherits(m, "error")) {
                    showNotification(paste("Training failed:", m$message),
                                     duration = 6, type = "error")
                    return()
                }
                elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
                new_row <- tibble::tibble(
                    k = val, ROC = m$results$ROC, Sens = m$results$Sens,
                    Spec = m$results$Spec, elapsed = elapsed,
                    predictions = list(extract_caret_preds(m))
                )
                knn_live(dplyr::bind_rows(existing, new_row))
                incProgress(0.9, detail = sprintf("done in %.1f s", elapsed))
            })
    })

    output$knn_live_status <- renderUI({
        live <- knn_live()
        if (nrow(live) == 0) {
            return(helpText(sprintf(
                "No live retrain yet. Click \"Recompute\" to run a fresh 5-fold CV (parallel on up to %d workers).",
                min(n_cores, N_FOLDS))))
        }
        last <- live[nrow(live), ]
        helpText(sprintf("Live points: %d   -   Last: k=%d, AUC=%.2f%%, %.1fs",
                         nrow(live), last$k, last$ROC * 100, last$elapsed))
    })

    # CART live retrain
    cart_live <- reactiveVal(tibble::tibble(
        cp = double(), ROC = double(), Sens = double(),
        Spec = double(), elapsed = double(), predictions = list()
    ))

    observeEvent(input$cart_recompute, {
        val <- as.numeric(isolate(input$cart_cp))
        if (is.null(val) || is.na(val)) return()
        existing <- cart_live()
        if (any(abs(existing$cp - val) < 1e-12)) {
            showNotification(sprintf("cp = %s already computed live",
                                     formatC(val, format = "g")),
                             duration = 3, type = "message")
            return()
        }
        xy <- get_xy(cart_model)
        withProgress(
            message = sprintf("Live CART  -  cp = %s  -  5-fold CV",
                              formatC(val, format = "g")),
            detail  = sprintf("training on %d workers...", min(n_cores, N_FOLDS)),
            value   = 0.1, {
                t0 <- Sys.time()
                m <- tryCatch(recompute_caret("rpart", xy$x, xy$y,
                                              data.frame(cp = val)),
                              error = function(e) e)
                if (inherits(m, "error")) {
                    showNotification(paste("Training failed:", m$message),
                                     duration = 6, type = "error")
                    return()
                }
                elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
                new_row <- tibble::tibble(
                    cp = val, ROC = m$results$ROC, Sens = m$results$Sens,
                    Spec = m$results$Spec, elapsed = elapsed,
                    predictions = list(extract_caret_preds(m))
                )
                cart_live(dplyr::bind_rows(existing, new_row))
                incProgress(0.9, detail = sprintf("done in %.1f s", elapsed))
            })
    })

    output$cart_live_status <- renderUI({
        live <- cart_live()
        if (nrow(live) == 0) {
            return(helpText(sprintf(
                "No live retrain yet. Click \"Recompute\" to run a fresh 5-fold CV (parallel on up to %d workers).",
                min(n_cores, N_FOLDS))))
        }
        last <- live[nrow(live), ]
        helpText(sprintf("Live points: %d   -   Last: cp=%s, AUC=%.2f%%, %.1fs",
                         nrow(live), formatC(last$cp, format = "g"),
                         last$ROC * 100, last$elapsed))
    })

    # NB live retrain
    nb_live <- reactiveVal(tibble::tibble(
        usekernel = logical(), adjust = double(), ROC = double(),
        Sens = double(), Spec = double(), elapsed = double(), predictions = list()
    ))

    observeEvent(input$nb_recompute, {
        uk  <- as.logical(isolate(input$nb_kernel))
        adj <- as.numeric(isolate(input$nb_adjust))
        if (is.null(uk) || is.null(adj) || is.na(uk) || is.na(adj)) return()
        existing <- nb_live()
        match_idx <- which(existing$usekernel == uk &
                           abs(existing$adjust - adj) < 1e-9)
        if (length(match_idx) > 0) {
            showNotification(sprintf("(%s, adjust=%g) already computed live",
                                     if (uk) "Kernel" else "Gaussian", adj),
                             duration = 3, type = "message")
            return()
        }
        xy <- get_xy(nb_model)
        withProgress(
            message = sprintf("Live NB  -  %s, adjust=%g  -  5-fold CV",
                              if (uk) "Kernel" else "Gaussian", adj),
            detail  = sprintf("training on %d workers...", min(n_cores, N_FOLDS)),
            value   = 0.1, {
                t0 <- Sys.time()
                m <- tryCatch(recompute_caret(
                    "nb", xy$x, xy$y,
                    data.frame(usekernel = uk, fL = 0, adjust = adj)
                ), error = function(e) e)
                if (inherits(m, "error")) {
                    showNotification(paste("Training failed:", m$message),
                                     duration = 6, type = "error")
                    return()
                }
                elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
                new_row <- tibble::tibble(
                    usekernel = uk, adjust = adj,
                    ROC = m$results$ROC, Sens = m$results$Sens,
                    Spec = m$results$Spec, elapsed = elapsed,
                    predictions = list(extract_caret_preds(m))
                )
                nb_live(dplyr::bind_rows(existing, new_row))
                incProgress(0.9, detail = sprintf("done in %.1f s", elapsed))
            })
    })

    output$nb_live_status <- renderUI({
        live <- nb_live()
        if (nrow(live) == 0) {
            return(helpText(sprintf(
                "No live retrain yet. Click \"Recompute\" to run a fresh 5-fold CV (parallel on up to %d workers).",
                min(n_cores, N_FOLDS))))
        }
        last <- live[nrow(live), ]
        helpText(sprintf("Live points: %d   -   Last: %s, adjust=%g, AUC=%.2f%%, %.1fs",
                         nrow(live), if (last$usekernel) "Kernel" else "Gaussian",
                         last$adjust, last$ROC * 100, last$elapsed))
    })

    # XGB live retrain
    xgb_live <- reactiveVal(tibble::tibble(
        nrounds = integer(), max_depth = integer(), eta = double(),
        ROC = double(), Sens = double(), Spec = double(),
        elapsed = double(), predictions = list()
    ))

    observeEvent(input$xgb_recompute, {
        nr <- as.integer(isolate(input$xgb_nrounds))
        md <- as.integer(isolate(input$xgb_max_depth))
        et <- as.numeric(isolate(input$xgb_eta))
        if (anyNA(c(nr, md, et))) return()
        existing <- xgb_live()
        if (any(existing$nrounds == nr & existing$max_depth == md &
                abs(existing$eta - et) < 1e-9)) {
            showNotification(sprintf("(nrounds=%d, max_depth=%d, eta=%g) already computed",
                                     nr, md, et),
                             duration = 3, type = "message")
            return()
        }
        xy <- get_xy(xgb_model)
        withProgress(
            message = sprintf("Live XGB  -  nrounds=%d, depth=%d, eta=%g  -  5-fold CV",
                              nr, md, et),
            detail  = sprintf("training on %d workers...", min(n_cores, N_FOLDS)),
            value   = 0.1, {
                t0 <- Sys.time()
                m <- tryCatch(recompute_caret(
                    "xgbTree", xy$x, xy$y,
                    data.frame(nrounds = nr, max_depth = md, eta = et,
                               gamma = 0, colsample_bytree = 1,
                               min_child_weight = 1, subsample = 1)
                ), error = function(e) e)
                if (inherits(m, "error")) {
                    showNotification(paste("Training failed:", m$message),
                                     duration = 6, type = "error")
                    return()
                }
                elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
                new_row <- tibble::tibble(
                    nrounds = nr, max_depth = md, eta = et,
                    ROC = m$results$ROC, Sens = m$results$Sens,
                    Spec = m$results$Spec, elapsed = elapsed,
                    predictions = list(extract_caret_preds(m))
                )
                xgb_live(dplyr::bind_rows(existing, new_row))
                incProgress(0.9, detail = sprintf("done in %.1f s", elapsed))
            })
    })

    output$xgb_live_status <- renderUI({
        live <- xgb_live()
        if (nrow(live) == 0) {
            return(helpText(sprintf(
                "No live retrain yet. Click \"Recompute\" to run a fresh 5-fold CV (parallel on up to %d workers).",
                min(n_cores, N_FOLDS))))
        }
        last <- live[nrow(live), ]
        helpText(sprintf(
            "Live points: %d   -   Last: nrounds=%d, depth=%d, eta=%g, AUC=%.2f%%, %.1fs",
            nrow(live), last$nrounds, last$max_depth, last$eta,
            last$ROC * 100, last$elapsed))
    })

    selected_row <- reactive({
        m <- input$tune_model
        if (m == "LR") return(lr_model$results[1, ])
        if (m == "RF") {
            val <- if (is.null(input$rf_mtry)) rf_model$bestTune$mtry else input$rf_mtry
            live <- rf_live()
            if (nrow(live) > 0 && any(live$mtry == val)) {
                lr <- live[live$mtry == val, ][1, ]
                return(data.frame(mtry = lr$mtry, ROC = lr$ROC, Sens = lr$Sens, Spec = lr$Spec))
            }
            sv  <- snap_to(val, rf_model$results$mtry)
            return(rf_model$results %>% filter(mtry == sv))
        }
        if (m == "KNN") {
            val <- if (is.null(input$knn_k)) knn_model$bestTune$k else input$knn_k
            live <- knn_live()
            if (nrow(live) > 0 && any(live$k == val)) {
                lr <- live[live$k == val, ][1, ]
                return(data.frame(k = lr$k, ROC = lr$ROC, Sens = lr$Sens, Spec = lr$Spec))
            }
            return(knn_model$results %>% filter(k == val))
        }
        if (m == "CART") {
            raw <- input$cart_cp
            val <- if (is.null(raw) || is.na(raw)) cart_model$bestTune$cp else as.numeric(raw)
            live <- cart_live()
            if (nrow(live) > 0 && any(abs(live$cp - val) < 1e-12)) {
                lr <- live[abs(live$cp - val) < 1e-12, ][1, ]
                return(data.frame(cp = lr$cp, ROC = lr$ROC, Sens = lr$Sens, Spec = lr$Spec))
            }
            return(cart_model$results %>% filter(abs(cp - val) < 1e-10))
        }
        if (m == "NB") {
            uk  <- as.logical(if (is.null(input$nb_kernel)) as.character(nb_model$bestTune$usekernel) else input$nb_kernel)
            raw <- input$nb_adjust
            adj <- if (is.null(raw) || is.na(raw)) nb_model$bestTune$adjust else as.numeric(raw)
            live <- nb_live()
            if (nrow(live) > 0) {
                idx <- which(live$usekernel == uk & abs(live$adjust - adj) < 1e-9)
                if (length(idx) > 0) {
                    lr <- live[idx[1], ]
                    return(data.frame(usekernel = lr$usekernel, adjust = lr$adjust,
                                      ROC = lr$ROC, Sens = lr$Sens, Spec = lr$Spec))
                }
            }
            return(nb_model$results %>% filter(usekernel == uk, abs(adjust - adj) < 0.01))
        }
        if (m == "XGB") {
            r_nr <- input$xgb_nrounds
            r_md <- input$xgb_max_depth
            r_et <- input$xgb_eta
            nr <- as.integer(if (is.null(r_nr) || is.na(r_nr)) xgb_model$bestTune$nrounds   else r_nr)
            md <- as.integer(if (is.null(r_md) || is.na(r_md)) xgb_model$bestTune$max_depth else r_md)
            et <- as.numeric(if (is.null(r_et) || is.na(r_et)) xgb_model$bestTune$eta       else r_et)
            live <- xgb_live()
            if (nrow(live) > 0) {
                idx <- which(live$nrounds == nr & live$max_depth == md & abs(live$eta - et) < 1e-9)
                if (length(idx) > 0) {
                    lr <- live[idx[1], ]
                    return(data.frame(nrounds = lr$nrounds, max_depth = lr$max_depth, eta = lr$eta,
                                      ROC = lr$ROC, Sens = lr$Sens, Spec = lr$Spec))
                }
            }
            return(xgb_model$results %>% filter(nrounds == nr, max_depth == md, abs(eta - et) < 1e-9))
        }
    })

    # Predictions for the selected (model, config) - drives all diagnostics.
    # Priority: live retrain > saved test-set predictions (when at bestTune) > saved OOF CV (for off-best configs).
    selected_predictions <- reactive({
        m <- input$tune_model

        rf_val   <- if (is.null(input$rf_mtry)) rf_model$bestTune$mtry else input$rf_mtry
        knn_val  <- if (is.null(input$knn_k))   knn_model$bestTune$k   else input$knn_k
        cart_raw <- input$cart_cp
        cart_val <- if (is.null(cart_raw) || cart_raw == "" ||
                        is.na(suppressWarnings(as.numeric(cart_raw))))
                        cart_model$bestTune$cp else as.numeric(cart_raw)
        nb_uk    <- as.logical(if (is.null(input$nb_kernel))
                               as.character(nb_model$bestTune$usekernel)
                               else input$nb_kernel)
        nb_adj_raw <- input$nb_adjust
        nb_adj   <- if (is.null(nb_adj_raw) || is.na(nb_adj_raw))
                        nb_model$bestTune$adjust else as.numeric(nb_adj_raw)
        xgb_nr   <- as.integer(if (is.null(input$xgb_nrounds)   || is.na(input$xgb_nrounds))
                               xgb_model$bestTune$nrounds   else input$xgb_nrounds)
        xgb_md   <- as.integer(if (is.null(input$xgb_max_depth) || is.na(input$xgb_max_depth))
                               xgb_model$bestTune$max_depth else input$xgb_max_depth)
        xgb_et   <- as.numeric(if (is.null(input$xgb_eta)       || is.na(input$xgb_eta))
                               xgb_model$bestTune$eta       else input$xgb_eta)

        # 1) Live retrain hit?
        if (m == "RF") {
            live <- rf_live()
            if (nrow(live) > 0 && any(live$mtry == rf_val))
                return(list(df = live$predictions[[which(live$mtry == rf_val)[1]]],
                            source = "live"))
        }
        if (m == "KNN") {
            live <- knn_live()
            if (nrow(live) > 0 && any(live$k == knn_val))
                return(list(df = live$predictions[[which(live$k == knn_val)[1]]],
                            source = "live"))
        }
        if (m == "CART") {
            live <- cart_live()
            if (nrow(live) > 0 && any(abs(live$cp - cart_val) < 1e-12))
                return(list(df = live$predictions[[which(abs(live$cp - cart_val) < 1e-12)[1]]],
                            source = "live"))
        }
        if (m == "NB") {
            live <- nb_live()
            if (nrow(live) > 0) {
                idx <- which(live$usekernel == nb_uk & abs(live$adjust - nb_adj) < 1e-9)
                if (length(idx) > 0)
                    return(list(df = live$predictions[[idx[1]]], source = "live"))
            }
        }
        if (m == "XGB") {
            live <- xgb_live()
            if (nrow(live) > 0) {
                idx <- which(live$nrounds == xgb_nr & live$max_depth == xgb_md &
                             abs(live$eta - xgb_et) < 1e-9)
                if (length(idx) > 0)
                    return(list(df = live$predictions[[idx[1]]], source = "live"))
            }
        }

        # 2) At the saved bestTune config? Use the held-out test-set predictions.
        is_best <- switch(m,
            LR   = TRUE,
            RF   = !is.na(rf_val)   && rf_val   == rf_model$bestTune$mtry,
            KNN  = !is.na(knn_val)  && knn_val  == knn_model$bestTune$k,
            CART = !is.na(cart_val) && abs(cart_val - cart_model$bestTune$cp) < 1e-12,
            NB   = !is.na(nb_uk) && !is.na(nb_adj) &&
                   nb_uk == nb_model$bestTune$usekernel &&
                   abs(nb_adj - nb_model$bestTune$adjust) < 1e-9,
            XGB  = !is.na(xgb_nr) && !is.na(xgb_md) && !is.na(xgb_et) &&
                   xgb_nr == xgb_model$bestTune$nrounds &&
                   xgb_md == xgb_model$bestTune$max_depth &&
                   abs(xgb_et - xgb_model$bestTune$eta) < 1e-9,
            FALSE
        )
        if (is_best) return(list(df = test_preds[[m]], source = "test"))

        # 3) Off-best pre-tested config - fall back to saved OOF CV predictions.
        params <- switch(m,
            RF   = list(mtry = rf_val),
            KNN  = list(k    = knn_val),
            CART = list(cp   = cart_val),
            NB   = list(usekernel = nb_uk, adjust = nb_adj),
            XGB  = list(nrounds = xgb_nr, max_depth = xgb_md, eta = xgb_et),
            list()
        )
        pred_obj <- switch(m,
            LR = lr_model$pred, RF = rf_model$pred, KNN = knn_model$pred,
            CART = cart_model$pred, NB = nb_model$pred, XGB = xgb_model$pred
        )
        list(df = filter_pred(pred_obj, params), source = "oof")
    })

    output$diag_source_note <- renderUI({
        sp <- selected_predictions()
        if (is.null(sp$df) || nrow(sp$df) == 0) {
            return(helpText(em(paste0(
                "No predictions available for this configuration yet - ",
                "click \"Recompute live\" to retrain and generate predictions."))))
        }
        n <- nrow(sp$df)
        msg <- switch(sp$source,
            test = sprintf("Test-set evaluation  -  %d held-out predictions (saved bestTune model on the 20%% holdout)", n),
            live = sprintf("Live 5-fold CV recompute  -  %d out-of-fold predictions on the training set", n),
            oof  = sprintf("Saved 5-fold OOF CV  -  %d training-row predictions (each row was held out once during CV)", n),
            ""
        )
        helpText(msg)
    })

    output$metrics_table <- renderTable({
        df <- selected_predictions()$df
        compute_all_metrics(df)
    }, striped = TRUE, hover = TRUE, width = "60%", align = "lr")

    output$diag_roc <- renderPlot(plot_roc(selected_predictions()$df), res = 96)

    output$diag_specific <- renderPlot({
        m  <- input$tune_model
        df <- selected_predictions()$df
        if (m == "LR")   return(plot_varimp_lr(lr_model))
        if (m == "RF")   return(plot_varimp(rf_model,  "Feature importance (RF, scaled)"))
        if (m == "XGB")  return(plot_varimp(xgb_model, "Feature importance (XGB gain, scaled)"))
        if (m == "CART") return(plot_varimp(cart_model, "Feature importance (CART, scaled)"))
        if (m == "KNN")  return(plot_knn_elbow(knn_model, knn_live(), input$knn_k))
        if (m == "NB")   return(plot_varimp(nb_model, "Feature importance (NB, scaled)"))
        ggplot() + theme_void()
    }, res = 96)

    output$tune_table <- renderTable({
        m  <- input$tune_model
        sr <- selected_row()
        if (m == "LR") return(NULL)

        has_sel <- !is.null(sr) && nrow(sr) > 0
        get_sel <- function(field, default = NA) if (has_sel) sr[[field]][1] else default

        if (m == "RF") {
            sel_v <- get_sel("mtry", NA_integer_)
            base <- rf_model$results %>%
                mutate(Source   = "Pre-tested",
                       Selected = ifelse(!is.na(sel_v) & mtry == sel_v, "▶", ""),
                       Best     = ifelse(mtry == rf_model$bestTune$mtry, "★", "")) %>%
                transmute(mtry,
                          `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                          Sensitivity  = sprintf("%.2f%%", Sens * 100),
                          Specificity  = sprintf("%.2f%%", Spec * 100),
                          Source, Selected, Best)
            live_now <- rf_live()
            if (nrow(live_now) > 0) {
                live_tbl <- live_now %>%
                    arrange(mtry) %>%
                    mutate(Source   = "Live",
                           Selected = ifelse(!is.na(sel_v) & mtry == sel_v, "▶", ""),
                           Best     = "") %>%
                    transmute(mtry,
                              `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                              Sensitivity  = sprintf("%.2f%%", Sens * 100),
                              Specificity  = sprintf("%.2f%%", Spec * 100),
                              Source, Selected, Best)
                base <- dplyr::bind_rows(base, live_tbl)
            }
            return(base)
        }

        if (m == "KNN") {
            sel_v <- get_sel("k", NA_integer_)
            base <- knn_model$results %>%
                mutate(Source   = "Pre-tested",
                       Selected = ifelse(!is.na(sel_v) & k == sel_v, "▶", ""),
                       Best     = ifelse(k == knn_model$bestTune$k, "★", "")) %>%
                transmute(k,
                          `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                          Sensitivity  = sprintf("%.2f%%", Sens * 100),
                          Specificity  = sprintf("%.2f%%", Spec * 100),
                          Source, Selected, Best)
            live_now <- knn_live()
            if (nrow(live_now) > 0) {
                live_tbl <- live_now %>%
                    arrange(k) %>%
                    mutate(Source   = "Live",
                           Selected = ifelse(!is.na(sel_v) & k == sel_v, "▶", ""),
                           Best     = "") %>%
                    transmute(k,
                              `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                              Sensitivity  = sprintf("%.2f%%", Sens * 100),
                              Specificity  = sprintf("%.2f%%", Spec * 100),
                              Source, Selected, Best)
                base <- dplyr::bind_rows(base, live_tbl)
            }
            return(base)
        }

        if (m == "CART") {
            sel_v <- get_sel("cp", NA_real_)
            fmt_cp <- function(x) trimws(format(x, scientific = FALSE, drop0trailing = TRUE))
            base <- cart_model$results %>%
                mutate(Source   = "Pre-tested",
                       Selected = ifelse(!is.na(sel_v) & abs(cp - sel_v) < 1e-10, "▶", ""),
                       Best     = ifelse(cp == cart_model$bestTune$cp, "★", "")) %>%
                transmute(cp = fmt_cp(cp),
                          `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                          Sensitivity  = sprintf("%.2f%%", Sens * 100),
                          Specificity  = sprintf("%.2f%%", Spec * 100),
                          Source, Selected, Best)
            live_now <- cart_live()
            if (nrow(live_now) > 0) {
                live_tbl <- live_now %>%
                    arrange(cp) %>%
                    mutate(Source   = "Live",
                           Selected = ifelse(!is.na(sel_v) & abs(cp - sel_v) < 1e-10, "▶", ""),
                           Best     = "") %>%
                    transmute(cp = fmt_cp(cp),
                              `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                              Sensitivity  = sprintf("%.2f%%", Sens * 100),
                              Specificity  = sprintf("%.2f%%", Spec * 100),
                              Source, Selected, Best)
                base <- dplyr::bind_rows(base, live_tbl)
            }
            return(base)
        }

        if (m == "NB") {
            sel_uk  <- get_sel("usekernel", NA)
            sel_adj <- get_sel("adjust", NA_real_)
            base <- nb_model$results %>%
                mutate(
                    Density  = ifelse(usekernel, "Kernel", "Gaussian"),
                    Source   = "Pre-tested",
                    Selected = ifelse(!is.na(sel_uk) & !is.na(sel_adj) &
                                      usekernel == sel_uk & abs(adjust - sel_adj) < 0.01, "▶", ""),
                    Best     = ifelse(usekernel == nb_model$bestTune$usekernel &
                                      abs(adjust - nb_model$bestTune$adjust) < 0.01, "★", "")
                ) %>%
                transmute(Density, Adjust = adjust,
                          `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                          Sensitivity  = sprintf("%.2f%%", Sens * 100),
                          Specificity  = sprintf("%.2f%%", Spec * 100),
                          Source, Selected, Best)
            live_now <- nb_live()
            if (nrow(live_now) > 0) {
                live_tbl <- live_now %>%
                    arrange(usekernel, adjust) %>%
                    mutate(
                        Density  = ifelse(usekernel, "Kernel", "Gaussian"),
                        Source   = "Live",
                        Selected = ifelse(!is.na(sel_uk) & !is.na(sel_adj) &
                                          usekernel == sel_uk & abs(adjust - sel_adj) < 1e-9, "▶", ""),
                        Best     = ""
                    ) %>%
                    transmute(Density, Adjust = adjust,
                              `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                              Sensitivity  = sprintf("%.2f%%", Sens * 100),
                              Specificity  = sprintf("%.2f%%", Spec * 100),
                              Source, Selected, Best)
                base <- dplyr::bind_rows(base, live_tbl)
            }
            return(base)
        }

        if (m == "XGB") {
            sel_nr <- get_sel("nrounds",   NA_integer_)
            sel_md <- get_sel("max_depth", NA_integer_)
            sel_et <- get_sel("eta",       NA_real_)
            sel_ok <- !is.na(sel_nr) & !is.na(sel_md) & !is.na(sel_et)
            base <- xgb_model$results %>%
                mutate(
                    Source   = "Pre-tested",
                    Selected = ifelse(sel_ok & nrounds == sel_nr & max_depth == sel_md &
                                      abs(eta - sel_et) < 1e-9, "▶", ""),
                    Best     = ifelse(nrounds == xgb_model$bestTune$nrounds &
                                      max_depth == xgb_model$bestTune$max_depth &
                                      abs(eta - xgb_model$bestTune$eta) < 1e-9, "★", "")
                ) %>%
                arrange(nrounds, max_depth, eta) %>%
                transmute(nrounds, max_depth, eta,
                          `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                          Sensitivity  = sprintf("%.2f%%", Sens * 100),
                          Specificity  = sprintf("%.2f%%", Spec * 100),
                          Source, Selected, Best)
            live_now <- xgb_live()
            if (nrow(live_now) > 0) {
                live_tbl <- live_now %>%
                    arrange(nrounds, max_depth, eta) %>%
                    mutate(
                        Source   = "Live",
                        Selected = ifelse(sel_ok & nrounds == sel_nr & max_depth == sel_md &
                                          abs(eta - sel_et) < 1e-9, "▶", ""),
                        Best     = ""
                    ) %>%
                    transmute(nrounds, max_depth, eta,
                              `CV AUC-ROC` = sprintf("%.2f%%", ROC * 100),
                              Sensitivity  = sprintf("%.2f%%", Sens * 100),
                              Specificity  = sprintf("%.2f%%", Spec * 100),
                              Source, Selected, Best)
                base <- dplyr::bind_rows(base, live_tbl)
            }
            return(base)
        }

        NULL
    }, striped = TRUE, hover = TRUE, width = "100%")

    # Model Comparison  -  Section 1 (method comparison)
    cmp_named_dfs <- reactive({
        keys <- input$cmp_models
        if (is.null(keys) || length(keys) == 0) return(list())
        out <- lapply(keys, function(k) test_preds[[k]])
        names(out) <- keys
        out
    })

    output$cmp_roc_overlay <- renderPlot({
        plot_roc_overlay(cmp_named_dfs())
    }, res = 96)

    output$cmp_metrics_table <- renderTable({
        tbl <- metrics_matrix(cmp_named_dfs())
        if (nrow(tbl) == 0) return(NULL)
        tbl$Family <- ifelse(grepl("Statistical", tbl$Family), "Statistical",
                     ifelse(grepl("Ensemble", tbl$Family), "Ensemble", "Partitioning"))
        tbl[order(tbl$Family, tbl$Model), ]
    }, striped = TRUE, hover = TRUE, width = "100%")

    output$cmp_topfeats <- renderPlot({
        plot_topfeats_panel(input$cmp_models, top_n = 5)
    }, res = 96)

    # Model Comparison  -  Section 2 (feature selection)
    fs_lambda <- reactive({
        if (is.null(input$fs_log_lambda)) return(lasso_default_lambda)
        10 ^ input$fs_log_lambda
    })

    observeEvent(input$fs_use_default, {
        m <- input$fs_method
        lam <- switch(m,
            lasso = lasso_default_lambda,
            elnet = elnet_default_lambda,
            NULL)
        if (!is.null(lam)) {
            updateSliderInput(session, "fs_log_lambda", value = log10(lam))
        }
    })

    fs_predictions <- reactive({
        m <- input$fs_method
        if (m == "none")    return(fs_full_preds)
        if (m == "forward") return(predict_fwd_test())
        fit <- switch(m, lasso = lasso_fit, elnet = elnet_fit)
        predict_glmnet_test(fit, fs_lambda())
    })

    fs_retained <- reactive({
        m <- input$fs_method
        if (m == "none") {
            co <- coef(lr_model$finalModel)
            co <- co[names(co) != "(Intercept)"]
            return(data.frame(feature = names(co), beta = unname(co)))
        }
        if (m == "forward") {
            co <- coef(fwd_model)
            co <- co[names(co) != "(Intercept)"]
            return(data.frame(feature = names(co), beta = unname(co)))
        }
        fit <- switch(m, lasso = lasso_fit, elnet = elnet_fit)
        retained_at(fit, fs_lambda())
    })

    output$fs_roc <- renderPlot({
        m <- input$fs_method
        if (m == "none") {
            plot_roc_overlay(list("Full LR (no FS)" = fs_full_preds))
        } else {
            label <- switch(m,
                forward = "Forward stepwise",
                lasso   = "LASSO",
                elnet   = sprintf("Elastic Net (%s)", enet_alpha_label))
            overlay <- list(fs_full_preds, fs_predictions())
            names(overlay) <- c("Full LR (no FS)", label)
            plot_roc_overlay(overlay)
        }
    }, res = 96)

    output$fs_retained <- renderPlot({
        m <- input$fs_method
        title <- switch(m,
            none    = "Full LR  -  all features kept",
            forward = "Forward stepwise  -  selected features",
            lasso   = "LASSO  -  retained features",
            elnet   = sprintf("Elastic Net (%s)  -  retained features", enet_alpha_label))
        plot_retained_bar(fs_retained(),
                          total_features = ncol(xs_train_mat),
                          title = title)
    }, res = 96)

    output$fs_path <- renderPlot({
        m <- input$fs_method
        fit <- switch(m, lasso = lasso_fit, elnet = elnet_fit, NULL)
        if (is.null(fit)) return(NULL)
        title <- switch(m,
            lasso = "LASSO coefficient path (α = 1)",
            elnet = sprintf("Elastic Net coefficient path (%s)", enet_alpha_label))
        plot_glmnet_path(fit, fs_lambda(), title)
    }, res = 96)

    output$fs_aic_path <- renderPlot({
        plot_fwd_aic_path(fwd_aic_path)
    }, res = 96)

    output$fs_lambda_summary <- renderUI({
        m <- input$fs_method
        if (!m %in% c("lasso", "elnet")) return(NULL)
        default_lam <- switch(m,
            lasso = lasso_default_lambda,
            elnet = elnet_default_lambda)
        helpText(sprintf("λ = %.5f  -  1-SE default for this method = %.5f",
                         fs_lambda(), default_lam))
    })

    # Data Explorer
    default_raw_rv <- reactiveVal(NULL)

    observeEvent(input$load_default_csv, {
        if (is.na(default_raw_path)) {
            showNotification("CSV not found at the default location.",
                             duration = 6, type = "error")
            return()
        }
        if (!is.null(default_raw_rv())) return()
        withProgress(message = "Loading CSV...", value = 0.5, {
            df <- tryCatch(load_default_raw_csv(default_raw_path),
                           error = function(e) {
                               showNotification(paste("CSV read failed:", e$message),
                                                duration = 8, type = "error")
                               NULL
                           })
            default_raw_rv(df)
        })
    })

    raw_data <- reactive({
        if (input$data_source == "upload") {
            req(input$data_file)
            df <- tryCatch(
                read.csv(input$data_file$datapath,
                         stringsAsFactors = FALSE,
                         check.names = FALSE,
                         na.strings = c("", "NA")),
                error = function(e) {
                    showNotification(paste("CSV read failed:", e$message),
                                     duration = 8, type = "error")
                    NULL
                }
            )
            return(df)
        }
        default_raw_rv()
    })

    eda_data <- reactive({
        prepare_eda_data(raw_data())
    })

    output$eda_status <- renderUI({
        eda <- eda_data()
        if (!is.null(eda$error)) {
            return(tags$p(tags$b(eda$error), style = "color:#c0392b"))
        }
        helpText("EDA is computed from complete Oracle's Elixir rows, pivoted to one row per game as in analysis.Qmd.")
    })

    output$eda_basic_metrics <- renderUI({
        eda <- eda_data()
        if (!is.null(eda$error)) return(em("No EDA dataset available."))
        games <- eda$games
        counts <- eda$counts
        gold_df <- games %>%
            filter(!is.na(blue_golddiffat15)) %>%
            transmute(pred_blue_win = blue_golddiffat15 > 0,
                      actual = blue_win == 1)
        side_test <- prop.test(sum(games$blue_win), nrow(games), p = 0.5)
        tags$ul(
            tags$li(tags$b("Raw rows: "), format(counts$raw, big.mark = ",")),
            tags$li(tags$b("Complete rows: "), format(counts$complete, big.mark = ",")),
            tags$li(tags$b("Team rows: "), format(counts$teams, big.mark = ",")),
            tags$li(tags$b("Player rows: "), format(counts$players, big.mark = ",")),
            tags$li(tags$b("Games: "), format(nrow(games), big.mark = ",")),
            tags$li(tags$b("Blue win rate: "), sprintf("%.2f%%", mean(games$blue_win) * 100)),
            tags$li(tags$b("Side-balance p-value: "), formatC(side_test$p.value, format = "e", digits = 2)),
            tags$li(tags$b("Gold @15 rule accuracy: "),
                    sprintf("%.2f%%", mean(gold_df$pred_blue_win == gold_df$actual) * 100))
        )
    })

    output$eda_target_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Target distribution  -  no data"))
        games <- eda$games %>%
            mutate(blue_win = factor(blue_win, levels = c(0, 1),
                                     labels = c("Red won", "Blue won")))
        ggplot(games, aes(x = blue_win, fill = blue_win)) +
            geom_bar() +
            geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
            scale_fill_manual(values = c("Red won" = "#e74c3c", "Blue won" = "#3498db")) +
            geom_hline(yintercept = nrow(games) / 2, linetype = "dashed", color = "gray40") +
            labs(title = "Target Variable Distribution (blue_win)",
                 x = "Outcome", y = "Count") +
            theme_minimal() +
            theme(legend.position = "none")
    }, res = 96)

    output$eda_descriptive_table <- DT::renderDataTable({
        eda <- eda_data()
        if (!is.null(eda$error)) return(NULL)
        DT::datatable(eda_numeric_summary(eda$games), rownames = FALSE,
            filter = "top",
            options = list(pageLength = 10, lengthMenu = c(10, 25, 50),
                           scrollX = TRUE),
            class = "stripe hover compact")
    })

    output$eda_correlation_table <- DT::renderDataTable({
        eda <- eda_data()
        if (!is.null(eda$error)) return(NULL)
        tbl <- eda_result_correlations(eda$games) %>%
            mutate(correlation = round(correlation, 3))
        DT::datatable(tbl, rownames = FALSE, filter = "top",
            options = list(pageLength = 12, lengthMenu = c(12, 25, 50),
                           scrollX = TRUE),
            class = "stripe hover compact")
    })

    output$eda_correlation_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Correlation matrix  -  no data"))
        cor_data <- eda$games %>% select(all_of(EDA_EARLY_COLS)) %>% drop_na()
        if (nrow(cor_data) < 3) return(empty_plot("Correlation matrix  -  not enough rows"))
        cor_mat <- cor(cor_data, use = "pairwise.complete.obs")
        corrplot::corrplot(cor_mat, method = "color", type = "upper",
                           tl.cex = 0.65, tl.col = "black", tl.srt = 45,
                           addCoef.col = "black", number.cex = 0.45,
                           col = colorRampPalette(c("#e74c3c", "white", "#2ecc71"))(200),
                           title = "Feature Correlation Matrix",
                           mar = c(0, 0, 2, 0))
    }, res = 96)

    output$eda_outcome_means_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Correlation with target  -  no data"))
        result_cors <- eda_result_correlations(eda$games)
        if (nrow(result_cors) == 0) return(empty_plot("Correlation with target  -  no complete rows"))
        ggplot(result_cors, aes(
            x = correlation,
            y = reorder(feature, abs(correlation)),
            fill = correlation > 0
        )) +
            geom_col() +
            geom_vline(xintercept = 0, color = "gray45") +
            scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.25)) +
            scale_fill_manual(values = c("TRUE" = "#2ecc71", "FALSE" = "#e74c3c"),
                              guide = "none") +
            labs(title = "Feature correlation with blue_win",
                 x = "Pearson correlation", y = NULL) +
            theme_minimal()
    }, res = 96)

    output$eda_outcome_means_table <- DT::renderDataTable({
        eda <- eda_data()
        if (!is.null(eda$error)) return(NULL)
        DT::datatable(eda_outcome_means(eda$games), rownames = FALSE,
            filter = "top",
            options = list(pageLength = 12, lengthMenu = c(12, 25, 50),
                           scrollX = TRUE),
            class = "stripe hover compact")
    })

    output$eda_gold_density_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Gold differential  -  no data"))
        eda$games %>%
            filter(!is.na(blue_golddiffat15)) %>%
            mutate(blue_win = factor(blue_win, levels = c(0, 1),
                                     labels = c("Red won", "Blue won"))) %>%
            ggplot(aes(x = blue_golddiffat15, fill = blue_win)) +
            geom_density(alpha = 0.5) +
            scale_fill_manual(values = c("Red won" = "#e74c3c", "Blue won" = "#3498db")) +
            geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
            labs(title = "Gold Differential at 15 Minutes",
                 x = "blue_golddiffat15", y = "Density", fill = "Outcome") +
            theme_minimal()
    }, res = 96)

    output$eda_gold_scatter_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Gold diff @10 vs @15  -  no data"))
        eda$games %>%
            filter(!is.na(blue_golddiffat10), !is.na(blue_golddiffat15)) %>%
            mutate(blue_win = factor(blue_win, levels = c(0, 1),
                                     labels = c("Red won", "Blue won"))) %>%
            ggplot(aes(x = blue_golddiffat10, y = blue_golddiffat15, color = blue_win)) +
            geom_point(alpha = 0.15, size = 0.8) +
            scale_color_manual(values = c("Red won" = "#e74c3c", "Blue won" = "#3498db")) +
            geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
            geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
            labs(title = "Blue Gold Diff @10 vs @15",
                 x = "blue_golddiffat10", y = "blue_golddiffat15", color = "Outcome") +
            theme_minimal()
    }, res = 96)

    output$eda_distribution_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Feature distributions  -  no data"))
        scale_check <- eda$games %>%
            select(
                blue_goldat15, blue_xpat15, blue_csat15, blue_killsat15,
                blue_golddiffat15, blue_xpdiffat15, blue_csdiffat15,
                blue_void_grubs
            ) %>%
            drop_na() %>%
            pivot_longer(everything(), names_to = "feature", values_to = "value")
        ggplot(scale_check, aes(x = value)) +
            geom_histogram(bins = 50, fill = "#3498db", alpha = 0.7) +
            facet_wrap(~ feature, scales = "free", ncol = 4) +
            labs(title = "Feature Distributions - Scale Check",
                 x = "Value", y = "Count") +
            theme_minimal()
    }, res = 96)

    output$eda_objectives_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("First objective win rates  -  no data"))
        obj_winrates <- eda_objective_winrates(eda$games)
        ggplot(obj_winrates,
            aes(x = reorder(objective, win_rate_blue),
                y = win_rate_blue, fill = secured_by)
        ) +
            geom_col(position = position_dodge(width = 0.7), width = 0.6) +
            geom_text(aes(label = sprintf("%.1f%% (n=%d)", win_rate_blue, n)),
                      position = position_dodge(width = 0.7),
                      hjust = -0.05, size = 3) +
            coord_flip() +
            scale_fill_manual(values = c("Blue secured" = "#3498db",
                                         "Red secured" = "#e74c3c")) +
            scale_y_continuous(limits = c(0, 100)) +
            geom_hline(yintercept = 50, linetype = "dashed", color = "gray40") +
            labs(title = "P(Blue wins) by First Objective",
                 x = NULL, y = "Blue Win Rate (%)", fill = NULL) +
            theme_minimal()
    }, res = 96)

    output$eda_grubs_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Void grubs  -  no data"))
        eda$games %>%
            filter(!is.na(blue_void_grubs)) %>%
            group_by(blue_void_grubs) %>%
            summarise(win_rate = mean(blue_win) * 100, n = n(), .groups = "drop") %>%
            ggplot(aes(x = factor(blue_void_grubs), y = win_rate, fill = win_rate)) +
            geom_col() +
            geom_text(aes(label = sprintf("%.1f%%\nn=%d", win_rate, n)),
                      vjust = -0.3, size = 3) +
            scale_fill_gradient(low = "#e74c3c", high = "#3498db") +
            scale_y_continuous(limits = c(0, 100)) +
            geom_hline(yintercept = 50, linetype = "dashed", color = "gray40") +
            labs(title = "Blue Win Rate by Void Grubs Secured",
                 x = "blue_void_grubs", y = "P(Blue wins) (%)") +
            theme_minimal() +
            theme(legend.position = "none")
    }, res = 96)

    output$eda_side_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Blue-side advantage  -  no data"))
        games <- eda$games
        n_games <- nrow(games)
        n_blue <- sum(games$blue_win)
        side_pt <- prop.test(n_blue, n_games, p = 0.5)
        tibble(
            label = "Blue side",
            win_rate = n_blue / n_games * 100,
            ci_low = side_pt$conf.int[1] * 100,
            ci_high = side_pt$conf.int[2] * 100
        ) %>%
            ggplot(aes(x = label, y = win_rate)) +
            geom_col(width = 0.4, fill = "#3498db") +
            geom_errorbar(aes(ymin = ci_low, ymax = ci_high),
                          width = 0.15, color = "gray20") +
            geom_text(aes(label = sprintf(
                "%.2f%%  [%.2f, %.2f]\nn=%d   -   chi^2=%.1f, p=%.2g",
                win_rate, ci_low, ci_high, n_games,
                side_pt$statistic, side_pt$p.value
            )), vjust = -0.6, size = 3.5) +
            geom_hline(yintercept = 50, linetype = "dashed", color = "gray40") +
            scale_y_continuous(limits = c(0, 70)) +
            labs(title = "Blue-Side Advantage",
                 x = NULL, y = "P(Blue wins) (%)") +
            theme_minimal() +
            theme(legend.position = "none")
    }, res = 96)

    output$eda_roles_plot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Role correlations  -  no data"))
        role_cors <- eda_role_correlations(eda$games, eda$players)
        ggplot(role_cors, aes(
            x = reorder(position, correlation),
            y = correlation, fill = correlation
        )) +
            geom_col() +
            geom_text(aes(label = sprintf("r = %.3f", correlation)),
                      hjust = -0.1, size = 3.5) +
            coord_flip() +
            scale_y_continuous(limits = c(0, 0.42)) +
            scale_fill_gradient(low = "#f39c12", high = "#2ecc71") +
            labs(title = "Role Gold Diff @15 vs blue_win",
                 x = NULL, y = "Pearson Correlation") +
            theme_minimal() +
            theme(legend.position = "none")
    }, res = 96)

    output$eda_roles_boxplot <- renderPlot({
        eda <- eda_data()
        if (!is.null(eda$error)) return(empty_plot("Role gold diff boxplot  -  no data"))
        role_diffs_blue <- eda$players %>%
            filter(position %in% c("top", "jng", "mid", "bot", "sup"),
                   side == "Blue", !is.na(golddiffat15)) %>%
            select(gameid, position, golddiffat15)
        role_diffs_blue %>%
            inner_join(eda$games %>% select(gameid, blue_win), by = "gameid") %>%
            mutate(
                blue_win = factor(blue_win, levels = c(0, 1),
                                  labels = c("Red won", "Blue won")),
                position = toupper(position)
            ) %>%
            ggplot(aes(x = position, y = golddiffat15, fill = blue_win)) +
            geom_boxplot(outlier.alpha = 0.1) +
            scale_fill_manual(values = c("Red won" = "#e74c3c", "Blue won" = "#3498db")) +
            geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
            labs(title = "Player Gold Differential @15 by Role and Outcome",
                 x = "Role", y = "Gold Diff @15 min", fill = "Outcome") +
            theme_minimal()
    }, res = 96)

    output$data_default_path_note <- renderUI({
        if (is.na(default_raw_path)) {
            return(tagList(
                tags$p(tags$b("No CSV found at the default location."), style = "color:#c0392b"),
                helpText("Drop ", tags$code(RAW_CSV_NAME),
                         " next to shiny_app.R, or switch to upload mode and pick the file.")
            ))
        }
        df <- default_raw_rv()
        if (is.null(df)) {
            tagList(
                helpText("Found at: ", tags$code(default_raw_path)),
                helpText(tags$em("Click the button below to load it."))
            )
        } else {
            tagList(
                helpText("Loaded from: ", tags$code(default_raw_path)),
                helpText(sprintf("%s rows  -  %s columns",
                                 format(nrow(df), big.mark = ","),
                                 format(ncol(df), big.mark = ",")))
            )
        }
    })

    raw_view <- reactive({
        df <- raw_data()
        if (is.null(df)) return(NULL)
        if (isTRUE(input$data_only_team_rows) && "position" %in% names(df)) {
            df <- df[df$position == "team", , drop = FALSE]
        }
        df
    })

    output$data_summary <- renderUI({
        df <- raw_view()
        if (is.null(df)) return(em("No dataset loaded."))
        n_rows <- nrow(df)
        n_cols <- ncol(df)
        n_games   <- if ("gameid"   %in% names(df)) length(unique(df$gameid))   else NA
        n_leagues <- if ("league"   %in% names(df)) length(unique(df$league))   else NA
        n_teams   <- if ("teamname" %in% names(df)) length(unique(df$teamname)) else NA
        miss_pct <- mean(is.na(df)) * 100
        date_range <- if ("date" %in% names(df)) {
            d <- suppressWarnings(as.Date(df$date))
            if (any(!is.na(d))) sprintf("%s - %s",
                format(min(d, na.rm = TRUE), "%Y-%m-%d"),
                format(max(d, na.rm = TRUE), "%Y-%m-%d")) else "-"
        } else "-"
        tags$ul(
            tags$li(tags$b("Rows: "),    format(n_rows, big.mark = ",")),
            tags$li(tags$b("Columns: "), format(n_cols, big.mark = ",")),
            if (!is.na(n_games))   tags$li(tags$b("Games: "),    format(n_games, big.mark = ",")),
            if (!is.na(n_leagues)) tags$li(tags$b("Leagues: "),  n_leagues),
            if (!is.na(n_teams))   tags$li(tags$b("Teams: "),    n_teams),
            tags$li(tags$b("Date range: "), date_range),
            tags$li(tags$b("Missing cells: "), sprintf("%.2f%%", miss_pct))
        )
    })

    output$data_table <- DT::renderDataTable({
        df <- raw_view()
        if (is.null(df)) return(NULL)
        DT::datatable(
            df,
            rownames = FALSE,
            filter   = "top",
            options  = list(
                scrollX     = TRUE,
                pageLength  = 25,
                lengthMenu  = c(10, 25, 50, 100, 250),
                deferRender = TRUE,
                scrollY     = "520px",
                scroller    = TRUE,
                dom         = "Bfrtip"
            ),
            class = "stripe hover compact nowrap"
        )
    })

    output$data_columns_table <- DT::renderDataTable({
        df <- raw_data()
        if (is.null(df)) return(NULL)
        col_info <- data.frame(
            Column     = names(df),
            Type       = vapply(df, function(x) class(x)[1], character(1)),
            `Distinct` = vapply(df, function(x) length(unique(x)), integer(1)),
            `Missing`  = vapply(df, function(x) sum(is.na(x)),     integer(1)),
            `Missing %` = vapply(df, function(x)
                round(mean(is.na(x)) * 100, 2), numeric(1)),
            Example    = vapply(df, function(x) {
                v <- x[!is.na(x)]
                if (length(v) == 0) return("-")
                as.character(v[1])
            }, character(1)),
            check.names = FALSE,
            stringsAsFactors = FALSE
        )
        DT::datatable(col_info, rownames = FALSE, filter = "top",
            options = list(pageLength = 15, lengthMenu = c(15, 30, 60, 120),
                           scrollX = TRUE),
            class = "stripe hover compact")
    })

    output$fs_summary_table <- renderTable({
        method_lbl <- switch(input$fs_method,
            none    = "Full LR (no FS)",
            forward = "Forward stepwise",
            lasso   = "LASSO",
            elnet   = sprintf("Elastic Net (%s)", enet_alpha_label))
        ret <- fs_retained()
        full_n <- ncol(xs_train_mat)
        full_metrics <- metrics_matrix(list("Full LR" = fs_full_preds))
        fs_input <- list(fs_predictions())
        names(fs_input) <- method_lbl
        fs_metrics   <- metrics_matrix(fs_input)
        out <- rbind(
            cbind(Method = full_metrics$Model,
                  `Features kept` = sprintf("%d / %d", full_n, full_n),
                  full_metrics[, c("AUC", "Acc", "Sens", "Spec", "F1", "Kappa")]),
            cbind(Method = fs_metrics$Model,
                  `Features kept` = sprintf("%d / %d", nrow(ret), full_n),
                  fs_metrics[, c("AUC", "Acc", "Sens", "Spec", "F1", "Kappa")])
        )
        if (input$fs_method == "none") return(out[1, , drop = FALSE])
        out
    }, striped = TRUE, hover = TRUE, width = "100%")

}

shinyApp(ui, server)
