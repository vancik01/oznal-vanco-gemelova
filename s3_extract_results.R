suppressPackageStartupMessages({
    library(MASS)       # load first so tidyverse/dplyr::select wins
    library(tidyverse)
    library(caret)
    library(glmnet)
    library(randomForest)
    library(pROC)
})

cat("=== Loading data ===\n")
raw <- read_csv("data/2025_LoL_esports_match_data_from_OraclesElixir.csv",
                show_col_types = FALSE)
complete <- raw %>% filter(datacompleteness == "complete")
teams   <- complete %>% filter(position == "team")
players <- complete %>% filter(position != "team")

# One row per game
game_cols <- c(
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
games_full <- teams %>%
    select(gameid, side, all_of(game_cols)) %>%
    mutate(side = tolower(side)) %>%
    pivot_wider(id_cols = gameid, names_from = side,
                values_from = all_of(game_cols), names_glue = "{side}_{.value}") %>%
    mutate(blue_win = blue_result) %>%
    select(-blue_result, -red_result)

redundant_mirrors <- c(
    "red_golddiffat10","red_xpdiffat10","red_csdiffat10",
    "red_golddiffat15","red_xpdiffat15","red_csdiffat15",
    "red_firstblood","red_firstdragon","red_firstherald","red_firsttower"
)
games <- games_full %>% select(-all_of(redundant_mirrors))

# Rolling win rate
team_wr <- teams %>%
    arrange(date, gameid) %>%
    group_by(teamid) %>%
    mutate(
        games_before = row_number() - 1,
        wins_before  = cumsum(result) - result,
        team_winrate = ifelse(games_before > 0, wins_before / games_before, 0.5)
    ) %>%
    ungroup() %>%
    mutate(side = tolower(side)) %>%
    select(gameid, side, team_winrate, games_before) %>%
    pivot_wider(id_cols = gameid, names_from = side,
                values_from = c(team_winrate, games_before),
                names_glue = "{side}_{.value}")
games_wr <- games %>%
    left_join(team_wr, by = "gameid") %>%
    mutate(winrate_diff = blue_team_winrate - red_team_winrate)

# Feature engineering
games_model <- games_wr %>%
    transmute(
        gameid, blue_win,
        golddiffat15   = blue_golddiffat15,
        xpdiffat15     = blue_xpdiffat15,
        csdiffat15     = blue_csdiffat15,
        gold_momentum  = blue_golddiffat15 - blue_golddiffat10,
        xp_momentum    = blue_xpdiffat15   - blue_xpdiffat10,
        cs_momentum    = blue_csdiffat15   - blue_csdiffat10,
        kill_diff_15   = blue_killsat15 - red_killsat15,
        assist_diff_15 = blue_assistsat15 - red_assistsat15,
        kill_pressure_10_15     = blue_killsat15 - blue_killsat10,
        opp_kill_pressure_10_15 = red_killsat15  - red_killsat10,
        gold_efficiency_15 = blue_goldat15 / pmax(blue_csat15, 1),
        kda_15             = (blue_killsat15 + blue_assistsat15) / pmax(blue_deathsat15, 1),
        plate_diff   = blue_turretplates - red_turretplates,
        grub_diff    = blue_void_grubs   - red_void_grubs,
        winrate_diff,
        firstblood   = blue_firstblood,
        firstdragon  = blue_firstdragon,
        firstherald  = blue_firstherald,
        firsttower   = blue_firsttower
    ) %>%
    drop_na()

cat(sprintf("Feature set: %d features, %d games\n",
            ncol(games_model) - 2, nrow(games_model)))
cat("Features:", paste(setdiff(names(games_model), c("gameid","blue_win")), collapse=", "), "\n\n")

# Train/test split
set.seed(42)
train_ids <- sample(games_model$gameid, size = floor(0.8 * nrow(games_model)))
train_h1  <- games_model %>% filter( gameid %in% train_ids) %>% select(-gameid)
test_h1   <- games_model %>% filter(!gameid %in% train_ids) %>% select(-gameid)

X_train <- train_h1 %>% select(-blue_win)
y_train <- train_h1$blue_win
X_test  <- test_h1  %>% select(-blue_win)
y_test  <- test_h1$blue_win

preproc        <- preProcess(X_train, method = c("center","scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled  <- predict(preproc, X_test)

y_train_fct <- factor(y_train, levels = c(0,1), labels = c("Loss","Win"))
y_test_fct  <- factor(y_test,  levels = c(0,1), labels = c("Loss","Win"))

# ── Minimal S1: LR + RF (needed for baseline and rf_imp) ──────────────────────
cat("=== Training LR (S1 baseline) ===\n")
ctrl_none <- trainControl(method="none", classProbs=TRUE)
lr_model <- train(x=X_train_scaled, y=y_train_fct, method="glm",
                  family="binomial", trControl=ctrl_none, metric="ROC")
lr_prob  <- predict(lr_model, X_test_scaled, type="prob")
lr_pred  <- predict(lr_model, X_test_scaled)
lr_cm    <- confusionMatrix(lr_pred, y_test_fct, positive="Win")
roc_lr   <- roc(y_test_fct, lr_prob[,"Win"], levels=c("Loss","Win"), quiet=TRUE)
cat(sprintf("LR baseline: Acc=%.1f%% AUC=%.3f\n",
            lr_cm$overall["Accuracy"]*100, auc(roc_lr)))

cat("\n=== Training RF (for rf_imp) ===\n")
set.seed(42)
rf_model <- randomForest(x=X_train_scaled, y=y_train_fct,
                         ntree=500, mtry=7, importance=TRUE)
rf_imp <- data.frame(
    feature = rownames(importance(rf_model)),
    Overall = importance(rf_model)[,"MeanDecreaseGini"]
) %>% arrange(desc(Overall))
cat("RF importance computed.\n\n")

# ── S3: RFE ───────────────────────────────────────────────────────────────────
cat("=== S3: RFE ===\n")
lrFuncs_roc <- lrFuncs
lrFuncs_roc$summary <- twoClassSummary
rfe_ctrl <- rfeControl(functions=lrFuncs_roc, method="cv", number=5, verbose=FALSE)
n_features   <- ncol(X_train_scaled)
sizes_to_try <- unique(c(1:5, seq(5,n_features,by=2), n_features))

set.seed(42)
rfe_result <- rfe(x=X_train_scaled, y=y_train_fct,
                  sizes=sizes_to_try, rfeControl=rfe_ctrl, metric="ROC")
rfe_selected <- predictors(rfe_result)

rfe_lr <- train(x=X_train_scaled[,rfe_selected,drop=FALSE], y=y_train_fct,
                method="glm", family="binomial", trControl=ctrl_none)
rfe_pred_class <- predict(rfe_lr, X_test_scaled[,rfe_selected,drop=FALSE])
rfe_pred_prob  <- predict(rfe_lr, X_test_scaled[,rfe_selected,drop=FALSE], type="prob")
rfe_cm  <- confusionMatrix(rfe_pred_class, y_test_fct, positive="Win")
rfe_roc <- roc(y_test_fct, rfe_pred_prob[,"Win"], levels=c("Loss","Win"), quiet=TRUE)

cat(sprintf("Optimal subset size: %d features\n", rfe_result$bestSubset))
cat(sprintf("Selected: %s\n", paste(rfe_selected, collapse=", ")))
cat(sprintf("Acc=%.1f%% AUC=%.3f\n\n", rfe_cm$overall["Accuracy"]*100, auc(rfe_roc)))

# CV profile across sizes
cat("=== RFE CV profile (ROC by n_features) ===\n")
rfe_profile <- rfe_result$results %>% select(Variables, ROC) %>% arrange(Variables)
print(rfe_profile)

# RFE coefficients
rfe_coef <- coef(rfe_lr$finalModel) %>%
    as.data.frame() %>% rownames_to_column("feature") %>%
    rename(coefficient=2) %>% filter(feature!="(Intercept)") %>%
    arrange(desc(abs(coefficient)))
cat("\nRFE coefficients (top features):\n")
print(rfe_coef)

# ── S3: Forward Stepwise ──────────────────────────────────────────────────────
cat("\n=== S3: Forward Stepwise ===\n")
fwd_train_df        <- X_train_scaled
fwd_train_df$result <- y_train_fct
null_model   <- glm(result~1, data=fwd_train_df, family="binomial")
full_formula <- as.formula(paste("result ~", paste(names(X_train_scaled), collapse=" + ")))
fwd_model  <- MASS::stepAIC(null_model,
                             scope=list(lower=null_model, upper=full_formula),
                             direction="forward", trace=FALSE)
fwd_selected <- names(coef(fwd_model))
fwd_selected <- fwd_selected[fwd_selected != "(Intercept)"]

fwd_pred_prob_raw <- predict(fwd_model, newdata=X_test_scaled, type="response")
fwd_pred_class    <- factor(ifelse(fwd_pred_prob_raw>0.5,"Win","Loss"), levels=c("Loss","Win"))
fwd_cm  <- confusionMatrix(fwd_pred_class, y_test_fct, positive="Win")
fwd_roc <- roc(y_test_fct, fwd_pred_prob_raw, levels=c("Loss","Win"), quiet=TRUE)

cat(sprintf("Selected %d features: %s\n", length(fwd_selected),
            paste(fwd_selected, collapse=", ")))
cat(sprintf("AIC=%.1f  Acc=%.1f%%  AUC=%.3f\n", AIC(fwd_model),
            fwd_cm$overall["Accuracy"]*100, auc(fwd_roc)))

# Overlap with RFE
in_both <- intersect(rfe_selected, fwd_selected)
only_rfe <- setdiff(rfe_selected, fwd_selected)
only_fwd <- setdiff(fwd_selected, rfe_selected)
cat(sprintf("In both RFE+Forward: %s\n", paste(in_both, collapse=", ")))
cat(sprintf("Only in RFE: %s\n", paste(only_rfe, collapse=", ")))
cat(sprintf("Only in Forward: %s\n", paste(only_fwd, collapse=", ")))

fwd_coef <- data.frame(feature=fwd_selected, coefficient=coef(fwd_model)[fwd_selected]) %>%
    arrange(desc(abs(coefficient)))
cat("\nForward coefficients:\n")
print(fwd_coef)

# ── S3: LASSO ─────────────────────────────────────────────────────────────────
cat("\n=== S3: LASSO ===\n")
set.seed(42)
lasso_cv <- cv.glmnet(x=as.matrix(X_train_scaled), y=y_train,
                      alpha=1, family="binomial", nfolds=5)

lasso_coefs <- coef(lasso_cv, s="lambda.1se")
lasso_coef_df <- data.frame(
    feature=rownames(lasso_coefs), coefficient=as.numeric(lasso_coefs)
) %>% filter(feature!="(Intercept)") %>% arrange(desc(abs(coefficient)))
lasso_selected   <- lasso_coef_df %>% filter(coefficient!=0)
lasso_eliminated <- lasso_coef_df %>% filter(coefficient==0)

lasso_pred_prob  <- predict(lasso_cv, newx=as.matrix(X_test_scaled),
                             s="lambda.1se", type="response")
lasso_pred_class <- factor(ifelse(lasso_pred_prob>0.5,"Win","Loss"), levels=c("Loss","Win"))
lasso_cm  <- confusionMatrix(lasso_pred_class, y_test_fct, positive="Win")
lasso_roc <- roc(y_test_fct, as.numeric(lasso_pred_prob), levels=c("Loss","Win"), quiet=TRUE)

# Deviance at lambda.min vs lambda.1se
idx_min <- which(lasso_cv$lambda == lasso_cv$lambda.min)
idx_1se <- which(lasso_cv$lambda == lasso_cv$lambda.1se)
dev_min <- lasso_cv$cvm[idx_min]
dev_1se <- lasso_cv$cvm[idx_1se]
dev_pct_increase <- (dev_1se - dev_min) / dev_min * 100

cat(sprintf("lambda.min=%.6f (dev=%.4f, %d features)\n",
            lasso_cv$lambda.min,
            dev_min,
            sum(coef(lasso_cv, s="lambda.min")[-1] != 0)))
cat(sprintf("lambda.1se=%.6f (dev=%.4f, %d features) -> deviance +%.2f%%\n",
            lasso_cv$lambda.1se, dev_1se,
            nrow(lasso_selected), dev_pct_increase))
cat(sprintf("Acc=%.1f%%  AUC=%.3f\n", lasso_cm$overall["Accuracy"]*100, auc(lasso_roc)))
cat(sprintf("Kept (%d): %s\n", nrow(lasso_selected),
            paste(lasso_selected$feature, collapse=", ")))
cat(sprintf("Eliminated (%d): %s\n", nrow(lasso_eliminated),
            paste(lasso_eliminated$feature, collapse=", ")))
cat("\nLASSO coefficients (non-zero):\n")
print(lasso_selected)

# ── S3: Elastic Net ───────────────────────────────────────────────────────────
cat("\n=== S3: Elastic Net ===\n")
alpha_grid <- seq(0, 1, by=0.05)
set.seed(42)
enet_cv_results <- map_dfr(alpha_grid, function(a) {
    cv_fit  <- cv.glmnet(x=as.matrix(X_train_scaled), y=y_train,
                          alpha=a, family="binomial", nfolds=5)
    idx     <- which(cv_fit$lambda == cv_fit$lambda.1se)
    n_kept  <- sum(coef(cv_fit, s="lambda.1se")[-1] != 0)
    tibble(alpha=a, lambda_1se=cv_fit$lambda.1se,
           cv_dev=cv_fit$cvm[idx], cv_dev_se=cv_fit$cvsd[idx], n_features=n_kept)
})
best_row   <- enet_cv_results %>% arrange(cv_dev) %>% slice(1)
best_alpha <- best_row$alpha
best_dev   <- best_row$cv_dev
best_se    <- best_row$cv_dev_se
within_1se <- enet_cv_results %>% filter(cv_dev <= best_dev + best_se)

cat(sprintf("Best alpha=%.2f (dev=%.4f)\n", best_alpha, best_dev))
cat(sprintf("Alphas within 1 SE: %s\n",
            paste(sprintf("%.2f", within_1se$alpha), collapse=", ")))
cat(sprintf("-> CV curve is %s\n",
            ifelse(nrow(within_1se) >= nrow(enet_cv_results)-2,
                   "FLAT (L1/L2 mix underdetermined)",
                   "has a clear optimum")))

set.seed(42)
enet_cv_final <- cv.glmnet(x=as.matrix(X_train_scaled), y=y_train,
                             alpha=best_alpha, family="binomial", nfolds=5)
enet_coef_raw <- coef(enet_cv_final, s="lambda.1se")
enet_coef_df  <- data.frame(feature=rownames(enet_coef_raw),
                              coefficient=as.numeric(enet_coef_raw)) %>%
    filter(feature!="(Intercept)") %>% arrange(desc(abs(coefficient)))
enet_selected_df <- enet_coef_df %>% filter(coefficient!=0)
enet_selected    <- enet_selected_df$feature

enet_pred_prob  <- predict(enet_cv_final, newx=as.matrix(X_test_scaled),
                            s="lambda.1se", type="response")
enet_pred_class <- factor(ifelse(enet_pred_prob>0.5,"Win","Loss"), levels=c("Loss","Win"))
enet_cm  <- confusionMatrix(enet_pred_class, y_test_fct, positive="Win")
enet_roc <- roc(y_test_fct, as.numeric(enet_pred_prob), levels=c("Loss","Win"), quiet=TRUE)

cat(sprintf("Kept (%d): %s\n", length(enet_selected), paste(enet_selected, collapse=", ")))
# Features LASSO dropped but Elastic Net kept
enet_extra <- setdiff(enet_selected, lasso_selected$feature)
cat(sprintf("In Elastic Net but not LASSO: %s\n",
            ifelse(length(enet_extra)==0, "none", paste(enet_extra, collapse=", "))))
cat(sprintf("Acc=%.1f%%  AUC=%.3f\n", enet_cm$overall["Accuracy"]*100, auc(enet_roc)))
cat("\nElastic Net coefficients (non-zero):\n")
print(enet_selected_df)

# ── S3: RF Importance ─────────────────────────────────────────────────────────
cat("\n=== S3: RF Importance ===\n")
rf_importance_threshold <- mean(rf_imp$Overall)
rf_s3_selected <- rf_imp %>% filter(Overall > rf_importance_threshold) %>% pull(feature)

rf_s3_lr <- train(x=X_train_scaled[,rf_s3_selected,drop=FALSE], y=y_train_fct,
                  method="glm", family="binomial", trControl=ctrl_none)
rf_s3_pred_class <- predict(rf_s3_lr, X_test_scaled[,rf_s3_selected,drop=FALSE])
rf_s3_pred_prob  <- predict(rf_s3_lr, X_test_scaled[,rf_s3_selected,drop=FALSE], type="prob")
rf_s3_cm  <- confusionMatrix(rf_s3_pred_class, y_test_fct, positive="Win")
rf_s3_roc <- roc(y_test_fct, rf_s3_pred_prob[,"Win"], levels=c("Loss","Win"), quiet=TRUE)

cat(sprintf("Threshold (mean importance): %.4f\n", rf_importance_threshold))
cat(sprintf("Selected %d / %d features: %s\n",
            length(rf_s3_selected), nrow(rf_imp),
            paste(rf_s3_selected, collapse=", ")))
above <- rf_imp %>% filter(Overall > rf_importance_threshold) %>%
    mutate(label=sprintf("%s=%.2f", feature, Overall))
below <- rf_imp %>% filter(Overall <= rf_importance_threshold) %>%
    mutate(label=sprintf("%s=%.2f", feature, Overall))
cat(sprintf("Above threshold: %s\n", paste(above$label, collapse=", ")))
cat(sprintf("Below threshold: %s\n", paste(below$label, collapse=", ")))
cat(sprintf("Acc=%.1f%%  AUC=%.3f\n", rf_s3_cm$overall["Accuracy"]*100, auc(rf_s3_roc)))

# ── Overlap summary ───────────────────────────────────────────────────────────
cat("\n=== OVERLAP SUMMARY ===\n")
selection_results <- list(
    RFE=rfe_selected, Forward=fwd_selected,
    LASSO=lasso_selected$feature, ElasticNet=enet_selected, RF_Imp=rf_s3_selected
)
all_features <- colnames(X_train_scaled)
overlap_df <- map_dfr(all_features, function(f) {
    sel <- map_lgl(selection_results, ~f %in% .x)
    tibble(feature=f, RFE=sel[["RFE"]], Forward=sel[["Forward"]],
           LASSO=sel[["LASSO"]], ElasticNet=sel[["ElasticNet"]],
           RF_Imp=sel[["RF_Imp"]], n_methods=sum(sel))
}) %>% arrange(desc(n_methods), feature)
print(overlap_df, n=Inf)

cat("\nConsensus (all 5):", paste(overlap_df$feature[overlap_df$n_methods==5], collapse=", "), "\n")
cat("4/5 methods:      ", paste(overlap_df$feature[overlap_df$n_methods==4], collapse=", "), "\n")
cat("Never selected:   ", paste(overlap_df$feature[overlap_df$n_methods==0], collapse=", "), "\n")

# ── Performance comparison ────────────────────────────────────────────────────
cat("\n=== PERFORMANCE COMPARISON ===\n")
perf <- tibble(
    Method     = c("LR_all","RFE","Forward","LASSO","ElasticNet","RF_Imp"),
    n_features = c(ncol(X_train_scaled), length(rfe_selected), length(fwd_selected),
                   nrow(lasso_selected), length(enet_selected), length(rf_s3_selected)),
    Accuracy   = c(lr_cm$overall["Accuracy"], rfe_cm$overall["Accuracy"],
                   fwd_cm$overall["Accuracy"], lasso_cm$overall["Accuracy"],
                   enet_cm$overall["Accuracy"], rf_s3_cm$overall["Accuracy"]) * 100,
    AUC        = c(auc(roc_lr), auc(rfe_roc), auc(fwd_roc), auc(lasso_roc),
                   auc(enet_roc), auc(rf_s3_roc)) * 100
)
print(perf %>% arrange(desc(AUC)))
