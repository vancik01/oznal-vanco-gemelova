#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(glmnet)
    library(xgboost)
    library(pROC)
    library(parallel)
})

t_start <- Sys.time()
set.seed(42)

HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
DATA_RDS <- file.path(HERE, "frames_dataset.rds")

if (!file.exists(DATA_RDS)) {
    stop("Missing frames_dataset.rds. Run `Rscript prepare_dataset.R` first.")
}

df <- readRDS(DATA_RDS)
df <- as.data.table(df)

key_cols <- c("oracle_gameid", "esports_game_id", "league", "frame_group", "side", "result")

frame_feature_cols <- setdiff(
    names(df),
    c(key_cols, grep("^oracle_", names(df), value = TRUE))
)

model_df <- copy(df[, c("oracle_gameid", "result", "league", frame_feature_cols), with = FALSE])

# Run 13: league one-hot indicators. Leagues differ in early-game pace
# (LCK methodical, LPL aggressive, minor regions more volatile), so the
# same gold-lead-at-15 may map to different win probabilities across leagues.
# Build dummies, then drop the raw league column so it stays out of the
# numeric feature matrix.
league_levels <- sort(unique(as.character(model_df$league)))
for (lv in league_levels) {
    safe_name <- paste0("lg_", gsub("[^A-Za-z0-9]", "_", lv))
    model_df[, (safe_name) := as.integer(league == lv)]
}
model_df[, league := NULL]

numeric_cols <- setdiff(names(model_df), c("oracle_gameid", "result"))
for (col in numeric_cols) {
    set(model_df, which(!is.finite(model_df[[col]]) | is.na(model_df[[col]])), col, 0)
}

# Run 5: engineered meta-features derived from the prepared columns.
# All features use information at or before minute 15 only.
safe_div <- function(a, b, eps = 1e-6) a / (abs(b) + eps) * sign(b + eps)
safe_ratio <- function(a, b, eps = 1e-6) a / (abs(b) + eps)
sign_int <- function(x) as.integer(sign(x))

eng <- function(d) {
    # Archetype indicators (binary 0/1)
    d[, arch_comeback_5_15 := as.integer(f_5_gold_diff < -200 & f_15_gold_diff > 200)]
    d[, arch_collapse_5_15 := as.integer(f_5_gold_diff > 200 & f_15_gold_diff < -200)]
    d[, arch_stable_lead := as.integer(gold_adv_time_share > 0.7 & gold_lead_changes <= 2 & f_15_gold_diff > 0)]
    d[, arch_stable_trail := as.integer(gold_trail_time_share > 0.7 & gold_lead_changes <= 2 & f_15_gold_diff < 0)]
    d[, arch_volatile_game := as.integer(gold_lead_changes >= 4)]
    d[, arch_early_snowball := as.integer(f_5_gold_diff > 500 & gold_momentum_5_10 > 0 & gold_momentum_10_15 > 0)]
    d[, arch_late_surge := as.integer(gold_diff_first5_mean < 100 & gold_diff_last3_mean > 400)]
    d[, arch_curb_stomp := as.integer(f_15_gold_diff > 3000 & f_15_tower_diff > 2)]
    d[, arch_dead_side := as.integer(f_15_gold_diff < -2000 & f_15_tower_diff < 0)]
    d[, arch_even_game := as.integer(abs(f_15_gold_diff) < 500 & abs(f_15_tower_diff) <= 1)]

    # Normalized / ratio features
    d[, gold_diff_ratio_15 := safe_ratio(f_15_gold_diff, f_15_gold)]
    d[, gold_diff_ratio_10 := safe_ratio(f_10_gold_diff, f_10_gold)]
    d[, xp_gold_balance_15 := f_15_level_diff - safe_ratio(f_15_gold_diff, 1500)]
    d[, gold_per_tower_15 := safe_div(f_15_gold_diff, f_15_tower_diff + 0.5)]
    d[, gold_per_kill_15 := safe_div(f_15_gold_diff, f_15_kill_diff + 0.5)]
    d[, lead_persistence := gold_adv_time_share * sign(f_15_gold_diff) *
                             pmin(abs(f_15_gold_diff) / 1000, 5)]
    d[, range_of_swing := max_gold_lead_0_15 - max_gold_deficit_0_15]

    # Shape / trajectory features
    d[, gold_half_delta := gold_diff_last5_mean - gold_diff_first5_mean]
    d[, gold_accel_5_15 := gold_momentum_10_15 - gold_momentum_5_10]
    d[, gold_diff_cv := gold_diff_sd_0_15 / (abs(gold_diff_mean_0_15) + 100)]
    d[, tempo_delta_5_15 := f_15_gold_diff - f_5_gold_diff]
    d[, tempo_delta_10_15 := f_15_gold_diff - f_10_gold_diff]
    d[, kills_accel_5_15 := kill_momentum_10_15 - kill_momentum_5_10]
    d[, cs_accel_5_15 := cs_momentum_10_15 - cs_momentum_5_10]

    # Role concentration / alignment (rowwise apply is slow — use matrix ops)
    role_cols_15 <- c("top_15_gold_diff", "jng_15_gold_diff", "mid_15_gold_diff",
                      "bot_15_gold_diff", "sup_15_gold_diff")
    role_mat <- as.matrix(d[, ..role_cols_15])
    role_abs_sum <- rowSums(abs(role_mat)) + 1
    role_max_abs <- do.call(pmax, as.data.frame(abs(role_mat)))
    d[, role_gold_concentration_15 := role_max_abs / role_abs_sum]
    d[, role_gold_sign_sum_15 := as.integer(rowSums(sign(role_mat)))]
    d[, role_gold_all_positive := as.integer(rowSums(role_mat > 0) == 5)]
    d[, role_gold_all_negative := as.integer(rowSums(role_mat < 0) == 5)]
    d[, role_gold_mean_15 := rowMeans(role_mat)]

    # Composite carry/solo/jungle cross-products
    d[, carry_solo_product := sign(carry_gold_diff_15) * sign(solo_lane_gold_diff_15) *
                               sqrt(abs(carry_gold_diff_15) * abs(solo_lane_gold_diff_15))]
    d[, carry_lead_binary := as.integer(carry_gold_diff_15 > 200)]
    d[, solo_lead_binary := as.integer(solo_lane_gold_diff_15 > 200)]

    # Objective synergy
    d[, first3_objectives_me := got_first_blood + got_first_dragon + got_first_tower]
    d[, objectives_uncontested := as.integer(got_first_blood == 1 & got_first_dragon == 1 &
                                             got_first_tower == 1)]
    d[, objectives_split := as.integer(got_first_blood != got_first_dragon |
                                       got_first_dragon != got_first_tower)]
    d[, dragon_dominance := drag_chemtech_diff_15 + drag_cloud_diff_15 + drag_hextech_diff_15 +
                             drag_infernal_diff_15 + drag_mountain_diff_15 + drag_ocean_diff_15]

    # Interactions between top-signal features
    d[, gold15_x_tower15 := f_15_gold_diff * f_15_tower_diff]
    d[, gold15_x_dragon15 := f_15_gold_diff * f_15_dragon_diff]
    d[, goldmom_x_firstdrag := gold_momentum_10_15 * first_dragon_me]
    d[, goldmom_x_firsttower := gold_momentum_10_15 * first_tower_me]
    d[, gold15_x_side := f_15_gold_diff * side_blue]

    # Clipped / signed transforms (reduce outlier influence)
    d[, gold15_clip := pmax(pmin(f_15_gold_diff, 6000), -6000)]
    d[, gold15_sqrt := sign(f_15_gold_diff) * sqrt(abs(f_15_gold_diff))]

    # Run 18: polynomial expansions of top-signal features. ENET is linear
    # in its inputs; x² / x³ terms let it represent the sigmoid-shape
    # win-prob response (steep near 0, flat at extremes). Scale dividing
    # so standardized magnitudes stay bounded before `scale()`.
    d[, gold15_sq := (f_15_gold_diff / 1000)^2]
    d[, gold15_cu := (f_15_gold_diff / 1000)^3]
    d[, gold10_sq := (f_10_gold_diff / 1000)^2]
    d[, goldmom1015_sq := gold_momentum_10_15^2]
    d[, goldmom0510_sq := gold_momentum_5_10^2]
    d[, tower15_sq := f_15_tower_diff^2 * sign(f_15_tower_diff)]
    d[, kill15_sq := f_15_kill_diff^2 * sign(f_15_kill_diff)]
    d[, halfdelta_sq := (gold_half_delta / 1000)^2]

    # Replace any NA / non-finite produced by the above
    new_cols <- setdiff(names(d), frame_feature_cols)
    new_cols <- setdiff(new_cols, c("oracle_gameid", "result"))
    for (col in new_cols) {
        set(d, which(!is.finite(d[[col]]) | is.na(d[[col]])), col, 0)
    }
    d
}

model_df <- eng(model_df)
numeric_cols <- setdiff(names(model_df), c("oracle_gameid", "result"))

game_ids <- unique(model_df$oracle_gameid)
train_games <- sample(game_ids, floor(0.8 * length(game_ids)))

train_df <- model_df[oracle_gameid %in% train_games]
test_df <- model_df[!oracle_gameid %in% train_games]

X_train <- as.matrix(train_df[, ..numeric_cols])
y_train <- train_df$result
X_test <- as.matrix(test_df[, ..numeric_cols])
y_test <- test_df$result

baseline_pred <- as.integer(test_df$f_15_gold_diff > 0)
baseline_acc <- mean(baseline_pred == y_test)

mu <- colMeans(X_train)
sdv <- apply(X_train, 2, sd)
sdv[sdv == 0] <- 1
X_train_std <- scale(X_train, center = mu, scale = sdv)
X_test_std <- scale(X_test, center = mu, scale = sdv)

n_cores <- parallel::detectCores()
data.table::setDTthreads(n_cores)

# Run 4: elastic-net with alpha grid. Shared foldid so the CV AUC across alphas
# is comparable. Parallelize across alphas (each fit is small, 6 alphas fit in
# 6 cores comfortably).
alpha_grid <- c(0, 0.1, 0.25, 0.5, 0.75, 1.0)

set.seed(42)
foldid <- sample(rep(seq_len(5), length.out = length(y_train)))

enet_eval_one <- function(a) {
    set.seed(42)
    cv <- cv.glmnet(
        x = X_train_std,
        y = y_train,
        family = "binomial",
        alpha = a,
        foldid = foldid,
        type.measure = "auc"
    )
    list(
        alpha = a,
        cv_auc = max(cv$cvm),
        lambda_min = cv$lambda.min,
        fit = cv
    )
}

t_enet <- Sys.time()
enet_results <- parallel::mclapply(
    alpha_grid,
    enet_eval_one,
    mc.cores = min(n_cores, length(alpha_grid)),
    mc.preschedule = FALSE
)
t_enet_sec <- as.numeric(difftime(Sys.time(), t_enet, units = "secs"))
cat(sprintf("Using %d cores\n", n_cores))
cat(sprintf("Elastic-net alpha grid (%d alphas, parallel) done in %.1fs\n", length(alpha_grid), t_enet_sec))

for (r in enet_results) {
    cat(sprintf("  alpha=%.2f -> cvAUC=%.5f (lambda.min=%.8f)\n", r$alpha, r$cv_auc, r$lambda_min))
}

enet_best_idx <- which.max(vapply(enet_results, function(r) r$cv_auc, numeric(1)))
enet_best <- enet_results[[enet_best_idx]]
cat(sprintf("Best alpha=%.2f cvAUC=%.5f lambda.min=%.8f\n",
            enet_best$alpha, enet_best$cv_auc, enet_best$lambda_min))

enet_prob <- as.numeric(predict(
    enet_best$fit, newx = X_test_std, s = "lambda.min", type = "response"
))

dtrain <- xgb.DMatrix(X_train, label = y_train)
dtest <- xgb.DMatrix(X_test, label = y_test)

xgb_cv <- xgb.cv(
    params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.05,
        max_depth = 4,
        min_child_weight = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        nthread = 1
    ),
    data = dtrain,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 25,
    verbose = 0,
    maximize = TRUE
)

xgb_fit <- xgb.train(
    params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.05,
        max_depth = 4,
        min_child_weight = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        nthread = 1
    ),
    data = dtrain,
    nrounds = xgb_cv$best_iteration,
    verbose = 0
)
xgb_prob <- predict(xgb_fit, dtest)

# Run 21: shift blend weight toward ENET now that Run 18's polynomial
# features strengthened it specifically (the x²/x³ curvature helps the
# linear arm more than the tree arm, which already handles curvature).
blend_prob <- 0.75 * enet_prob + 0.25 * xgb_prob
pred_class <- as.integer(blend_prob >= 0.5)

auc_value <- as.numeric(auc(roc(y_test, blend_prob, levels = c(0, 1), direction = "<")))
acc_value <- mean(pred_class == y_test)

duration <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

cat(sprintf("Train games: %d | Test games: %d\n", length(unique(train_df$oracle_gameid)), length(unique(test_df$oracle_gameid))))
cat(sprintf("Rows: train=%d test=%d | Features=%d\n", nrow(train_df), nrow(test_df), length(numeric_cols)))
cat(sprintf("Baseline acc (gold diff @15 > 0): %.5f\n", baseline_acc))
cat(sprintf("ENET cv AUC: %.5f | alpha=%.2f lambda.min=%.8f\n",
            enet_best$cv_auc, enet_best$alpha, enet_best$lambda_min))
cat(sprintf("XGB cv AUC: %.5f | rounds=%d\n", max(xgb_cv$evaluation_log$test_auc_mean), xgb_cv$best_iteration))

cat(sprintf("AUC: %.5f\n", auc_value))
cat(sprintf("ACC: %.5f\n", acc_value))
cat(sprintf("BASELINE_ACC: %.5f\n", baseline_acc))
cat(sprintf("N_FEATURES: %d\n", length(numeric_cols)))
cat(sprintf("DURATION_SEC: %.2f\n", duration))
