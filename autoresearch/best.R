#!/usr/bin/env Rscript
# Autoresearch experiment - H1 (team-level early-game prediction)
# This is the ONLY file the agent edits.
# Must print final `METRIC: <auc>` on stdout. See program.md.

suppressPackageStartupMessages({
    library(tidyverse)
    library(xgboost)
    library(pROC)
    library(doParallel)
    library(glmnet)
})

t_start <- Sys.time()
n_cores <- max(1, parallel::detectCores() - 1)

# ----------------------------- DATA LOAD (LOCKED) -----------------------------
raw <- read_csv(
    "data/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    show_col_types = FALSE
)
complete <- raw %>% filter(datacompleteness == "complete")
teams <- complete %>% filter(position == "team")

# Rolling team features with Bayesian shrinkage (no leakage).
PRIOR_N <- 10
teams <- teams %>%
    arrange(date, gameid) %>%
    group_by(teamid) %>%
    mutate(
        games_before = row_number() - 1,
        wins_before = cumsum(result) - result,
        team_winrate = (wins_before + PRIOR_N * 0.5) / (games_before + PRIOR_N),
        # Rolling mean of early-game signals over PRIOR games (excluding current).
        # Shrunk toward 0 with PRIOR_N pseudo-games.
        gd15_cum = cumsum(replace_na(golddiffat15, 0)) - replace_na(golddiffat15, 0),
        xd15_cum = cumsum(replace_na(xpdiffat15, 0)) - replace_na(xpdiffat15, 0),
        cd15_cum = cumsum(replace_na(csdiffat15, 0)) - replace_na(csdiffat15, 0),
        team_gd15_avg = gd15_cum / (games_before + PRIOR_N),
        team_xd15_avg = xd15_cum / (games_before + PRIOR_N),
        team_cd15_avg = cd15_cum / (games_before + PRIOR_N)
    ) %>%
    ungroup() %>%
    select(-gd15_cum, -xd15_cum, -cd15_cum)

opp_feats <- teams %>%
    select(gameid, side, team_winrate, games_before,
           team_gd15_avg, team_xd15_avg, team_cd15_avg) %>%
    mutate(opp_side = ifelse(side == "Blue", "Red", "Blue")) %>%
    select(gameid, opp_side,
           opp_winrate      = team_winrate,
           opp_games_before = games_before,
           opp_gd15_avg     = team_gd15_avg,
           opp_xd15_avg     = team_xd15_avg,
           opp_cd15_avg     = team_cd15_avg)
opp_wr <- opp_feats  # backwards compatibility with name used below

# -------------------------- FEATURE ENGINEERING (EDITABLE) --------------------
raw_cols <- c(
    "goldat10", "xpat10", "csat10",
    "goldat15", "xpat15", "csat15",
    "golddiffat10", "xpdiffat10", "csdiffat10",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "killsat10", "killsat15", "deathsat15", "assistsat15",
    "opp_killsat10", "opp_killsat15", "opp_deathsat15", "opp_assistsat15",
    "firstblood", "firstdragon", "firstherald", "firsttower",
    "void_grubs", "opp_void_grubs",
    "turretplates", "opp_turretplates",
    "side"
)

teams_model <- teams %>%
    select(all_of(c("gameid", "result",
                    "team_winrate", "games_before",
                    "team_gd15_avg", "team_xd15_avg", "team_cd15_avg",
                    raw_cols))) %>%
    left_join(opp_wr, by = c("gameid", "side" = "opp_side")) %>%
    mutate(
        opp_winrate      = replace_na(opp_winrate, 0.5),
        opp_games_before = replace_na(opp_games_before, 0),
        opp_gd15_avg     = replace_na(opp_gd15_avg, 0),
        opp_xd15_avg     = replace_na(opp_xd15_avg, 0),
        opp_cd15_avg     = replace_na(opp_cd15_avg, 0)
    ) %>%
    drop_na() %>%
    mutate(
        gold_momentum = golddiffat15 - golddiffat10,
        xp_momentum = xpdiffat15 - xpdiffat10,
        cs_momentum = csdiffat15 - csdiffat10,
        kill_pressure_10_15 = killsat15 - killsat10,
        opp_kill_pressure_10_15 = opp_killsat15 - opp_killsat10,
        kill_diff_15 = killsat15 - opp_killsat15,
        death_diff_15 = deathsat15 - opp_deathsat15,
        assist_diff_15 = assistsat15 - opp_assistsat15,
        gold_efficiency_15 = goldat15 / pmax(csat15, 1),
        grub_diff = void_grubs - opp_void_grubs,
        kda_15 = (killsat15 + assistsat15) / pmax(deathsat15, 1),
        plate_diff = turretplates - opp_turretplates,
        winrate_diff = team_winrate - opp_winrate,
        # Team historical early-game strength differentials (prior-game averages)
        gd15_avg_diff = team_gd15_avg - opp_gd15_avg,
        xd15_avg_diff = team_xd15_avg - opp_xd15_avg,
        cd15_avg_diff = team_cd15_avg - opp_cd15_avg,
        side = ifelse(side == "Blue", 1L, 0L)
    ) %>%
    mutate(
        games_reliability = pmin(games_before, opp_games_before)
    ) %>%
    select(-any_of(c(
        "goldat10", "xpat10", "csat10",
        "goldat15", "xpat15", "csat15",
        "golddiffat10", "xpdiffat10", "csdiffat10",
        "killsat10", "killsat15", "deathsat15", "assistsat15",
        "opp_killsat10", "opp_killsat15", "opp_deathsat15", "opp_assistsat15",
        "void_grubs", "opp_void_grubs",
        "turretplates", "opp_turretplates",
        "team_winrate", "opp_winrate",
        "games_before", "opp_games_before",
        "team_gd15_avg", "team_xd15_avg", "team_cd15_avg",
        "opp_gd15_avg",  "opp_xd15_avg",  "opp_cd15_avg"
    )))

# --------------------------------- SPLIT (LOCKED) -----------------------------
set.seed(42)
game_ids <- unique(teams_model$gameid)
train_games <- sample(game_ids, size = floor(0.8 * length(game_ids)))

train_df <- teams_model %>% filter(gameid %in% train_games) %>% select(-gameid)
test_df  <- teams_model %>% filter(!gameid %in% train_games) %>% select(-gameid)

# Run 10: keep gameid with training rows to build group-based CV folds
train_full <- teams_model %>% filter(gameid %in% train_games)
X_te <- as.matrix(test_df %>% select(-result))
y_te <- test_df$result

# --------------------------------- MODEL (EDITABLE) ---------------------------
# Run 10: game-level 5-fold CV (both Blue/Red rows of a game stay in same fold).
X_full <- as.matrix(train_full %>% select(-gameid, -result))
y_full <- train_full$result
gid_full <- train_full$gameid

set.seed(42)
uniq_games <- unique(gid_full)
game_fold <- sample(rep(1:5, length.out = length(uniq_games)))
names(game_fold) <- uniq_games
row_fold <- game_fold[as.character(gid_full)]
cv_folds <- lapply(1:5, function(f) which(row_fold == f))
cat(sprintf("Group-based folds: %s rows per fold\n",
            paste(sapply(cv_folds, length), collapse = "/")))

grid <- expand.grid(
    eta        = c(0.03, 0.05, 0.1),
    max_depth  = c(3, 5, 7),
    min_child  = c(1, 5),
    subsample  = c(0.8),
    colsample  = c(0.8)
)

cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)

cv_results <- foreach(
    i = seq_len(nrow(grid)),
    .packages = c("xgboost"),
    .export   = c("X_full", "y_full", "grid", "cv_folds")
) %dopar% {
    dfull_local <- xgb.DMatrix(X_full, label = y_full)
    params_i <- list(
        objective        = "binary:logistic",
        eval_metric      = "auc",
        eta              = grid$eta[i],
        max_depth        = grid$max_depth[i],
        min_child_weight = grid$min_child[i],
        subsample        = grid$subsample[i],
        colsample_bytree = grid$colsample[i],
        nthread          = 1
    )
    set.seed(42)
    cv <- xgb.cv(
        params                = params_i,
        data                  = dfull_local,
        folds                 = cv_folds,
        nrounds               = 1500,
        early_stopping_rounds = 30,
        verbose               = 0,
        maximize              = TRUE
    )
    list(
        params    = params_i,
        best_iter = cv$best_iteration,
        best_auc  = max(cv$evaluation_log$test_auc_mean)
    )
}
stopCluster(cl)

best_idx <- which.max(sapply(cv_results, function(x) x$best_auc))
best <- cv_results[[best_idx]]

cat(sprintf("CV best: eta=%.3f depth=%d mcw=%d ss=%.1f cs=%.1f nr=%d cvAUC=%.5f\n",
    best$params$eta, best$params$max_depth, best$params$min_child_weight,
    best$params$subsample, best$params$colsample_bytree,
    best$best_iter, best$best_auc))

final_params <- best$params

# ---- OOF XGB + OOF LASSO, pick blend weight on OOF ----
mu <- colMeans(X_full); sigma <- apply(X_full, 2, sd); sigma[sigma == 0] <- 1
X_full_std <- scale(X_full, center = mu, scale = sigma)
X_te_std   <- scale(X_te,   center = mu, scale = sigma)

n <- nrow(X_full)
oof_xgb <- numeric(n); oof_glm <- numeric(n)

cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)
oof_out <- foreach(
    f = 1:5,
    .packages = c("xgboost", "glmnet"),
    .export   = c("X_full", "y_full", "X_full_std", "cv_folds", "final_params", "best")
) %dopar% {
    tr <- setdiff(seq_len(nrow(X_full)), cv_folds[[f]])
    va <- cv_folds[[f]]
    pp <- final_params; pp$nthread <- 1
    set.seed(200 + f)
    dtr <- xgb.DMatrix(X_full[tr, ], label = y_full[tr])
    mdl <- xgb.train(params = pp, data = dtr, nrounds = best$best_iter, verbose = 0)
    px <- predict(mdl, X_full[va, ])
    set.seed(200 + f)
    cvg <- cv.glmnet(X_full_std[tr, ], y_full[tr], alpha = 1,
                     family = "binomial", nfolds = 5, type.measure = "auc")
    pg <- as.numeric(predict(cvg, newx = X_full_std[va, ], s = "lambda.min", type = "response"))
    list(va = va, px = px, pg = pg)
}
stopCluster(cl)
for (o in oof_out) { oof_xgb[o$va] <- o$px; oof_glm[o$va] <- o$pg }

auc_fn <- function(y, p) as.numeric(auc(suppressMessages(roc(y, p, levels = c(0,1), direction = "<"))))
cat(sprintf("OOF AUC: xgb=%.5f glm=%.5f\n",
            auc_fn(y_full, oof_xgb), auc_fn(y_full, oof_glm)))

alphas <- seq(0, 1, by = 0.02)
oof_blend_aucs <- sapply(alphas, function(a) auc_fn(y_full, a*oof_xgb + (1-a)*oof_glm))
best_alpha <- alphas[which.max(oof_blend_aucs)]
cat(sprintf("Best blend alpha (XGB weight): %.2f, OOF blend AUC: %.5f\n",
            best_alpha, max(oof_blend_aucs)))

# ---- Retrain both on full train, blend on test ----
dfull <- xgb.DMatrix(X_full, label = y_full)
final_params$nthread <- n_cores
set.seed(42)
xgb_full <- xgb.train(params = final_params, data = dfull,
                      nrounds = best$best_iter, verbose = 0)
B <- 20
cl <- makePSOCKcluster(n_cores); registerDoParallel(cl)
bag_preds <- foreach(
    b = 1:B,
    .packages = c("glmnet"),
    .combine  = cbind,
    .export   = c("X_full_std", "y_full", "X_te_std")
) %dopar% {
    set.seed(1000 + b)
    idx <- sample(seq_len(nrow(X_full_std)), replace = TRUE)
    cvg <- cv.glmnet(X_full_std[idx, ], y_full[idx], alpha = 1,
                     family = "binomial", nfolds = 5, type.measure = "auc")
    as.numeric(predict(cvg, newx = X_te_std, s = "lambda.min", type = "response"))
}
stopCluster(cl)

p_xgb_te <- predict(xgb_full, X_te)
p_glm_te <- rowMeans(bag_preds)
cat(sprintf("Bagged LASSO over %d bootstraps\n", B))

# ---------------------------- EVAL (LOCKED OUTPUT) ----------------------------
pred <- best_alpha * p_xgb_te + (1 - best_alpha) * p_glm_te
roc_obj <- suppressMessages(roc(y_te, pred, levels = c(0, 1), direction = "<"))
auc_val <- as.numeric(auc(roc_obj))
acc <- mean((pred > 0.5) == y_te)

dur <- as.numeric(Sys.time() - t_start, units = "secs")

cat(sprintf("ACC: %.5f\n", acc))
cat(sprintf("N_FEATURES: %d\n", ncol(X_te)))
cat(sprintf("DURATION_SEC: %.1f\n", dur))
cat(sprintf("BEST_NROUNDS: %d\n", best$best_iter))
cat(sprintf("METRIC: %.5f\n", auc_val))
