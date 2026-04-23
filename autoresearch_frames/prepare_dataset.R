#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(parallel)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
this_file <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[1]))
} else {
    normalizePath("prepare_dataset.R")
}
HERE <- dirname(this_file)
PROJECT <- dirname(HERE)

ORACLE_CSV <- file.path(PROJECT, "data", "2025_LoL_esports_match_data_from_OraclesElixir.csv")
POC_DIR <- file.path(PROJECT, "autoresearch", "poc")
FRAMES_DIR <- file.path(PROJECT, "data", "frames")

OUT_RDS <- file.path(HERE, "frames_dataset.rds")
OUT_CSV <- file.path(HERE, "frames_dataset.csv")
OUT_SUMMARY <- file.path(HERE, "dataset_summary.txt")

safe_num <- function(x, default = 0) {
    if (is.null(x) || length(x) == 0) return(default)
    out <- suppressWarnings(as.numeric(x))
    out[is.na(out)] <- default
    out
}

parse_time_utc <- function(x) {
    if (is.null(x) || length(x) == 0 || is.na(x) || !nzchar(x)) {
        return(as.POSIXct(NA, tz = "UTC"))
    }
    fmt <- if (grepl("T", x, fixed = TRUE)) "%Y-%m-%dT%H:%M:%OSZ" else "%Y-%m-%d %H:%M:%OS"
    as.POSIXct(x, tz = "UTC", format = fmt)
}

read_mapping_files <- function() {
    files <- list.files(POC_DIR, pattern = "^mapping.*\\.csv$", recursive = TRUE, full.names = TRUE)
    tables <- lapply(files, function(path) {
        dt <- tryCatch(fread(path), error = function(e) NULL)
        if (is.null(dt) || !all(c("oracle_gameid", "esports_game_id") %in% names(dt))) {
            return(NULL)
        }
        keep <- intersect(c("oracle_gameid", "esports_game_id", "date", "start_time"), names(dt))
        out <- unique(dt[, ..keep])
        for (col in keep) {
            set(out, j = col, value = as.character(out[[col]]))
        }
        if ("date" %in% names(out)) {
            setnames(out, "date", "mapping_date")
        }
        out[, source := path]
        out
    })
    maps <- rbindlist(Filter(Negate(is.null), tables), fill = TRUE)
    unique(maps, by = c("oracle_gameid", "esports_game_id"))
}

build_frame_index <- function() {
    files <- list.files(FRAMES_DIR, pattern = "\\.ndjson$", recursive = TRUE, full.names = TRUE)
    data.table(
        esports_game_id = sub("\\.ndjson$", "", basename(files)),
        frame_path = files,
        frame_group = basename(dirname(files))
    )
}

load_oracle_team_rows <- function() {
    cols <- c(
        "gameid", "league", "date", "side", "result", "position",
        "goldat10", "goldat15", "xpat10", "xpat15", "csat10", "csat15",
        "golddiffat10", "golddiffat15", "xpdiffat10", "xpdiffat15",
        "csdiffat10", "csdiffat15", "killsat10", "killsat15",
        "deathsat10", "deathsat15", "assistsat10", "assistsat15",
        "opp_killsat10", "opp_killsat15", "opp_deathsat10", "opp_deathsat15",
        "opp_assistsat10", "opp_assistsat15", "firstblood", "firstdragon",
        "firstherald", "firsttower", "void_grubs", "opp_void_grubs",
        "turretplates", "opp_turretplates"
    )
    dt <- fread(ORACLE_CSV, select = cols, showProgress = FALSE)
    dt[position == "team"][
        , oracle_gameid := as.character(gameid)
    ][
        , oracle_date := date
    ][
        , position := NULL
    ][]
}

team_snapshot <- function(gid, side, ts, raw_minute, actual_minute, team) {
    ps <- team$participants %||% list()
    if (is.data.frame(ps)) {
        pdt <- as.data.table(ps)
    } else if (length(ps)) {
        pdt <- rbindlist(lapply(ps, as.data.table), fill = TRUE)
    } else {
        pdt <- data.table()
    }
    if (!nrow(pdt)) {
        pdt <- data.table(
            participantId = integer(),
            totalGold = numeric(),
            level = numeric(),
            kills = numeric(),
            deaths = numeric(),
            assists = numeric(),
            creepScore = numeric(),
            currentHealth = numeric(),
            maxHealth = numeric()
        )
    }
    dragons <- team$dragons %||% character()
    data.table(
        esports_game_id = gid,
        side = side,
        ts = ts,
        raw_minute = raw_minute,
        actual_minute = actual_minute,
        kills = safe_num(team$totalKills),
        gold = safe_num(team$totalGold),
        towers = safe_num(team$towers),
        barons = safe_num(team$barons),
        inhibitors = safe_num(team$inhibitors),
        dragons_count = length(dragons),
        dragon_types = paste(dragons, collapse = "|"),
        cs_sum = sum(safe_num(pdt$creepScore), na.rm = TRUE),
        level_sum = sum(safe_num(pdt$level), na.rm = TRUE),
        level_max = max(c(0, safe_num(pdt$level)), na.rm = TRUE),
        n_alive = sum(safe_num(pdt$currentHealth) > 0, na.rm = TRUE),
        hp_sum = sum(safe_num(pdt$currentHealth), na.rm = TRUE),
        hp_ratio_sum = sum(safe_num(pdt$currentHealth) / pmax(safe_num(pdt$maxHealth), 1), na.rm = TRUE)
    )
}

player_snapshot <- function(gid, side, raw_minute, actual_minute, team) {
    ps <- team$participants %||% list()
    if (!length(ps)) return(data.table())
    if (is.data.frame(ps)) {
        pdt <- as.data.table(ps)
    } else {
        pdt <- rbindlist(lapply(ps, as.data.table), fill = TRUE)
    }
    if (!nrow(pdt)) return(data.table())
    roles <- c("top", "jng", "mid", "bot", "sup")
    pdt[, role := roles[((participantId - 1L) %% 5L) + 1L]]
    pdt[, `:=`(
        esports_game_id = gid,
        side = side,
        raw_minute = raw_minute,
        actual_minute = actual_minute,
        gold = safe_num(totalGold),
        level = safe_num(level),
        kills = safe_num(kills),
        deaths = safe_num(deaths),
        assists = safe_num(assists),
        cs = safe_num(creepScore),
        hp = safe_num(currentHealth),
        hp_ratio = safe_num(currentHealth) / pmax(safe_num(maxHealth), 1)
    )]
    pdt[, .(
        esports_game_id, side, raw_minute, actual_minute, role,
        gold, level, kills, deaths, assists, cs, hp, hp_ratio
    )]
}

calc_slope <- function(x, y) {
    ok <- is.finite(x) & is.finite(y)
    x <- x[ok]
    y <- y[ok]
    if (length(x) < 2L || length(unique(x)) < 2L) return(0)
    coef(lm(y ~ x))[2]
}

trapz_vec <- function(x, y) {
    ok <- is.finite(x) & is.finite(y)
    x <- x[ok]
    y <- y[ok]
    if (length(x) < 2L) return(0)
    ord <- order(x)
    x <- x[ord]
    y <- y[ord]
    sum(diff(x) * (head(y, -1L) + tail(y, -1L)) / 2)
}

longest_streak <- function(x, positive = TRUE) {
    flag <- if (positive) x > 0 else x < 0
    if (!length(flag) || !any(flag)) return(0L)
    runs <- rle(flag)
    max(runs$lengths[runs$values])
}

first_true_minute <- function(values, minutes, predicate) {
    idx <- which(predicate(values))[1]
    if (is.na(idx)) return(99)
    minutes[idx]
}

last_true_minute <- function(values, minutes, predicate) {
    idx <- tail(which(predicate(values)), 1)
    if (!length(idx)) return(99)
    minutes[idx]
}

closest_row <- function(dt, target_minute, tolerance = 2L) {
    if (!nrow(dt)) return(NULL)
    dd <- copy(dt)
    dd[, dist := abs(actual_minute - target_minute)]
    dd <- dd[dist == min(dist)]
    dd <- dd[order(dist, -as.integer(actual_minute <= target_minute), actual_minute)]
    if (!nrow(dd) || dd$dist[1] > tolerance) return(NULL)
    dd[1]
}

first_increment_time <- function(x, minutes, cutoff = 15L) {
    if (length(x) < 2L) return(99)
    diffs <- diff(x)
    idx <- which(diffs > 0)[1]
    if (is.na(idx)) return(99)
    minute <- minutes[idx + 1L]
    if (!is.finite(minute) || minute > cutoff) return(99)
    minute
}

lead_sign_changes <- function(x) {
    s <- sign(x)
    s <- s[s != 0]
    if (length(s) < 2L) return(0L)
    as.integer(sum(diff(s) != 0))
}

extract_game_rows <- function(frame_path, oracle_start) {
    lines <- readLines(frame_path, warn = FALSE)
    lines <- lines[nzchar(lines)]
    if (!length(lines)) return(list(team = data.table(), player = data.table(), meta = NULL))

    parsed <- lapply(lines, function(line) fromJSON(line, simplifyVector = FALSE))
    raw_minutes <- vapply(parsed, function(x) safe_num(x$minute), numeric(1))
    ts_vec <- as.POSIXct(
        vapply(parsed, function(x) x$ts %||% NA_character_, character(1)),
        tz = "UTC",
        format = "%Y-%m-%dT%H:%M:%OSZ"
    )

    actual_offset <- 0
    if (!is.na(oracle_start)) {
        diff_mins <- as.numeric(difftime(ts_vec, oracle_start, units = "mins"))
        actual_offset <- round(median(diff_mins - raw_minutes, na.rm = TRUE))
        if (!is.finite(actual_offset)) actual_offset <- 0
    }

    team_rows <- rbindlist(lapply(seq_along(parsed), function(i) {
        row <- parsed[[i]]
        actual_minute <- raw_minutes[i] + actual_offset
        rbindlist(list(
            team_snapshot(row$gid %||% sub("\\.ndjson$", "", basename(frame_path)), "Blue", row$ts %||% NA_character_, raw_minutes[i], actual_minute, row$blue %||% list()),
            team_snapshot(row$gid %||% sub("\\.ndjson$", "", basename(frame_path)), "Red",  row$ts %||% NA_character_, raw_minutes[i], actual_minute, row$red %||% list())
        ), fill = TRUE)
    }), fill = TRUE)

    player_rows <- rbindlist(lapply(seq_along(parsed), function(i) {
        row <- parsed[[i]]
        actual_minute <- raw_minutes[i] + actual_offset
        rbindlist(list(
            player_snapshot(row$gid %||% sub("\\.ndjson$", "", basename(frame_path)), "Blue", raw_minutes[i], actual_minute, row$blue %||% list()),
            player_snapshot(row$gid %||% sub("\\.ndjson$", "", basename(frame_path)), "Red",  raw_minutes[i], actual_minute, row$red %||% list())
        ), fill = TRUE)
    }), fill = TRUE)

    list(
        team = team_rows[order(side, actual_minute, raw_minute)],
        player = player_rows[order(side, actual_minute, raw_minute, role)],
        meta = list(actual_offset = actual_offset, n_lines = length(lines))
    )
}

build_features_for_side <- function(team_dt, player_dt, oracle_row) {
    my_side <- oracle_row$side[1]
    opp_side <- if (my_side == "Blue") "Red" else "Blue"

    my_team <- team_dt[side == my_side][order(actual_minute, raw_minute)]
    opp_team <- team_dt[side == opp_side][order(actual_minute, raw_minute)]
    if (!nrow(my_team) || !nrow(opp_team)) return(NULL)

    snap_targets <- c(5L, 10L, 15L)
    minute_targets <- 0:15
    snap_rows <- lapply(snap_targets, function(m) closest_row(my_team, m))
    opp_snap_rows <- lapply(snap_targets, function(m) closest_row(opp_team, m))
    names(snap_rows) <- names(opp_snap_rows) <- paste0("m", snap_targets)
    if (is.null(snap_rows$m15) || is.null(opp_snap_rows$m15)) return(NULL)

    lead <- merge(
        my_team[, .(actual_minute, gold, kills, cs_sum, level_sum, towers, dragons_count, barons, inhibitors, n_alive, hp_ratio_sum)],
        opp_team[, .(actual_minute, opp_gold = gold, opp_kills = kills, opp_cs = cs_sum, opp_level_sum = level_sum,
                     opp_towers = towers, opp_dragons = dragons_count, opp_barons = barons,
                     opp_inhibitors = inhibitors, opp_n_alive = n_alive, opp_hp_ratio_sum = hp_ratio_sum)],
        by = "actual_minute",
        all = FALSE
    )[actual_minute <= 15]

    if (!nrow(lead)) return(NULL)

    lead[, `:=`(
        gold_diff = gold - opp_gold,
        kill_diff = kills - opp_kills,
        cs_diff = cs_sum - opp_cs,
        level_diff = level_sum - opp_level_sum,
        tower_diff = towers - opp_towers,
        dragon_diff = dragons_count - opp_dragons,
        alive_diff = n_alive - opp_n_alive,
        hp_ratio_diff = hp_ratio_sum - opp_hp_ratio_sum
    )]

    minute_rows <- lapply(minute_targets, function(target) {
        me <- closest_row(my_team, target)
        op <- closest_row(opp_team, target)
        if (is.null(me) || is.null(op)) return(NULL)
        data.table(
            target_minute = target,
            actual_minute = me$actual_minute,
            minute_gap = abs(me$actual_minute - target),
            gold_diff = me$gold - op$gold,
            kill_diff = me$kills - op$kills,
            cs_diff = me$cs_sum - op$cs_sum,
            level_diff = me$level_sum - op$level_sum,
            tower_diff = me$towers - op$towers,
            dragon_diff = me$dragons_count - op$dragons_count,
            alive_diff = me$n_alive - op$n_alive,
            hp_ratio_diff = me$hp_ratio_sum - op$hp_ratio_sum,
            my_gold = me$gold,
            opp_gold = op$gold,
            my_kills = me$kills,
            opp_kills = op$kills
        )
    })
    minute_rows <- rbindlist(Filter(Negate(is.null), minute_rows), fill = TRUE)
    if (!nrow(minute_rows) || !any(minute_rows$target_minute == 15L)) return(NULL)

    my_minutes <- my_team$actual_minute
    opp_minutes <- opp_team$actual_minute

    events <- list(
        first_blood_me = first_increment_time(my_team$kills, my_minutes),
        first_blood_opp = first_increment_time(opp_team$kills, opp_minutes),
        first_dragon_me = first_increment_time(my_team$dragons_count, my_minutes),
        first_dragon_opp = first_increment_time(opp_team$dragons_count, opp_minutes),
        first_tower_me = first_increment_time(my_team$towers, my_minutes),
        first_tower_opp = first_increment_time(opp_team$towers, opp_minutes),
        first_baron_me = first_increment_time(my_team$barons, my_minutes),
        first_baron_opp = first_increment_time(opp_team$barons, opp_minutes)
    )

    out <- data.table(
        oracle_gameid = oracle_row$oracle_gameid[1],
        esports_game_id = oracle_row$esports_game_id[1],
        league = oracle_row$league[1],
        side = my_side,
        result = as.integer(oracle_row$result[1]),
        side_blue = as.integer(my_side == "Blue"),
        frame_group = oracle_row$frame_group[1]
    )

    for (i in seq_len(nrow(minute_rows))) {
        mr <- minute_rows[i]
        base <- paste0("m_", mr$target_minute)
        out[, paste0(base, "_actual_minute") := mr$actual_minute]
        out[, paste0(base, "_gap") := mr$minute_gap]
        out[, paste0(base, "_gold_diff") := mr$gold_diff]
        out[, paste0(base, "_kill_diff") := mr$kill_diff]
        out[, paste0(base, "_cs_diff") := mr$cs_diff]
        out[, paste0(base, "_level_diff") := mr$level_diff]
        out[, paste0(base, "_tower_diff") := mr$tower_diff]
        out[, paste0(base, "_dragon_diff") := mr$dragon_diff]
        out[, paste0(base, "_alive_diff") := mr$alive_diff]
        out[, paste0(base, "_hp_ratio_diff") := mr$hp_ratio_diff]
    }

    for (target in snap_targets) {
        me <- snap_rows[[paste0("m", target)]]
        op <- opp_snap_rows[[paste0("m", target)]]
        prefix <- paste0("f_", target)
        out[, paste0(prefix, "_actual_minute") := me$actual_minute]
        out[, paste0(prefix, "_minute_gap") := abs(me$actual_minute - target)]
        out[, paste0(prefix, "_gold") := me$gold]
        out[, paste0(prefix, "_gold_diff") := me$gold - op$gold]
        out[, paste0(prefix, "_kills") := me$kills]
        out[, paste0(prefix, "_kill_diff") := me$kills - op$kills]
        out[, paste0(prefix, "_cs") := me$cs_sum]
        out[, paste0(prefix, "_cs_diff") := me$cs_sum - op$cs_sum]
        out[, paste0(prefix, "_levels") := me$level_sum]
        out[, paste0(prefix, "_level_diff") := me$level_sum - op$level_sum]
        out[, paste0(prefix, "_towers") := me$towers]
        out[, paste0(prefix, "_tower_diff") := me$towers - op$towers]
        out[, paste0(prefix, "_dragons") := me$dragons_count]
        out[, paste0(prefix, "_dragon_diff") := me$dragons_count - op$dragons_count]
        out[, paste0(prefix, "_barons") := me$barons]
        out[, paste0(prefix, "_inhibitors") := me$inhibitors]
        out[, paste0(prefix, "_alive_diff") := me$n_alive - op$n_alive]
        out[, paste0(prefix, "_hp_ratio_diff") := me$hp_ratio_sum - op$hp_ratio_sum]
    }

    out[, `:=`(
        gold_momentum_5_10 = f_10_gold_diff - f_5_gold_diff,
        gold_momentum_10_15 = f_15_gold_diff - f_10_gold_diff,
        kill_momentum_5_10 = f_10_kill_diff - f_5_kill_diff,
        kill_momentum_10_15 = f_15_kill_diff - f_10_kill_diff,
        cs_momentum_5_10 = f_10_cs_diff - f_5_cs_diff,
        cs_momentum_10_15 = f_15_cs_diff - f_10_cs_diff,
        tower_momentum_10_15 = f_15_tower_diff - f_10_tower_diff,
        dragon_momentum_10_15 = f_15_dragon_diff - f_10_dragon_diff
    )]

    out[, `:=`(
        gold_diff_slope_0_15 = calc_slope(lead$actual_minute, lead$gold_diff),
        kill_diff_slope_0_15 = calc_slope(lead$actual_minute, lead$kill_diff),
        cs_diff_slope_0_15 = calc_slope(lead$actual_minute, lead$cs_diff),
        level_diff_slope_0_15 = calc_slope(lead$actual_minute, lead$level_diff),
        gold_adv_time_share = mean(lead$gold_diff > 0),
        gold_trail_time_share = mean(lead$gold_diff < 0),
        gold_lead_changes = lead_sign_changes(lead$gold_diff),
        max_gold_lead_0_15 = max(lead$gold_diff),
        max_gold_deficit_0_15 = min(lead$gold_diff),
        gold_diff_sd_0_15 = stats::sd(lead$gold_diff),
        kill_diff_sd_0_15 = stats::sd(lead$kill_diff),
        hp_ratio_diff_mean_0_15 = mean(lead$hp_ratio_diff),
        alive_diff_mean_0_15 = mean(lead$alive_diff)
    )]

    gold_delta_diff <- diff(minute_rows$gold_diff)
    kill_delta_diff <- diff(minute_rows$kill_diff)
    cs_delta_diff <- diff(minute_rows$cs_diff)
    level_delta_diff <- diff(minute_rows$level_diff)

    out[, `:=`(
        n_aligned_minutes = nrow(minute_rows),
        mean_minute_gap_0_15 = mean(minute_rows$minute_gap),
        max_minute_gap_0_15 = max(minute_rows$minute_gap),
        gold_auc_0_15 = trapz_vec(minute_rows$target_minute, minute_rows$gold_diff),
        gold_abs_auc_0_15 = trapz_vec(minute_rows$target_minute, abs(minute_rows$gold_diff)),
        kill_auc_0_15 = trapz_vec(minute_rows$target_minute, minute_rows$kill_diff),
        cs_auc_0_15 = trapz_vec(minute_rows$target_minute, minute_rows$cs_diff),
        level_auc_0_15 = trapz_vec(minute_rows$target_minute, minute_rows$level_diff),
        hp_ratio_auc_0_15 = trapz_vec(minute_rows$target_minute, minute_rows$hp_ratio_diff),
        gold_diff_mean_0_15 = mean(minute_rows$gold_diff),
        gold_diff_median_0_15 = median(minute_rows$gold_diff),
        gold_diff_range_0_15 = max(minute_rows$gold_diff) - min(minute_rows$gold_diff),
        kill_diff_range_0_15 = max(minute_rows$kill_diff) - min(minute_rows$kill_diff),
        cs_diff_range_0_15 = max(minute_rows$cs_diff) - min(minute_rows$cs_diff),
        level_diff_range_0_15 = max(minute_rows$level_diff) - min(minute_rows$level_diff),
        first_gold_lead_minute = first_true_minute(minute_rows$gold_diff, minute_rows$target_minute, function(v) v > 0),
        first_gold_trail_minute = first_true_minute(minute_rows$gold_diff, minute_rows$target_minute, function(v) v < 0),
        last_gold_lead_minute = last_true_minute(minute_rows$gold_diff, minute_rows$target_minute, function(v) v > 0),
        last_gold_trail_minute = last_true_minute(minute_rows$gold_diff, minute_rows$target_minute, function(v) v < 0),
        longest_gold_lead_streak = longest_streak(minute_rows$gold_diff, TRUE),
        longest_gold_trail_streak = longest_streak(minute_rows$gold_diff, FALSE),
        gold_positive_share_0_15 = mean(minute_rows$gold_diff > 0),
        gold_negative_share_0_15 = mean(minute_rows$gold_diff < 0),
        gold_diff_last3_mean = mean(tail(minute_rows$gold_diff, min(3L, nrow(minute_rows)))),
        gold_diff_last5_mean = mean(tail(minute_rows$gold_diff, min(5L, nrow(minute_rows)))),
        gold_diff_first5_mean = mean(head(minute_rows$gold_diff, min(5L, nrow(minute_rows)))),
        kill_diff_last5_mean = mean(tail(minute_rows$kill_diff, min(5L, nrow(minute_rows)))),
        cs_diff_last5_mean = mean(tail(minute_rows$cs_diff, min(5L, nrow(minute_rows)))),
        gold_delta_mean = if (length(gold_delta_diff)) mean(gold_delta_diff) else 0,
        gold_delta_sd = if (length(gold_delta_diff) > 1L) stats::sd(gold_delta_diff) else 0,
        gold_delta_max = if (length(gold_delta_diff)) max(gold_delta_diff) else 0,
        gold_delta_min = if (length(gold_delta_diff)) min(gold_delta_diff) else 0,
        gold_delta_positive_share = if (length(gold_delta_diff)) mean(gold_delta_diff > 0) else 0,
        kill_delta_mean = if (length(kill_delta_diff)) mean(kill_delta_diff) else 0,
        kill_delta_sd = if (length(kill_delta_diff) > 1L) stats::sd(kill_delta_diff) else 0,
        cs_delta_mean = if (length(cs_delta_diff)) mean(cs_delta_diff) else 0,
        cs_delta_sd = if (length(cs_delta_diff) > 1L) stats::sd(cs_delta_diff) else 0,
        level_delta_mean = if (length(level_delta_diff)) mean(level_delta_diff) else 0,
        level_delta_sd = if (length(level_delta_diff) > 1L) stats::sd(level_delta_diff) else 0,
        minute_of_max_gold_lead = minute_rows$target_minute[which.max(minute_rows$gold_diff)],
        minute_of_max_gold_deficit = minute_rows$target_minute[which.min(minute_rows$gold_diff)],
        minute_of_max_kill_lead = minute_rows$target_minute[which.max(minute_rows$kill_diff)],
        minute_of_max_cs_lead = minute_rows$target_minute[which.max(minute_rows$cs_diff)],
        close_game_share_0_15 = mean(abs(minute_rows$gold_diff) <= 500),
        contested_game_share_0_15 = mean(abs(minute_rows$gold_diff) <= 1000)
    )]

    out[, `:=`(
        got_first_blood = as.integer(events$first_blood_me < events$first_blood_opp),
        got_first_dragon = as.integer(events$first_dragon_me < events$first_dragon_opp),
        got_first_tower = as.integer(events$first_tower_me < events$first_tower_opp),
        got_first_baron = as.integer(events$first_baron_me < events$first_baron_opp),
        first_blood_me = events$first_blood_me,
        first_dragon_me = events$first_dragon_me,
        first_tower_me = events$first_tower_me,
        first_baron_me = events$first_baron_me,
        first_blood_opp = events$first_blood_opp,
        first_dragon_opp = events$first_dragon_opp,
        first_tower_opp = events$first_tower_opp,
        first_baron_opp = events$first_baron_opp
    )]
    out[, `:=`(
        first_blood_gap = first_blood_me - first_blood_opp,
        first_dragon_gap = first_dragon_me - first_dragon_opp,
        first_tower_gap = first_tower_me - first_tower_opp
    )]

    my_dragon_types <- strsplit(snap_rows$m15$dragon_types %||% "", "\\|", fixed = FALSE)[[1]]
    opp_dragon_types <- strsplit(opp_snap_rows$m15$dragon_types %||% "", "\\|", fixed = FALSE)[[1]]
    dragon_names <- c("chemtech", "cloud", "hextech", "infernal", "mountain", "ocean")
    for (dragon_name in dragon_names) {
        out[, paste0("drag_", dragon_name, "_diff_15") := sum(my_dragon_types == dragon_name) - sum(opp_dragon_types == dragon_name)]
    }

    for (target in c(10L, 15L)) {
        my_target_row <- closest_row(my_team, target)
        opp_target_row <- closest_row(opp_team, target)
        if (is.null(my_target_row) || is.null(opp_target_row)) next

        my_players <- player_dt[side == my_side & actual_minute == my_target_row$actual_minute]
        opp_players <- player_dt[side == opp_side & actual_minute == opp_target_row$actual_minute]
        if (!nrow(my_players) || !nrow(opp_players)) next
        merged_players <- merge(
            my_players[, .(role, gold, level, cs, kills, deaths, assists, hp_ratio)],
            opp_players[, .(role, opp_gold = gold, opp_level = level, opp_cs = cs,
                            opp_kills = kills, opp_deaths = deaths, opp_assists = assists, opp_hp_ratio = hp_ratio)],
            by = "role",
            all = TRUE
        )
        for (i in seq_len(nrow(merged_players))) {
            role <- merged_players$role[i]
            base <- paste0(role, "_", target)
            out[, paste0(base, "_gold_diff") := merged_players$gold[i] - merged_players$opp_gold[i]]
            out[, paste0(base, "_level_diff") := merged_players$level[i] - merged_players$opp_level[i]]
            out[, paste0(base, "_cs_diff") := merged_players$cs[i] - merged_players$opp_cs[i]]
            out[, paste0(base, "_kda_diff") := (
                (merged_players$kills[i] + merged_players$assists[i]) / pmax(merged_players$deaths[i], 1)
            ) - (
                (merged_players$opp_kills[i] + merged_players$opp_assists[i]) / pmax(merged_players$opp_deaths[i], 1)
            )]
            out[, paste0(base, "_hp_ratio_diff") := merged_players$hp_ratio[i] - merged_players$opp_hp_ratio[i]]
        }
    }

    roles <- c("top", "jng", "mid", "bot", "sup")
    col_or_zero <- function(dt, name) if (name %in% names(dt)) dt[[name]][1] else 0
    for (role_name in roles) {
        role_rows <- merge(
            player_dt[side == my_side & role == role_name, .(actual_minute, gold, level, cs, kills, deaths, assists, hp_ratio)],
            player_dt[side == opp_side & role == role_name, .(actual_minute, opp_gold = gold, opp_level = level, opp_cs = cs,
                                                           opp_kills = kills, opp_deaths = deaths, opp_assists = assists, opp_hp_ratio = hp_ratio)],
            by = "actual_minute",
            all = FALSE
        )[actual_minute <= 15]
        if (!nrow(role_rows)) next
        role_rows[, `:=`(
            gold_diff = gold - opp_gold,
            cs_diff = cs - opp_cs,
            level_diff = level - opp_level,
            hp_ratio_diff = hp_ratio - opp_hp_ratio,
            kda_diff = ((kills + assists) / pmax(deaths, 1)) - ((opp_kills + opp_assists) / pmax(opp_deaths, 1))
        )]
        out[, paste0(role_name, "_gold_diff_slope_0_15") := calc_slope(role_rows$actual_minute, role_rows$gold_diff)]
        out[, paste0(role_name, "_gold_auc_0_15") := trapz_vec(role_rows$actual_minute, role_rows$gold_diff)]
        out[, paste0(role_name, "_gold_sd_0_15") := if (nrow(role_rows) > 1L) stats::sd(role_rows$gold_diff) else 0]
        out[, paste0(role_name, "_gold_lead_share_0_15") := mean(role_rows$gold_diff > 0)]
        out[, paste0(role_name, "_cs_diff_slope_0_15") := calc_slope(role_rows$actual_minute, role_rows$cs_diff)]
        out[, paste0(role_name, "_level_diff_slope_0_15") := calc_slope(role_rows$actual_minute, role_rows$level_diff)]
        out[, paste0(role_name, "_hp_ratio_mean_0_15") := mean(role_rows$hp_ratio_diff)]
        out[, paste0(role_name, "_kda_mean_0_15") := mean(role_rows$kda_diff)]
    }

    role_gold_15 <- c(
        col_or_zero(out, "top_15_gold_diff"),
        col_or_zero(out, "jng_15_gold_diff"),
        col_or_zero(out, "mid_15_gold_diff"),
        col_or_zero(out, "bot_15_gold_diff"),
        col_or_zero(out, "sup_15_gold_diff")
    )
    role_gold_10 <- c(
        col_or_zero(out, "top_10_gold_diff"),
        col_or_zero(out, "jng_10_gold_diff"),
        col_or_zero(out, "mid_10_gold_diff"),
        col_or_zero(out, "bot_10_gold_diff"),
        col_or_zero(out, "sup_10_gold_diff")
    )
    out[, `:=`(
        role_gold_diff_sd_10 = stats::sd(role_gold_10),
        role_gold_diff_sd_15 = stats::sd(role_gold_15),
        role_gold_diff_max_15 = max(role_gold_15),
        role_gold_diff_min_15 = min(role_gold_15),
        carry_gold_diff_15 = col_or_zero(out, "top_15_gold_diff") + col_or_zero(out, "mid_15_gold_diff") + col_or_zero(out, "bot_15_gold_diff"),
        solo_lane_gold_diff_15 = col_or_zero(out, "top_15_gold_diff") + col_or_zero(out, "mid_15_gold_diff"),
        jungle_support_gold_diff_15 = col_or_zero(out, "jng_15_gold_diff") + col_or_zero(out, "sup_15_gold_diff")
    )]

    oracle_keep <- c(
        "goldat10", "goldat15", "xpat10", "xpat15", "csat10", "csat15",
        "golddiffat10", "golddiffat15", "xpdiffat10", "xpdiffat15",
        "csdiffat10", "csdiffat15", "killsat10", "killsat15",
        "deathsat10", "deathsat15", "assistsat10", "assistsat15",
        "opp_killsat10", "opp_killsat15", "opp_deathsat10", "opp_deathsat15",
        "opp_assistsat10", "opp_assistsat15", "firstblood", "firstdragon",
        "firstherald", "firsttower", "void_grubs", "opp_void_grubs",
        "turretplates", "opp_turretplates"
    )
    for (col in oracle_keep) {
        out[, paste0("oracle_", col) := safe_num(oracle_row[[col]][1], default = NA_real_)]
    }

    out
}

main <- function() {
    cat("Loading mappings and frame index...\n")
    mappings <- read_mapping_files()
    frame_idx <- build_frame_index()
    oracle <- load_oracle_team_rows()

    mapped <- merge(mappings, frame_idx, by = "esports_game_id", all = FALSE)
    setorder(mapped, oracle_gameid, esports_game_id, source)
    mapped <- unique(mapped, by = "oracle_gameid")

    oracle_labeled <- merge(
        oracle,
        mapped,
        by.x = "oracle_gameid",
        by.y = "oracle_gameid",
        all = FALSE
    )

    games <- unique(oracle_labeled[, .(
        oracle_gameid, esports_game_id, league, frame_group, frame_path, oracle_date
    )])
    setorder(games, oracle_gameid)

    cat(sprintf("Oracle team rows with mapped frame file: %d\n", uniqueN(oracle_labeled$oracle_gameid)))

    configured_cores <- suppressWarnings(as.integer(Sys.getenv("AR_N_CORES", "14")))
    detected_cores <- detectCores(logical = TRUE)
    if (is.finite(configured_cores) && configured_cores >= 1L) {
        n_cores <- configured_cores
    } else if (is.finite(detected_cores) && detected_cores >= 1L) {
        n_cores <- as.integer(detected_cores)
    } else {
        n_cores <- 14L
    }
    cat(sprintf("Using %d cores for per-game extraction.\n", n_cores))

    process_game <- function(i) {
        game <- games[i]
        tryCatch({
            oracle_start <- parse_time_utc(game$oracle_date)
            extracted <- extract_game_rows(game$frame_path, oracle_start)
            if (is.null(extracted) || !nrow(extracted$team)) {
                return(list(ok = TRUE, data = NULL))
            }

            game_oracle_rows <- oracle_labeled[oracle_gameid == game$oracle_gameid]
            side_features <- lapply(split(game_oracle_rows, game_oracle_rows$side), function(orow) {
                build_features_for_side(extracted$team, extracted$player, as.data.table(orow))
            })
            side_features <- Filter(Negate(is.null), side_features)
            if (length(side_features) != 2L) {
                return(list(ok = TRUE, data = NULL))
            }

            list(ok = TRUE, data = rbindlist(side_features, fill = TRUE))
        }, error = function(e) {
            list(ok = FALSE, oracle_gameid = game$oracle_gameid, msg = conditionMessage(e))
        })
    }

    chunk_size <- 100L
    index_chunks <- split(seq_len(nrow(games)), ceiling(seq_len(nrow(games)) / chunk_size))
    feature_rows <- vector("list", length(index_chunks))

    for (chunk_id in seq_along(index_chunks)) {
        idx <- index_chunks[[chunk_id]]
        chunk_rows <- mclapply(idx, process_game, mc.cores = n_cores)
        errors <- Filter(function(x) is.list(x) && identical(x$ok, FALSE), chunk_rows)
        if (length(errors)) {
            stop(sprintf("Worker error on game %s: %s", errors[[1]]$oracle_gameid, errors[[1]]$msg))
        }
        good_rows <- lapply(Filter(function(x) is.list(x) && identical(x$ok, TRUE) && !is.null(x$data), chunk_rows), `[[`, "data")
        feature_rows[[chunk_id]] <- if (length(good_rows)) rbindlist(good_rows, fill = TRUE) else NULL
        cat(sprintf("Completed chunk %d / %d (%d / %d games)\n",
                    chunk_id, length(index_chunks), max(idx), nrow(games)))
    }

    features <- rbindlist(Filter(function(x) !is.null(x) && nrow(x) > 0L, feature_rows), fill = TRUE)
    kept <- uniqueN(features$oracle_gameid)

    if (kept == 0L) {
        stop("No games survived feature extraction.")
    }

    lead_cols <- c("oracle_gameid", "esports_game_id", "league", "frame_group", "side", "result", "side_blue")
    setcolorder(features, c(lead_cols, setdiff(names(features), lead_cols)))

    fwrite(features, OUT_CSV, na = "NA")
    saveRDS(features, OUT_RDS)

    summary_lines <- c(
        sprintf("rows=%d", nrow(features)),
        sprintf("games=%d", uniqueN(features$oracle_gameid)),
        sprintf("features=%d", ncol(features) - 7L),
        sprintf("positive_rate=%.4f", mean(features$result)),
        sprintf("frame_groups=%d", uniqueN(features$frame_group)),
        sprintf("leagues=%d", uniqueN(features$league)),
        sprintf("mapped_games_with_frames=%d", nrow(games)),
        sprintf("kept_games=%d", kept)
    )
    writeLines(summary_lines, OUT_SUMMARY)

    cat("\nDataset build complete.\n")
    cat(paste(summary_lines, collapse = "\n"), "\n")
}

main()
