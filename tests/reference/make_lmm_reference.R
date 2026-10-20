#!/usr/bin/env Rscript
# Reference values for csvstats.lmm's Satterthwaite degrees of freedom.
#
# Run ONCE (and again only on purpose, noting it in the commit):
#
#   install.packages(c("lme4", "lmerTest", "emmeans", "pbkrtest", "jsonlite"))
#
#   From a terminal:   Rscript tests/reference/make_lmm_reference.R
#   From R / RStudio:  source("<path to csv-stats>/tests/reference/make_lmm_reference.R")
#
# For every scenario below this script GENERATES the dataset itself (fixed
# seed per scenario), fits it with lme4 + lmerTest, and writes:
#
#   datasets/<id>.csv   the long data, one row per observation
#   results/<id>.json   lme4/lmerTest/emmeans results + R package versions
#
# Python's tests (tests/test_lmm_reference.py) read both, so Python and R
# never have to agree on random numbers. The scenarios cover every dimension
# that changes a Satterthwaite df: design shape, missing-data pattern,
# sample size, how much of the variance is between subjects (ICC, including
# exactly 0 = a singular fit), null vs real effects, numeric scale, odd level
# labels and level order, and (marked "future") random slopes.
#
# Plan: .claude/plan-lmm-satterthwaite.md in the scistack repo. The fit ->
# JSON code is shared with make_lmm_scenarios.R in lmm_reference_common.R.

# The folder this script lives in, whether it was started with Rscript
# (--file=) or with source() from the R console / RStudio (ofile). Outputs
# always go next to the script, never into whatever folder R happens to be in.
# (This bootstrap is how lmm_reference_common.R is found, so it can't live
# there.)
script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) return(dirname(normalizePath(f)))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  getwd()
})
source(file.path(script_dir, "lmm_reference_common.R"))
data_dir <- file.path(script_dir, "datasets")
res_dir <- file.path(script_dir, "results")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Design shapes. `between` / `within`: factor -> levels IN ORDER. `time`:
# the within factor that monotone dropout follows. `reps`: observations per
# subject-cell.
# ---------------------------------------------------------------------------
DESIGNS <- list(
  W2 = list(between = list(), within = list(time = c("pre", "post")), time = "time", reps = 1),
  W3 = list(between = list(), within = list(time = c("t1", "t2", "t3")), time = "time", reps = 1),
  # W3 with 3 observations per subject per level. Random slopes need this:
  # with one observation per cell, (1 + time | subject) has as many random
  # effects as observations and is unidentifiable (first run, 2026-10-19).
  # Subjects also differ in their time slope (slope_sd per level step), so a
  # random-slope model has a real slope variance to estimate (a first version
  # without this gave a singular fit at 0, 2026-10-19).
  W3rep = list(between = list(), within = list(time = c("t1", "t2", "t3")), time = "time", reps = 3,
               slope_sd = 0.4),
  W7 = list(between = list(), within = list(session = paste0("s", 1:7)), time = "session", reps = 1),
  W2xW3 = list(between = list(), within = list(speed = c("slow", "fast"), session = c("s1", "s2", "s3")),
               time = "session", reps = 1),
  B2W3 = list(between = list(group = c("ctl", "trt")), within = list(time = c("t1", "t2", "t3")),
              time = "time", reps = 1),
  B3W2 = list(between = list(group = c("g1", "g2", "g3")), within = list(time = c("pre", "post")),
              time = "time", reps = 1),
  B2B2W2 = list(between = list(group = c("ctl", "trt"), site = c("A", "B")),
                within = list(time = c("pre", "post")), time = "time", reps = 1),
  # Between-only, but each subject observed 3 times: the random intercept is
  # still identifiable.
  B2rep = list(between = list(group = c("ctl", "trt")), within = list(), time = NA, reps = 3),
  # Level labels with spaces and symbols, in an order that is NOT
  # alphabetical (alphabetical would put "12 wk" first and "A-arm" first).
  LABELS = list(between = list(arm = c("Z-arm", "A-arm")),
                within = list(time_point = c("pre op", "6 wk", "12 wk")), time = "time_point", reps = 1)
)

all_levels <- function(d) c(d$between, d$within)
factors_of <- function(d) c(names(d$between), names(d$within))

# ---------------------------------------------------------------------------
# Scenario list
# ---------------------------------------------------------------------------
SCENARIOS <- list()
add <- function(design, missing = "complete", n = 17, icc = 0.5, effect = "effect",
                scale = "unit", slopes = FALSE) {
  id <- paste(design, missing, paste0("n", n), paste0("icc", round(icc * 100)), effect, scale,
              if (slopes) "slopes" else "intercept", sep = "_")
  if (!is.null(SCENARIOS[[id]])) return(invisible(NULL))
  # A slopes scenario shares its intercept-only twin's seed, so the two
  # are fitted to the SAME data (the twin is the baseline).
  codes <- utf8ToInt(sub("_slopes$", "_intercept", id))
  seed <- sum(codes * seq_along(codes)) %% .Machine$integer.max
  SCENARIOS[[id]] <<- list(id = id, design = design, missing = missing, n = n, icc = icc,
                           effect = effect, scale = scale, slopes = slopes, future = slopes, seed = seed)
}

has_between <- function(design) length(DESIGNS[[design]]$between) > 0
has_within <- function(design) length(DESIGNS[[design]]$within) > 0

# 1. Every design shape x {complete, MCAR 15%, monotone 15%}.
for (des in setdiff(names(DESIGNS), "LABELS")) {
  if (has_within(des)) {
    for (m in c("complete", "mcar15", "monotone15")) add(des, m)
  } else {
    add(des, "complete")
    add(des, "unequal_reps")
  }
}
# 2. W2, W3, B2W3 x every missing-data pattern.
PATTERNS <- c("complete", "mcar05", "mcar15", "mcar30", "monotone15", "monotone40",
              "singletons", "unequal_reps", "sparse_cell")
for (des in c("W2", "W3", "B2W3")) {
  for (m in PATTERNS) add(des, m)
  if (has_between(des)) for (m in c("level_missing_one_group", "unequal_groups")) add(des, m)
}
# 3. W3, B2W3 x every sample size x {complete, monotone 15%}.
for (des in c("W3", "B2W3")) for (n in c(6, 12, 17, 40)) for (m in c("complete", "monotone15")) add(des, m, n = n)
# 4. W3 x every ICC (0 = identical subject means: a singular fit) x {complete, MCAR 15%}.
for (icc in c(0.8, 0.5, 0.1, 0)) for (m in c("complete", "mcar15")) add("W3", m, icc = icc)
# 5. Null vs real effects where interactions exist.
for (des in c("W2xW3", "B2W3")) for (eff in c("null", "effect")) for (m in c("complete", "monotone40")) add(des, m, effect = eff)
# 6. Numeric scale, and odd labels / level order.
for (sc in c("small", "large")) for (m in c("complete", "mcar15")) add("W3", m, scale = sc)
for (m in c("complete", "monotone15")) add("LABELS", m)
# 7. Random slopes (future scope: csv-stats has intercept-only models today),
# on replicated data, plus the same data intercept-only as a baseline.
for (m in c("complete", "mcar15")) for (sl in c(TRUE, FALSE)) add("W3rep", m, slopes = sl)

remove_stale(c(data_dir, res_dir), names(SCENARIOS))

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
generate <- function(sc) {
  d <- DESIGNS[[sc$design]]
  set.seed(sc$seed)
  n <- sc$n
  subjects <- sprintf("S%02d", seq_len(n))
  sub <- data.frame(subject = subjects, stringsAsFactors = FALSE)
  if (length(d$between)) {
    bgrid <- expand.grid(d$between, stringsAsFactors = FALSE)
    nb <- nrow(bgrid)
    w <- if (sc$missing == "unequal_groups") c(1, rep(3, nb - 1)) else rep(1, nb)
    target <- n * w / sum(w)
    sizes <- pmax(2, floor(target))
    while (sum(sizes) < n) { i <- which.max(target - sizes); sizes[i] <- sizes[i] + 1 }
    while (sum(sizes) > n) { i <- which.max(sizes - target); sizes[i] <- sizes[i] - 1 }
    sub <- cbind(sub, bgrid[rep(seq_len(nb), sizes), , drop = FALSE])
    rownames(sub) <- NULL
  }
  wgrid <- if (length(d$within)) expand.grid(d$within, stringsAsFactors = FALSE) else data.frame(.w = 1)
  cells <- merge(sub, wgrid, by = NULL)
  cells$.w <- NULL
  reps <- if (sc$missing == "unequal_reps") sample(1:4, nrow(cells), replace = TRUE) else rep(d$reps, nrow(cells))
  data <- cells[rep(seq_len(nrow(cells)), reps), , drop = FALSE]
  rownames(data) <- NULL

  facs <- factors_of(d)
  lv <- all_levels(d)
  k <- lapply(facs, function(f) match(data[[f]], lv[[f]]) - 1)
  names(k) <- facs
  lin <- rep(0, nrow(data))
  if (sc$effect == "effect") {
    for (f in facs) lin <- lin + 0.5 * k[[f]]
    if (length(facs) >= 2) lin <- lin + 0.25 * k[[facs[1]]] * k[[facs[2]]]
  }
  u <- setNames(rnorm(n, 0, sqrt(sc$icc)), subjects)
  e <- rnorm(nrow(data), 0, sqrt(1 - sc$icc))
  # ICC exactly 0: centre the noise within each subject, so subject means
  # differ only by the fixed effects -> the subject variance is 0 (boundary).
  if (sc$icc == 0) e <- e - ave(e, data$subject)
  # Drawn AFTER u and e, so designs without slope_sd keep identical data.
  if (!is.null(d$slope_sd)) {
    slope <- setNames(rnorm(n, 0, d$slope_sd), subjects)
    lin <- lin + slope[data$subject] * k[[d$time]]
    # The slope is LINEAR in the level index, so the random-slope model uses
    # this numeric covariate: (1 + time_num | subject), a 2x2 covariance.
    # Fitting (1 + time | subject) with time as a 3-level factor asks for a
    # 3x3 covariance the data can't support: singular by construction.
    data$time_num <- k[[d$time]]
  }
  y <- 10 + lin + u[data$subject] + e
  y <- switch(sc$scale, unit = y, small = y * 1e-3, large = 1e4 + 100 * y)
  data$y <- as.numeric(y)
  apply_missing(data, sc, d)
}

apply_missing <- function(data, sc, d) {
  pat <- sc$missing
  facs <- factors_of(d)
  cellkey <- do.call(paste, c(data[c("subject", names(d$within))], sep = "\r"))
  drop <- rep(FALSE, nrow(data))
  all_subjects <- unique(data$subject)
  if (startsWith(pat, "mcar")) {
    p <- as.numeric(sub("mcar", "", pat)) / 100
    u <- unique(cellkey)
    drop <- cellkey %in% sample(u, round(p * length(u)))
  } else if (startsWith(pat, "monotone")) {
    # Later levels missing for a fraction of subjects (longitudinal dropout).
    p <- as.numeric(sub("monotone", "", pat)) / 100
    tf <- d$time
    L <- d$within[[tf]]
    for (s in sample(all_subjects, round(p * length(all_subjects)))) {
      cut <- sample(seq_len(length(L) - 1), 1)
      drop <- drop | (data$subject == s & match(data[[tf]], L) > cut)
    }
  } else if (pat == "level_missing_one_group") {
    g <- names(d$between)[1]
    grp <- unique(data$subject[data[[g]] == d$between[[g]][1]])
    who <- sample(grp, max(1, round(0.3 * length(grp))))
    drop <- data$subject %in% who & data[[d$time]] == tail(d$within[[d$time]], 1)
  } else if (pat == "singletons") {
    # A fifth of subjects keep exactly one observation.
    for (s in sample(all_subjects, max(1, round(0.2 * length(all_subjects))))) {
      mine <- which(data$subject == s)
      drop[setdiff(mine, mine[sample.int(length(mine), 1)])] <- TRUE
    }
  } else if (pat == "sparse_cell") {
    # The last design cell keeps only 2 subjects.
    dk <- do.call(paste, c(data[facs], sep = "\r"))
    target <- tail(sort(unique(dk)), 1)
    keep2 <- head(sort(unique(data$subject[dk == target])), 2)
    drop <- dk == target & !(data$subject %in% keep2)
  }
  removed <- data[drop, , drop = FALSE]
  kept <- data[!drop, , drop = FALSE]
  # Guards, so every scenario is a model lme4 can fit as specified:
  # every subject keeps >= 1 row; every design cell keeps >= 2 subjects.
  for (s in setdiff(all_subjects, unique(kept$subject))) {
    kept <- rbind(kept, removed[removed$subject == s, , drop = FALSE][1, , drop = FALSE])
  }
  if (length(facs)) {
    dkey <- function(df) do.call(paste, c(df[facs], sep = "\r"))
    for (cell in unique(dkey(data))) {
      have <- unique(kept$subject[dkey(kept) == cell])
      if (length(have) < 2) {
        pool <- removed[dkey(removed) == cell & !(removed$subject %in% have), , drop = FALSE]
        add_s <- head(unique(pool$subject), 2 - length(have))
        kept <- rbind(kept, pool[pool$subject %in% add_s, , drop = FALSE])
      }
    }
  }
  kept <- kept[order(kept$subject), , drop = FALSE]
  rownames(kept) <- NULL
  kept
}

# ---------------------------------------------------------------------------
# Fitting and extraction
# ---------------------------------------------------------------------------
# finite_or_na, contrasts_for, fit_quietly, describe_* and versions live in
# lmm_reference_common.R. The JSON keys and their order are unchanged, so the
# committed results reproduce byte for byte.
fit_and_describe <- function(sc, data) {
  d <- DESIGNS[[sc$design]]
  facs <- factors_of(d)
  lv <- all_levels(d)
  for (f in facs) data[[f]] <- factor(data[[f]], levels = lv[[f]])
  data$subject <- factor(data$subject)
  re <- if (isTRUE(sc$slopes)) "(1 + time_num | subject)" else "(1 | subject)"
  form <- as.formula(paste("y ~", paste(facs, collapse = " * "), "+", re))
  q <- fit_quietly(form, data)
  c(
    list(formula = paste(deparse(form), collapse = " "), coding = "contr.sum"),
    describe_fixed(q$fit),
    describe_contrasts(q$fit, facs, lv),
    describe_variance(q$fit, q$notes)
  )
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cat(sprintf("%d scenarios -> %s\n", length(SCENARIOS), script_dir))
status <- c(ok = 0, noted = 0, failed = 0)
for (sc in SCENARIOS) {
  d <- DESIGNS[[sc$design]]
  data <- generate(sc)
  write.csv(data, file.path(data_dir, paste0(sc$id, ".csv")), row.names = FALSE)
  out <- list(
    id = sc$id, scenario = sc[setdiff(names(sc), "id")],
    # I(): always JSON arrays. Under auto_unbox a one-element vector became a
    # bare string and names(list()) became null (first fixtures, 2026-10-19;
    # the Python reader tolerates both).
    factors = I(as.character(factors_of(d))), between = I(as.character(names(d$between))),
    within = I(as.character(names(d$within))),
    levels = all_levels(d), subject = "subject", dv = "y",
    # The numeric covariate a random slope uses (null for intercept-only).
    random_slope = if (isTRUE(sc$slopes)) "time_num" else NULL,
    n_rows = nrow(data), n_subjects = length(unique(data$subject)),
    versions = versions
  )
  res <- tryCatch(fit_and_describe(sc, data), error = function(e) list(error = conditionMessage(e)))
  out <- c(out, res)
  write_result(out, file.path(res_dir, paste0(sc$id, ".json")))
  if (!is.null(res$error)) {
    status["failed"] <- status["failed"] + 1
    cat(sprintf("FAILED %-55s %s\n", sc$id, res$error))
  } else if (length(res$notes) || isTRUE(res$singular)) {
    status["noted"] <- status["noted"] + 1
    cat(sprintf("noted  %-55s singular=%s %s\n", sc$id, res$singular, paste(unlist(res$notes), collapse = " | ")))
  } else {
    status["ok"] <- status["ok"] + 1
    cat(sprintf("ok     %-55s rows=%d subjects=%d\n", sc$id, nrow(data), out$n_subjects))
  }
}
cat(sprintf("\nDone: %d ok, %d with notes (singular fits / warnings, expected for some), %d failed\n",
            status["ok"], status["noted"], status["failed"]))
cat("Commit tests/reference/datasets and tests/reference/results (or send them back).\n")
