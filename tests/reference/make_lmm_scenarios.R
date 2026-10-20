#!/usr/bin/env Rscript
# The SAMPLED LMM scenario set: R (lme4 / lmerTest) as the oracle for
# csv-stats' mixed models beyond one random intercept.
#
# Run ONCE (and again only on purpose, noting it in the commit):
#
#   install.packages(c("lme4", "lmerTest", "emmeans", "pbkrtest", "jsonlite"))
#
#   From a terminal:   Rscript tests/reference/make_lmm_scenarios.R
#   From R / RStudio:  source("<path to csv-stats>/tests/reference/make_lmm_scenarios.R")
#
# Writes, next to this script:
#   scenarios/datasets/<id>.csv   the long data, one row per observation
#   scenarios/results/<id>.json   the scenario spec + lme4/lmerTest results
#   scenarios/manifest.csv        one row per scenario: every sampled
#                                 dimension + R's verdict (coverage at a glance)
#
# 8 random-effect structures x 25 scenarios. Each scenario's dimensions are
# DRAWN from a fixed seed (MASTER_SEED + structure + index), stratified so the
# hard cases are always there: balanced data, a zero true variance, a
# near-+-1 correlation, small n, heavy missingness, a non-unit scale and an
# exactly collinear fixed-effect column. The curated, hand-named set
# (make_lmm_reference.R) is separate and unchanged.
#
# Beyond the curated JSON, each result records what a general fit needs to be
# checked independently of any optimizer:
#   probes      lme4's REML and ML deviance (devFunOnly) at fixed theta, so
#               the OBJECTIVE is compared exactly, with no optimizer involved
#   optimizers  refits with 4 optimizers; their spread sets the per-scenario
#               tolerance on estimates (measured, not guessed)
#   theta, theta_lower, fixed_terms, fixed_columns, dropped_columns, vcov_beta
#
# Plan: .claude/plan-lmm-scenario-generator.md in the scistack repo.
# Doc:  docs/claude/lmm-r-oracle-testing.md in the scistack repo.

# The folder this script lives in, whether it was started with Rscript
# (--file=) or with source() from the R console / RStudio (ofile). (This
# bootstrap is how lmm_reference_common.R is found, so it can't live there.)
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
out_dir <- file.path(script_dir, "scenarios")
data_dir <- file.path(out_dir, "datasets")
res_dir <- file.path(out_dir, "results")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

MASTER_SEED <- "lmm-scenarios-2026-10-19"
PER_STRUCTURE <- 25L

# ---------------------------------------------------------------------------
# Fixed-effect designs. `between` / `within`: factor -> levels IN ORDER.
# `covariate`: a numeric x, drawn per observation (so it varies within
# subject and a random slope on it is identifiable).
# ---------------------------------------------------------------------------
GROUP <- list(group = c("ctl", "trt"))
TIME <- list(time = c("t1", "t2", "t3"))
COND <- list(cond = c("A", "B"))
FIXED <- list(
  none    = list(between = list(), within = list(), covariate = FALSE),
  B2      = list(between = GROUP, within = list(), covariate = FALSE),
  W3      = list(between = list(), within = TIME, covariate = FALSE),
  B2W3    = list(between = GROUP, within = TIME, covariate = FALSE),
  W2xW3   = list(between = list(), within = list(speed = c("slow", "fast"), session = c("s1", "s2", "s3")),
                 covariate = FALSE),
  x       = list(between = list(), within = list(), covariate = TRUE),
  B2_x    = list(between = GROUP, within = list(), covariate = TRUE),
  cond    = list(between = list(), within = COND, covariate = FALSE),
  B2_cond = list(between = GROUP, within = COND, covariate = FALSE)
)

# ---------------------------------------------------------------------------
# Random-effect structures. `random`: one block per (... | g) term, in the
# language-neutral form the JSON records (and Python reads; never the R
# formula). `group` with several names means their interaction. `reps`: the
# range of observations per subject-cell. `crossed`: subject x item rows
# (cond, when present, assigned by a Latin square). `nested`: subjects in
# sites.
# ---------------------------------------------------------------------------
block <- function(group, terms = "(Intercept)", correlated = TRUE) list(group = group, terms = terms, correlated = correlated)
STRUCTURES <- list(
  RI             = list(random = list(block("subject")),
                        fixed = c("none", "B2", "W3", "B2W3", "W2xW3", "x", "B2_x"), reps = 1:4),
  RIS            = list(random = list(block("subject", c("(Intercept)", "x"))), fixed = c("x", "B2_x"), reps = 3:6),
  RIS_UNCORR     = list(random = list(block("subject", c("(Intercept)", "x"), correlated = FALSE)),
                        fixed = c("x", "B2_x"), reps = 3:6),
  RIF            = list(random = list(block("subject", c("(Intercept)", "cond"))), fixed = c("cond", "B2_cond"), reps = 2:5),
  RM             = list(random = list(block("subject"), block(c("subject", "time"))), fixed = c("W3", "B2W3"), reps = 2:4),
  CROSSED        = list(random = list(block("subject"), block("item")), fixed = c("none", "cond", "x"), reps = 1L,
                        crossed = TRUE),
  NESTED         = list(random = list(block("site"), block(c("site", "subject"))), fixed = c("W3", "B2W3", "x"),
                        reps = 1:3, nested = TRUE),
  CROSSED_SLOPES = list(random = list(block("subject", c("(Intercept)", "cond")), block("item")), fixed = "cond",
                        reps = 1L, crossed = TRUE)
)

# Deterministic strata by index, so the quotas hold by construction (the
# Python tests check them against the manifest). "corr_boundary" only means
# something for a block with a correlation; elsewhere it is a free draw.
STRATA <- c(rep("balanced", 3), rep("zero_variance", 4), rep("corr_boundary", 3), rep("small_n", 3),
            rep("heavy_missing", 3), rep("scale", 2), "collinear", rep("free", 6))
stopifnot(length(STRATA) == PER_STRUCTURE)

has_correlation <- function(st) any(vapply(st$random, function(b) b$correlated && length(b$terms) > 1, TRUE))

seed_for <- function(text) {
  codes <- utf8ToInt(text)
  sum(codes * seq_along(codes)) %% .Machine$integer.max
}

log_uniform <- function(lo, hi) exp(runif(1, log(lo), log(hi)))

# ---------------------------------------------------------------------------
# The spec: every sampled dimension, drawn from the scenario's own seed.
# ---------------------------------------------------------------------------
draw_spec <- function(struct_id, idx) {
  st <- STRUCTURES[[struct_id]]
  id <- sprintf("%s_%03d", struct_id, idx)
  seed <- seed_for(paste(MASTER_SEED, struct_id, idx))
  set.seed(seed)
  stratum <- STRATA[idx]
  if (stratum == "corr_boundary" && !has_correlation(st)) stratum <- "free"
  crossed <- isTRUE(st$crossed)
  nested <- isTRUE(st$nested)
  fixed <- if (length(st$fixed) == 1) st$fixed else sample(st$fixed, 1)
  fd <- FIXED[[fixed]]
  small <- stratum == "small_n"
  n_subjects <- if (small) sample(5:8, 1) else round(log_uniform(8, if (crossed) 40 else 60))
  n_items <- if (crossed) (if (small) sample(4:6, 1) else round(log_uniform(6, 24))) else NA
  # small_n: at most 4 sites, so >= 2 subjects per site still allows n <= 8
  # (with up to 6 sites the floor below made every NESTED n >= 10; first
  # run, 2026-10-19).
  n_sites <- if (nested) (if (small) sample(3:4, 1) else sample(3:6, 1)) else NA
  if (nested) n_subjects <- max(n_subjects, 2 * n_sites)
  reps <- if (length(st$reps) == 1) st$reps else sample(st$reps, 1)
  # A subject with no within cells needs >= 3 observations: 2 to separate a
  # random intercept from the residual, 3 for a slope on x.
  if (!crossed && !length(fd$within)) reps <- max(reps, 3L)
  has_time <- any(c("time", "session") %in% names(fd$within))
  missing <- if (stratum == "balanced") "complete"
             else if (stratum == "heavy_missing") sample(c("mcar", "monotone"), 1)
             else sample(c("complete", "mcar", "monotone", "unequal_groups", "singletons"), 1,
                         prob = c(0.3, 0.3, 0.15, 0.1, 0.15))
  if (missing == "monotone" && !has_time) missing <- "mcar"
  if (missing == "unequal_groups" && !length(fd$between)) missing <- "unequal_reps"
  missing_p <- if (missing %in% c("mcar", "monotone")) {
    if (stratum == "heavy_missing") runif(1, 0.25, 0.4) else runif(1, 0.05, 0.25)
  } else NA
  scale <- if (stratum == "scale") sample(c("small", "large"), 1) else "unit"
  uses_x <- fd$covariate || any(vapply(st$random, function(b) "x" %in% b$terms, TRUE))
  # An offset x (e.g. age 40-80) pushes the intercept-slope correlation of
  # the fitted (raw x) model towards +-1: the classic convergence stress.
  x_offset <- uses_x && stratum != "corr_boundary" && runif(1) < 0.3
  effect <- if (runif(1) < 0.2) "null" else "effect"
  # True random-effect SDs (residual SD = 1) and correlations, per block.
  truth <- lapply(st$random, function(b) {
    sd <- vapply(b$terms, function(t) if (t == "(Intercept)") log_uniform(0.3, 2) else log_uniform(0.15, 1), 0)
    corr <- if (b$correlated && length(b$terms) > 1) runif(1, -0.8, 0.8) else NA
    list(sd = unname(sd), corr = corr)
  })
  if (stratum == "zero_variance") {
    # The LAST term of a random block: a slope where there is one, else an
    # intercept (for RI, the subject intercept itself).
    k <- sample(seq_along(truth), 1)
    truth[[k]]$sd[length(truth[[k]]$sd)] <- 0
  }
  if (stratum == "corr_boundary") {
    k <- which(vapply(st$random, function(b) b$correlated && length(b$terms) > 1, TRUE))[1]
    truth[[k]]$corr <- sample(c(-1, 1), 1) * runif(1, 0.97, 0.995)
  }
  list(id = id, structure = struct_id, index = idx, stratum = stratum, seed = seed, fixed = fixed,
       n_subjects = n_subjects, n_items = n_items, n_sites = n_sites, reps = reps,
       missing = missing, missing_p = missing_p, scale = scale, x_offset = x_offset,
       collinear = stratum == "collinear", effect = effect, truth = truth)
}

# ---------------------------------------------------------------------------
# Data generation (continues the spec's random stream: deterministic per id)
# ---------------------------------------------------------------------------
generate <- function(spec) {
  st <- STRUCTURES[[spec$structure]]
  fd <- FIXED[[spec$fixed]]
  n <- spec$n_subjects
  subjects <- sprintf("S%02d", seq_len(n))
  sub <- data.frame(subject = subjects, stringsAsFactors = FALSE)
  if (isTRUE(st$nested)) sub$site <- sprintf("site%d", rep_len(seq_len(spec$n_sites), n))
  if (length(fd$between)) {
    g <- fd$between$group
    w <- if (spec$missing == "unequal_groups") c(1, 3) else c(1, 1)
    sizes <- pmax(2, round(n * w / sum(w)))
    sizes[2] <- n - sizes[1]
    sub$group <- rep(g, sizes)
  }
  if (isTRUE(st$crossed)) {
    items <- sprintf("I%02d", seq_len(spec$n_items))
    rows <- merge(sub, data.frame(item = items, stringsAsFactors = FALSE), by = NULL)
    if ("cond" %in% names(fd$within)) {
      # Latin square: each subject sees every item once, half in each condition.
      rows$cond <- fd$within$cond[(match(rows$subject, subjects) + match(rows$item, items)) %% 2 + 1]
    }
  } else {
    wgrid <- if (length(fd$within)) expand.grid(fd$within, stringsAsFactors = FALSE) else data.frame(.w = 1)
    cells <- merge(sub, wgrid, by = NULL)
    cells$.w <- NULL
    rows <- cells[rep(seq_len(nrow(cells)), spec$reps), , drop = FALSE]
  }
  rows <- rows[order(rows$subject), , drop = FALSE]
  rownames(rows) <- NULL
  N <- nrow(rows)

  # x: uniform on [0, 1] (centred for the data-generating effects), written
  # raw: -1..1, or 40..80 when offset.
  uses_x <- fd$covariate || any(vapply(st$random, function(b) "x" %in% b$terms, TRUE))
  x01 <- if (uses_x) runif(N) else rep(0.5, N)
  xc <- x01 - 0.5
  if (uses_x) rows$x <- if (spec$x_offset) 40 + 40 * x01 else 2 * x01 - 1

  facs <- c(names(fd$between), names(fd$within))
  lv <- c(fd$between, fd$within)
  k <- lapply(facs, function(f) match(rows[[f]], lv[[f]]) - 1)
  names(k) <- facs
  lin <- rep(0, N)
  if (spec$effect == "effect") {
    for (f in facs) lin <- lin + 0.5 * k[[f]]
    if (length(facs) >= 2) lin <- lin + 0.25 * k[[facs[1]]] * k[[facs[2]]]
    if (uses_x) lin <- lin + 1.0 * xc
    if (uses_x && "group" %in% facs) lin <- lin + 0.5 * xc * k[["group"]]
  }

  # Random effects, block by block: b ~ N(0, D R D), as rows of a
  # (levels x terms) matrix; a zero SD gives an exactly zero column.
  re <- rep(0, N)
  for (j in seq_along(st$random)) {
    b <- st$random[[j]]
    tr <- spec$truth[[j]]
    gkey <- do.call(paste, c(rows[b$group], sep = ":"))
    glev <- unique(gkey)
    Z <- sapply(b$terms, function(t) switch(t, "(Intercept)" = rep(1, N), x = xc,
                                            cond = ifelse(rows$cond == "A", 1, -1)))
    Z <- matrix(Z, nrow = N)
    R <- diag(length(b$terms))
    if (!is.na(tr$corr)) R[1, 2] <- R[2, 1] <- tr$corr
    u <- matrix(rnorm(length(glev) * length(b$terms)), ncol = length(b$terms)) %*% chol(R)
    u <- sweep(u, 2, tr$sd, `*`)
    re <- re + rowSums(Z * u[match(gkey, glev), , drop = FALSE])
  }
  e <- rnorm(N)
  # RI with a zero subject SD: centre the noise within each subject as well,
  # so the subject means differ only by the fixed effects and lme4 lands
  # EXACTLY on the boundary (as the curated ICC-0 scenarios).
  if (spec$structure == "RI" && spec$truth[[1]]$sd[1] == 0) e <- e - ave(e, rows$subject)
  y <- 10 + lin + re + e
  y <- switch(spec$scale, unit = y, small = y * 1e-3, large = 1e4 + 100 * y)
  rows$y <- as.numeric(y)

  # An exactly collinear column z: lme4 drops it ("fixed-effect model matrix
  # is rank deficient"), and records which (dropped_columns).
  if (spec$collinear) {
    rows$z <- if (uses_x) 2 * rows$x - 1 else if ("group" %in% names(rows)) as.numeric(rows$group == "ctl") else 3
  }
  apply_missing(rows, spec, fd, isTRUE(st$crossed))
}

apply_missing <- function(data, spec, fd, crossed) {
  pat <- spec$missing
  all_subjects <- unique(data$subject)
  drop <- rep(FALSE, nrow(data))
  if (pat == "mcar") {
    drop[sample.int(nrow(data), round(spec$missing_p * nrow(data)))] <- TRUE
  } else if (pat == "monotone") {
    tf <- intersect(c("time", "session"), names(fd$within))[1]
    L <- fd$within[[tf]]
    for (s in sample(all_subjects, max(1, round(spec$missing_p * length(all_subjects))))) {
      cut <- sample(seq_len(length(L) - 1), 1)
      drop <- drop | (data$subject == s & match(data[[tf]], L) > cut)
    }
  } else if (pat == "unequal_reps") {
    rate <- setNames(runif(length(all_subjects), 0, 0.6), all_subjects)
    drop <- runif(nrow(data)) < rate[data$subject]
  } else if (pat == "singletons") {
    for (s in sample(all_subjects, max(1, round(0.2 * length(all_subjects))))) {
      mine <- which(data$subject == s)
      drop[setdiff(mine, mine[sample.int(length(mine), 1)])] <- TRUE
    }
  }
  removed <- data[drop, , drop = FALSE]
  kept <- data[!drop, , drop = FALSE]
  # Guards, so every scenario is a model lme4 can fit as specified: every
  # subject (and item) keeps >= 1 row; every fixed-factor cell keeps rows
  # from >= 2 subjects.
  restore <- function(kept, rows) rbind(kept, rows)
  for (s in setdiff(all_subjects, unique(kept$subject))) {
    kept <- restore(kept, removed[removed$subject == s, , drop = FALSE][1, , drop = FALSE])
  }
  if (crossed) for (it in setdiff(unique(data$item), unique(kept$item))) {
    kept <- restore(kept, removed[removed$item == it, , drop = FALSE][1, , drop = FALSE])
  }
  facs <- c(names(fd$between), names(fd$within))
  if (length(facs)) {
    dkey <- function(df) do.call(paste, c(df[facs], sep = "\r"))
    for (cell in unique(dkey(data))) {
      have <- unique(kept$subject[dkey(kept) == cell])
      if (length(have) < 2) {
        pool <- removed[dkey(removed) == cell & !(removed$subject %in% have), , drop = FALSE]
        add_s <- head(unique(pool$subject), 2 - length(have))
        kept <- restore(kept, pool[pool$subject %in% add_s, , drop = FALSE])
      }
    }
  }
  kept <- unique(kept)
  kept <- kept[order(kept$subject), , drop = FALSE]
  rownames(kept) <- NULL
  kept
}

# ---------------------------------------------------------------------------
# Fitting and recording
# ---------------------------------------------------------------------------
random_rhs <- function(random) {
  paste(vapply(random, function(b) {
    lhs <- paste(ifelse(b$terms == "(Intercept)", "1", b$terms), collapse = " + ")
    sprintf("(%s %s %s)", lhs, if (b$correlated) "|" else "||", paste(b$group, collapse = ":"))
  }, ""), collapse = " + ")
}

fixed_rhs <- function(fd, collinear) {
  facs <- c(names(fd$between), names(fd$within))
  parts <- c(facs, if (fd$covariate) "x")
  rhs <- if (length(parts)) paste(parts, collapse = " * ") else "1"
  if (collinear) rhs <- paste(rhs, "+ z")
  rhs
}

# lme4's REML and ML deviance at fixed theta (devFunOnly): the objective,
# with no optimizer involved. Probes: the estimate; each component x0.9 and
# x1.1 (a zero component: +0.05 only); every diagonal (variance) element at
# 0; three random points inside the bounds.
probe_deviances <- function(form, data, theta, lower) {
  dev_reml <- lme4::lmer(form, data = data, REML = TRUE, devFunOnly = TRUE)
  dev_ml <- lme4::lmer(form, data = data, REML = FALSE, devFunOnly = TRUE)
  diag_ix <- which(lower == 0)
  off_ix <- which(lower != 0)
  thetas <- list(estimate = theta)
  for (j in seq_along(theta)) {
    up <- theta; dn <- theta
    if (theta[j] == 0) {
      up[j] <- 0.05
      thetas[[sprintf("theta%d_up", j)]] <- up
    } else {
      up[j] <- theta[j] * 1.1; dn[j] <- theta[j] * 0.9
      thetas[[sprintf("theta%d_up", j)]] <- up
      thetas[[sprintf("theta%d_down", j)]] <- dn
    }
  }
  b <- theta; b[diag_ix] <- 0
  thetas$boundary <- b
  for (r in 1:3) {
    t <- theta
    t[diag_ix] <- runif(length(diag_ix), 0, 2)
    t[off_ix] <- runif(length(off_ix), -1, 1)
    thetas[[sprintf("random%d", r)]] <- t
  }
  unname(lapply(names(thetas), function(nm) {
    t <- thetas[[nm]]
    list(label = nm, theta = I(unname(t)), reml = dev_reml(t), ml = dev_ml(t))
  }))
}

OPTIMIZERS <- list(
  bobyqa = lmerControl(optimizer = "bobyqa"),
  Nelder_Mead = lmerControl(optimizer = "Nelder_Mead"),
  nlopt_bobyqa = lmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA")),
  nlopt_neldermead = lmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_NELDERMEAD"))
)

spread_of <- function(m) if (is.null(m) || nrow(m) < 2) rep(0, ncol(m)) else apply(m, 2, function(v) diff(range(v)))

# Refit with every optimizer (lme4::lmer: no Satterthwaite needed). The
# spread between the converged ones (all of them, if fewer than 2 converged)
# is how far two good optimizers legitimately disagree on this dataset.
optimizer_spread <- function(form, data) {
  fits <- lapply(names(OPTIMIZERS), function(nm) {
    r <- tryCatch({
      q <- fit_quietly(form, data, fitter = lme4::lmer, control = OPTIMIZERS[[nm]])
      m <- q$fit
      list(optimizer = nm, ok = TRUE,
           converged = !any(startsWith(q$notes, "warning:")),
           criterion = as.numeric(REMLcrit(m)),
           theta = I(unname(getME(m, "theta"))),
           fixef = as.list(fixef(m)),
           se = as.list(setNames(sqrt(diag(as.matrix(vcov(m)))), names(fixef(m)))),
           vcov = I(as.data.frame(VarCorr(m))$vcov),
           notes = as.list(q$notes))
    }, error = function(e) list(optimizer = nm, ok = FALSE, converged = FALSE, error = conditionMessage(e)))
    r
  })
  ok <- Filter(function(f) f$ok, fits)
  use <- Filter(function(f) f$converged, ok)
  if (length(use) < 2) use <- ok
  mat <- function(key) if (length(use)) do.call(rbind, lapply(use, function(f) unlist(f[[key]]))) else NULL
  spread <- if (length(use)) {
    fx <- mat("fixef"); se <- mat("se")
    list(criterion = diff(range(vapply(use, function(f) f$criterion, 0))),
         theta = I(spread_of(mat("theta"))),
         fixef = as.list(setNames(spread_of(fx), colnames(fx))),
         se = as.list(setNames(spread_of(se), colnames(se))),
         vcov = I(spread_of(mat("vcov"))))
  } else NULL
  list(fits = fits, n_converged = sum(vapply(fits, function(f) isTRUE(f$converged), TRUE)),
       n_used = length(use), spread = spread)
}

fit_scenario <- function(spec, data) {
  st <- STRUCTURES[[spec$structure]]
  fd <- FIXED[[spec$fixed]]
  facs <- c(names(fd$between), names(fd$within))
  lv <- c(fd$between, fd$within)
  for (f in facs) data[[f]] <- factor(data[[f]], levels = lv[[f]])
  for (g in intersect(c("subject", "item", "site"), names(data))) data[[g]] <- factor(data[[g]])
  fixed_part <- paste("y ~", fixed_rhs(fd, spec$collinear))
  form <- as.formula(paste(fixed_part, "+", random_rhs(st$random)))
  q <- fit_quietly(form, data)
  fit <- q$fit
  X <- getME(fit, "X")
  # emmeans contrasts only where they mean what csv-stats' lmm_pairwise
  # computes: factors only (no covariate, no collinear column).
  contrasts <- if (length(facs) && !fd$covariate && !spec$collinear) describe_contrasts(fit, facs, lv)
               else list(contrasts_satterthwaite = list(), contrasts_wald = list())
  c(
    list(formula = paste(deparse(form, width.cutoff = 500), collapse = " "), coding = "contr.sum",
         fixed_terms = I(attr(terms(as.formula(fixed_part)), "term.labels")),
         fixed_columns = I(colnames(X)),
         dropped_columns = I(as.character(names(attr(X, "col.dropped"))))),
    describe_fixed(fit),
    contrasts,
    describe_variance(fit, q$notes),
    list(theta = I(unname(getME(fit, "theta"))),
         theta_names = I(names(getME(fit, "theta"))),
         theta_lower = I(unname(getME(fit, "lower"))),
         vcov_beta = unname(as.matrix(vcov(fit))),
         probes = probe_deviances(form, data, getME(fit, "theta"), getME(fit, "lower")),
         optimizers = optimizer_spread(form, data))
  )
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PLAN <- expand.grid(index = seq_len(PER_STRUCTURE), structure = names(STRUCTURES), stringsAsFactors = FALSE)
remove_stale(c(data_dir, res_dir), sprintf("%s_%03d", PLAN$structure, PLAN$index))

cat(sprintf("%d scenarios (%d structures x %d) -> %s\n", nrow(PLAN), length(STRUCTURES), PER_STRUCTURE, out_dir))
manifest <- list()
for (r in seq_len(nrow(PLAN))) {
  # draw_spec seeds the stream; generate() continues it, so each scenario's
  # data depend only on its own id.
  spec <- draw_spec(PLAN$structure[r], PLAN$index[r])
  st <- STRUCTURES[[spec$structure]]
  fd <- FIXED[[spec$fixed]]
  data <- generate(spec)
  write.csv(data, file.path(data_dir, paste0(spec$id, ".csv")), row.names = FALSE)
  out <- list(
    id = spec$id, scenario = spec[setdiff(names(spec), c("id", "truth"))],
    truth = spec$truth,
    random = lapply(st$random, function(b) list(group = I(b$group), terms = I(b$terms), correlated = b$correlated)),
    # I(): always JSON arrays (auto_unbox would make a one-element vector a
    # bare string; the Python reader tolerates both anyway).
    factors = I(as.character(c(names(fd$between), names(fd$within)))),
    between = I(as.character(names(fd$between))),
    within = I(as.character(names(fd$within))),
    covariates = I(if (fd$covariate) "x" else character()),
    collinear = if (spec$collinear) "z" else NULL,
    levels = c(fd$between, fd$within), subject = "subject", dv = "y",
    n_rows = nrow(data), n_subjects = length(unique(data$subject)),
    versions = versions
  )
  t0 <- Sys.time()
  res <- tryCatch(fit_scenario(spec, data), error = function(e) list(error = conditionMessage(e)))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  out <- c(out, res)
  write_result(out, file.path(res_dir, paste0(spec$id, ".json")))
  verdict <- if (!is.null(res$error)) "failed"
             else if (isTRUE(res$singular)) "singular"
             else if (length(res$notes)) "noted"
             else "ok"
  # NA when R failed or no optimizer refit succeeded (spread is then null).
  spread <- if (is.null(res$error) && !is.null(res$optimizers$spread)) res$optimizers$spread$criterion else NA_real_
  manifest[[spec$id]] <- data.frame(
    id = spec$id, structure = spec$structure, stratum = spec$stratum, fixed = spec$fixed,
    n_subjects = out$n_subjects, n_items = spec$n_items, n_sites = spec$n_sites, reps = spec$reps,
    missing = spec$missing, missing_p = round(spec$missing_p, 3), scale = spec$scale, x_offset = spec$x_offset,
    collinear = spec$collinear, effect = spec$effect,
    true_zero_variance = any(vapply(spec$truth, function(t) any(t$sd == 0), TRUE)),
    true_max_abs_corr = suppressWarnings(max(abs(unlist(lapply(spec$truth, `[[`, "corr"))), na.rm = TRUE)),
    n_rows = nrow(data), verdict = verdict,
    singular = if (is.null(res$error)) isTRUE(res$singular) else NA,
    n_dropped = if (is.null(res$error)) length(res$dropped_columns) else NA,
    criterion_spread = spread,
    n_converged = if (is.null(res$error)) res$optimizers$n_converged else NA,
    seconds = round(secs, 1),
    stringsAsFactors = FALSE
  )
  msg <- if (!is.null(res$error)) res$error else paste(unlist(res$notes), collapse = " | ")
  cat(sprintf("%-8s %-18s %-13s %-7s rows=%-5d spread=%-8.1e dropped=%s %4.1fs %s\n",
              verdict, spec$id, spec$stratum, spec$fixed, nrow(data), spread,
              if (is.null(res$error)) length(res$dropped_columns) else "-", secs, substr(msg, 1, 120)))
}
manifest <- do.call(rbind, manifest)
manifest$true_max_abs_corr[!is.finite(manifest$true_max_abs_corr)] <- NA
write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)

cat("\nVerdicts per structure:\n")
print(table(manifest$structure, manifest$verdict))
cat(sprintf("\nDone: %d scenarios, %d failed, %d singular. Total %.0f s.\n", nrow(manifest),
            sum(manifest$verdict == "failed"), sum(manifest$singular, na.rm = TRUE), sum(manifest$seconds)))
cat("Commit tests/reference/scenarios (or send it back).\n")
