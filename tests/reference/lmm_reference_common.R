# Shared by the two LMM reference scripts:
#   make_lmm_reference.R   the curated set (named scenarios, random intercept)
#   make_lmm_scenarios.R   the sampled set (stratified draws, 9 random-effect
#                          structures)
#
# ONE owner for how a fitted model becomes JSON, so the two fixture sets can
# never record the same quantity differently (and the Python tests read both
# with the same helpers, tests/_lmm_oracle.py).
#
# Each script finds its own folder first, then sources this file:
#   source(file.path(script_dir, "lmm_reference_common.R"))
# That small bootstrap stays in each script: it is how this file is found.
#
# Plan: .claude/plan-lmm-scenario-generator.md in the scistack repo.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(jsonlite)
})
has_kr <- requireNamespace("pbkrtest", quietly = TRUE)

# Sum-to-zero coding, as csv-stats uses (type-III style main effects).
options(contrasts = c("contr.sum", "contr.poly"))
# Our datasets are small; never let emmeans silently switch df methods.
emm_options(lmerTest.limit = 1e6, pbkrtest.limit = 1e6)

finite_or_na <- function(df) {
  df[] <- lapply(df, function(x) if (is.numeric(x)) ifelse(is.finite(x), x, NA) else as.character(x))
  df
}

pair_method <- function(levels) {
  meth <- list()
  for (i in seq_along(levels)) for (j in seq_along(levels)) if (i < j) {
    v <- rep(0, length(levels)); v[i] <- 1; v[j] <- -1
    meth[[paste0(i, "_", j)]] <- v
  }
  meth
}

contrasts_for <- function(fit, f, others, levels, df_method) {
  emm <- emmeans(fit, specs = f, by = if (length(others)) others else NULL, lmer.df = df_method)
  s <- as.data.frame(summary(contrast(emm, method = pair_method(levels), adjust = "none")))
  ij <- strsplit(as.character(s$contrast), "_")
  s$a <- vapply(ij, function(p) levels[as.integer(p[1])], "")
  s$b <- vapply(ij, function(p) levels[as.integer(p[2])], "")
  s$factor <- f
  s$contrast <- NULL
  stat <- intersect(c("t.ratio", "z.ratio"), names(s))
  names(s)[names(s) == stat] <- "statistic"
  finite_or_na(s)
}

# Fit, keeping every warning and message as a note instead of printing it.
# `fitter` is lmerTest::lmer for the reference fit (Satterthwaite needs its
# class) and lme4::lmer for the optimizer refits.
fit_quietly <- function(form, data, REML = TRUE, fitter = lmerTest::lmer, ...) {
  notes <- character()
  fit <- withCallingHandlers(
    fitter(form, data = data, REML = REML, ...),
    warning = function(w) { notes <<- c(notes, paste("warning:", conditionMessage(w))); invokeRestart("muffleWarning") },
    message = function(m) { notes <<- c(notes, paste("message:", trimws(conditionMessage(m)))); invokeRestart("muffleMessage") }
  )
  list(fit = fit, notes = notes)
}

anova_table <- function(fit, ddf) {
  a <- as.data.frame(anova(fit, type = 3, ddf = ddf))
  a <- cbind(term = rownames(a), finite_or_na(a[c("NumDF", "DenDF", "F value", "Pr(>F)")]))
  names(a) <- c("term", "num_df", "den_df", "F", "p")
  a
}

# Fixed effects with Satterthwaite df, the type-III anova (Satterthwaite and,
# when pbkrtest is installed, Kenward-Roger) and the Wald chi2 per term.
# An anova lmerTest cannot compute is recorded as an "unavailable: ..."
# string (the Python tests skip it), never a failed scenario.
describe_fixed <- function(fit) {
  coefs <- as.data.frame(summary(fit)$coefficients)
  names(coefs) <- c("estimate", "se", "df", "t", "p")
  coefs <- cbind(name = rownames(coefs), finite_or_na(coefs))
  a_s <- tryCatch(anova_table(fit, "Satterthwaite"), error = function(e) paste("unavailable:", conditionMessage(e)))
  wald_terms <- if (is.data.frame(a_s)) {
    finite_or_na(data.frame(term = a_s$term, chi2 = a_s$F * a_s$num_df, df = a_s$num_df,
                            p = pchisq(a_s$F * a_s$num_df, a_s$num_df, lower.tail = FALSE)))
  } else a_s
  a_kr <- if (has_kr) tryCatch(anova_table(fit, "Kenward-Roger"),
                               error = function(e) paste("unavailable:", conditionMessage(e))) else "pbkrtest not installed"
  list(fixed = coefs, anova_satterthwaite = a_s, anova_kenward_roger = a_kr, wald_terms = wald_terms)
}

# Every pairwise contrast of every factor, within every combination of the
# others, from the FULL model (emmeans): Satterthwaite and asymptotic (Wald).
describe_contrasts <- function(fit, facs, lv) {
  sat_contrasts <- list(); wald_contrasts <- list()
  for (f in facs) {
    others <- setdiff(facs, f)
    sat_contrasts[[f]] <- tryCatch(contrasts_for(fit, f, others, lv[[f]], "satterthwaite"),
                                   error = function(e) paste("unavailable:", conditionMessage(e)))
    wald_contrasts[[f]] <- tryCatch(contrasts_for(fit, f, others, lv[[f]], "asymptotic"),
                                    error = function(e) paste("unavailable:", conditionMessage(e)))
  }
  list(contrasts_satterthwaite = sat_contrasts, contrasts_wald = wald_contrasts)
}

describe_variance <- function(fit, notes) {
  vc <- as.data.frame(VarCorr(fit))
  vc <- finite_or_na(vc[c("grp", "var1", "var2", "vcov", "sdcor")])
  list(
    varcorr = vc,
    reml_criterion = as.numeric(REMLcrit(fit)),
    loglik_reml = as.numeric(logLik(fit)),
    singular = isSingular(fit),
    convergence_messages = as.list(fit@optinfo$conv$lme4$messages),
    notes = as.list(notes)
  )
}

versions <- list(
  R = R.version.string,
  lme4 = as.character(packageVersion("lme4")),
  lmerTest = as.character(packageVersion("lmerTest")),
  emmeans = as.character(packageVersion("emmeans")),
  pbkrtest = if (has_kr) as.character(packageVersion("pbkrtest")) else NA,
  jsonlite = as.character(packageVersion("jsonlite"))
)

write_result <- function(out, path) {
  write_json(out, path, auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE)
}

# Remove outputs of scenarios no longer in the list, so results/ is exactly
# this run's scenarios (the Python tests parametrize over every file there).
remove_stale <- function(dirs, ids) {
  for (dir in dirs) {
    have <- sub("\\.(csv|json)$", "", list.files(dir, pattern = "\\.(csv|json)$"))
    for (id in setdiff(have, ids)) {
      unlink(file.path(dir, paste0(id, c(".csv", ".json"))))
      cat(sprintf("removed stale %s from %s/\n", id, basename(dir)))
    }
  }
}
