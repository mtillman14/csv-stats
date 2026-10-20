# When csv-stats' REML criterion beats lme4's on a sampled scenario: is that
# a real, better optimum, or a csv-stats criterion bug away from the probes?
#
# Input: scenarios/ours_check.json, {case: {theta, names, criterion}},
# written from Python (see .claude/plan-lmm-general-fit.md, "lme4 stopped
# short"). For each case this prints:
#   - lme4's reported criterion, and its devFunOnly at lme4's own theta
#   - lme4's devFunOnly at csv-stats' theta (the decisive number: lme4's
#     own function says which theta is better)
#   - lme4 refitted with every optimizer, STARTED at csv-stats' theta
#
# Run:  Rscript tests/reference/check_lmm_optimum.R   (from the csv-stats folder)
suppressPackageStartupMessages({ library(lme4); library(jsonlite) })
options(contrasts = c("contr.sum", "contr.poly"))

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) return(dirname(normalizePath(f)))
  getwd()
})
scen <- file.path(script_dir, "scenarios")
ours <- fromJSON(file.path(scen, "ours_check.json"), simplifyVector = TRUE)

OPTIMIZERS <- list(
  bobyqa = lmerControl(optimizer = "bobyqa"),
  Nelder_Mead = lmerControl(optimizer = "Nelder_Mead"),
  nlopt_bobyqa = lmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"))
)

for (case in names(ours)) {
  o <- ours[[case]]
  ref <- fromJSON(file.path(scen, "results", paste0(case, ".json")), simplifyVector = TRUE)
  data <- read.csv(file.path(scen, "datasets", paste0(case, ".csv")), stringsAsFactors = FALSE)
  for (col in names(data)) if (is.character(data[[col]])) data[[col]] <- factor(data[[col]])
  form <- as.formula(ref$formula)
  fit <- lmer(form, data = data, REML = TRUE)
  r_names <- names(getME(fit, "theta"))
  if (!identical(unname(r_names), unname(o$names))) {
    cat(case, ": theta names differ; R", r_names, "| ours", o$names, "\n")
    next
  }
  dev <- lmer(form, data = data, REML = TRUE, devFunOnly = TRUE)
  r_theta <- unname(getME(fit, "theta"))
  cat(sprintf("\n%s  %s\n", case, ref$formula))
  cat(sprintf("  lme4 reported criterion   %.10f  (fixture %.10f)\n", REMLcrit(fit), ref$reml_criterion))
  cat(sprintf("  devFun at lme4 theta      %.10f  theta = %s\n", dev(r_theta), paste(signif(r_theta, 6), collapse = ", ")))
  cat(sprintf("  devFun at csv-stats theta %.10f  theta = %s\n", dev(o$theta), paste(signif(o$theta, 6), collapse = ", ")))
  cat(sprintf("  csv-stats criterion       %.10f\n", o$criterion))
  for (nm in names(OPTIMIZERS)) {
    r <- tryCatch({
      m <- lmer(form, data = data, REML = TRUE, control = OPTIMIZERS[[nm]], start = list(theta = o$theta))
      sprintf("%.10f  theta = %s", REMLcrit(m), paste(signif(unname(getME(m, "theta")), 6), collapse = ", "))
    }, error = function(e) paste("error:", conditionMessage(e)))
    cat(sprintf("  refit from csv-stats (%-12s) %s\n", nm, r))
  }
}
