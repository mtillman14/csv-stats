# LMM reference values (R lme4 / lmerTest / emmeans)

`make_lmm_reference.R` generates about 75 datasets covering every dimension
that changes a Satterthwaite degrees-of-freedom computation:
- design shape (within, crossed within, between + within, between-only with
  repeats)
- missing-data pattern (MCAR 5–30%, monotone dropout, a level missing in one
  group, single-observation subjects, unequal groups, unequal replicates, a
  sparse cell)
- sample size (6–40)
- ICC (0.8 down to exactly 0, which gives a singular fit)
- null vs real effects
- numeric scale
- odd labels and level order
- random slopes (marked `"future"`)

For each dataset it writes `datasets/<id>.csv` and `results/<id>.json`. Each
results file holds:
- the fixed effects with Satterthwaite df
- type-III F tests (Satterthwaite and Kenward–Roger)
- every pairwise contrast from the FULL model (emmeans, Satterthwaite and Wald)
- the variance components
- the REML criterion
- the singular-fit and convergence flags
- the R package versions

Run once (R >= 4.3). In R, install the packages:

```r
install.packages(c("lme4", "lmerTest", "emmeans", "pbkrtest", "jsonlite"))
```

Then EITHER, from a terminal in the csv-stats folder:

```
Rscript tests/reference/make_lmm_reference.R
```

OR, from the R console or RStudio:

```r
source("<path to csv-stats>/tests/reference/make_lmm_reference.R")
```

The outputs always land next to the script.

Commit the generated `datasets/` and `results/`. Regenerate only on purpose,
and say why in the commit: the Python tests (`tests/test_lmm_reference.py`)
treat these files as ground truth.
