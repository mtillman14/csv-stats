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

## The sampled scenario set (`make_lmm_scenarios.R`)

`make_lmm_scenarios.R` writes a second, larger set into `scenarios/`. It is
the test bed for random-effect structures beyond `(1 | subject)`. The curated
set above stays as it is.

It covers 8 random-effect structures, 25 scenarios each (200 in all):

| id | lme4 formula |
|---|---|
| RI | `(1 \| subject)` |
| RIS | `(1 + x \| subject)` |
| RIS_UNCORR | `(1 + x \|\| subject)` |
| RIF | `(1 + cond \| subject)` |
| RM | `(1 \| subject) + (1 \| subject:time)` |
| CROSSED | `(1 \| subject) + (1 \| item)` |
| NESTED | `(1 \| site) + (1 \| site:subject)` |
| CROSSED_SLOPES | `(1 + cond \| subject) + (1 \| item)` |

Each scenario's dimensions are drawn from its own seed. The draws are
stratified, so every structure always includes:
- balanced data
- a true variance of exactly 0
- a correlation near ±1 (correlated structures only)
- n ≤ 8
- heavy missing data
- a non-unit scale
- an exactly collinear column, which lme4 drops

Beyond what the curated JSON holds, each result records:
- `probes`: lme4's REML and ML deviance at fixed θ (`devFunOnly`). The
  Python tests compare the objective itself at these points, with no
  optimizer involved.
- `optimizers`: refits with bobyqa, Nelder_Mead and two nloptwrap
  algorithms. Their spread sets each scenario's tolerance.
- `random`: the structure in a language-neutral form. Python reads this,
  never the formula.
- `theta`, `theta_lower`, `fixed_terms`, `fixed_columns`, `dropped_columns`
  and `vcov_beta`.

`scenarios/manifest.csv` has one row per scenario: every sampled dimension
plus R's verdict. `tests/test_lmm_scenarios.py` checks its coverage quotas.

Run it the same way as the curated script, from a terminal in the csv-stats
folder:

```
Rscript tests/reference/make_lmm_scenarios.R
```

Expect a few minutes; the four refits per scenario are the slow part. Then
commit `scenarios/`. Expect about 3–5 MB.

Both scripts share `lmm_reference_common.R`, the one place that turns a fit
into JSON. If you change it, rerun both scripts: the curated results must
come out byte-identical unless the change was meant to alter them.
