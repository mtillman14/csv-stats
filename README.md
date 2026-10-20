# csv-stats
Python package for rapid hypothesis testing on CSV files with data in long table format, which are common in the life sciences. Test results are saved to PDF as a rendered JSON string, or can be returned as a Python dictionary.

## Installation
```bash
pip install "csv-stats[full]"   # hypothesis tests + PDF reports
pip install csv-stats           # descriptive statistics only (numpy, pandas, scipy)
```

## Descriptive statistics
`csvstats.descriptive` is the one definition of every descriptive formula: NaN
skipped (counted in `n_missing`), SD with n - 1, linear quartiles, and a
t-based 95% CI of the mean. Fewer than two values give NaN, not 0.

```python
from csvstats.descriptive import summarize, ci_interval, ci_half_width

summarize([4.1, 5.0, 6.2, None])   # n=3, n_missing=1, mean, sd, sem, ci_low, ci_high, median, q1, q3, iqr, skewness, kurtosis, ...
ci_interval([4.1, 5.0, 6.2])       # (low, high); also a seaborn errorbar= callable
ci_half_width(sd=1.2, n=5)         # t(0.975, 4) * 1.2 / sqrt(5)
```

## Linear mixed models (0.4.0)
`csvstats.lmm` uses every observation: a subject missing some within-subject
levels is kept, not dropped (a repeated-measures ANOVA or paired t-test would
have to drop it). REML, sum-to-zero coding. The random effects default to an
intercept per subject. `random=` takes any lme4-style structure:
- random slopes, correlated or uncorrelated (`||`)
- crossed groupings, e.g. subject + item
- nested and interaction groupings, e.g. `site:subject`

Numeric `covariates=` are crossed with the factors. Term tests are type III,
as in lmerTest and SAS. So a factor crossed with a covariate is tested at
covariate = 0, and a `type3_at_covariate_zero` warning says so. Centre the
covariate to test the factor at the covariate's mean.

`csvstats.lmm_formula` reads and writes the lme4 spelling of these models
(`parse("y ~ group * time * x + (1 + x | subject)")`, `render(...)`), with
errors that point at the offending character. It accepts what `lmm` can fit:
the fixed part is ONE crossing (`a * b * x`, or `1`), and random terms are
`(… | g)`, `(… || g)`, `a:b` and `a/b`. A model with no more observations
than random effects is refused before fitting, as lme4 does.

`lmm_pairwise` takes the same `covariates=` and compares levels with each
covariate at its mean (emmeans' reference grid), or at the value you give
in `covariates_at={"age": 65}`. The record's `details["covariates_at"]`
states the value used. So with a covariate, `lmm`'s F for a factor (at 0)
and `lmm_pairwise`'s contrasts (at the mean) answer different questions,
exactly as lmerTest's anova and emmeans do in R.

The fit is checked against lme4 on 225 sampled scenarios over 9
random-effect structures (tests/reference/scenarios). That covers the
criterion itself at fixed θ, the estimates, θ and the Satterthwaite df.
Tests use **Satterthwaite** degrees of freedom, as R's lmerTest does: t(df) per
coefficient, a type-III F(q, ddf) per term, and t(df) per pairwise contrast. The
pairwise contrasts come from the full model (emmeans-style). Everything is checked
against lme4 / lmerTest / emmeans on 75 reference datasets (tests/reference).
If a df cannot be computed, the Wald test is reported, labelled `df_method: "wald"`.

```python
from csvstats.lmm import lmm, lmm_pairwise
from csvstats.core import assumption_checks

lmm(df, "step_length", within=["session", "speed"], subject="subject")       # one F(q, ddf) per term
lmm_pairwise(df, "step_length", within=["session", "speed"], subject="subject", factor="session",
             levels={"session": ["post", "pre"]})   # post − pre within each speed, t(df)
assumption_checks(df, "step_length", within=["session"], subject="subject")  # Shapiro–Wilk on residuals (+ Levene)

# Random slopes and crossed effects: y ~ cond * x + (1 + x | subject) + (1 | item)
lmm(df, "rt", within=["cond"], subject="subject", covariates=["x"],
    random=[{"group": "subject", "terms": ["(Intercept)", "x"]},
            {"group": "item"}])
```

## Examples
All code examples below use the following constants:
```python
DATA_PATH = "path/to/data.csv" # Path to your CSV file
DATA_COLUMN = 'values' # The column to run the hypothesis tests on
GROUP_COLUMN = 'groups' # Grouping variable (i.e. statistical factor)
REPEATED_MEASURES_COLUMN = 'subject_id' # Column indicating repeated measures (e.g. subject IDs)
```

# ANOVA
One way ANOVA is currently supported. They include tests of homogeneity of variance and normality of residuals. Repeated measures ANOVA is also supported, including tests of sphericity.

NOTE: Two- and three-way ANOVA support is planned, but not yet implemented.

```python
from csv_stats.anova import anova1way

# One way ANOVA, independent samples
result_anova1way = anova1way(DATA_PATH, 
                            GROUP_COLUMN, 
                            DATA_COLUMN,                            
                            filename = "anova1way_results.pdf", # Default save name. Enter `None` to not save.
                            render_plot = False # For speed, by default no plots are generated
                        )

# One way ANOVA, repeated measures
result_anova1way_rm = anova1way(DATA_PATH, 
                            GROUP_COLUMN, 
                            DATA_COLUMN,                             
                            repeated_measures_column = REPEATED_MEASURES_COLUMN,
                            filename = "anova1way_results.pdf", # Default save name. Enter `None` to not save.     
                            render_plot = False # For speed, by default no plots are generated                       
                        )
```

# t-test
Both independent samples (one and two samples) and paired samples t-tests are supported. They include tests of homogeneity of variance and normality of residuals.
```python
from csv_stats.ttest import ttest_ind, ttest_dep

# Independent samples t-test 
# Two sample when the GROUP_COLUMN has two groups
# One smaple when the GROUP_COLUMN has one group
result_ttest_ind = ttest_ind(DATA_PATH, 
                            GROUP_COLUMN, 
                            DATA_COLUMN,
                            popmean = 0, # Test against a population mean of 0 (default)
                            filename = "ttest_ind_results.pdf", # Default save name. Enter `None` to not save.
                            render_plot = False # For speed, by default no plots are generated
                        )

# Paired samples t-test
result_ttest_rel = ttest_dep(DATA_PATH, 
                            GROUP_COLUMN, 
                            DATA_COLUMN, 
                            repeated_measures_column = REPEATED_MEASURES_COLUMN,
                            filename = "ttest_dep_results.pdf", # Default save name. Enter `None` to not save.
                            render_plot = False # For speed, by default no plots are generated
                        )
```

# Multiple Data Columns
If you have multiple columns of data that you want to run the same test on, you can specify the `data_column` argument as "_". This will automatically loop over all columns. Note that to save these results to a file, the filename should be an f-string containing `f"{data_column}"`. The test function will replace `data_column` with the column name in the file name.