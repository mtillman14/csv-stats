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
have to drop it). Random intercept per subject, REML, sum-to-zero coding.
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