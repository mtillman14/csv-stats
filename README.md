# csv-stats
Python package for rapid hypothesis testing on CSV files with data in long table format, which are common in the life sciences. Test results are saved to PDF as a rendered JSON string, or can be returned as a Python dictionary.

## Installation
```bash
pip install csv-stats
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