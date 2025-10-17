# csv-stats
Python package for rapid hypothesis testing on CSV files with data in long table format, which are common in the life sciences. Test results are saved to PDF as a rendered JSON string, or can be returned as a Python dictionary.

## Installation
```bash
pip install csv-stats
```

## Examples
All code examples below assume the following import and metadata:
```python
from csv_stats import anova1way, anova2way, anova3way, anova_repeated
data_path = "path/to/data.csv" # Path to your CSV file
data_column = 'values' # The column to run the hypothesis tests on
group1_column = 'group1' # First grouping variable (i.e. statistical factor)
group2_column = 'group2' # Second grouping variable (i.e. statistical factor)
group3_column = 'group3' # Third grouping variable (i.e. statistical factor)
repeated_measure_column = 'subject_id' # Column indicating repeated measures (e.g. subject IDs)
```

# ANOVA
One way ANOVA is currently supported. They include tests of homogeneity of variance and normality of residuals. Repeated measures ANOVA is also supported, including tests of sphericity.

NOTE: Two- and three-way ANOVA support is planned, but not yet implemented.

```python
# One way ANOVA, independent samples
result_anova1way = anova1way(data_path, group1_column, data_column)

# One way ANOVA, repeated measures
result_anova1way_rm = anova1way(data_path, group1_column, data_column, repeated_measure_column)

# Two way ANOVA, independent samples
result_anova2way = anova2way(data_path, group1_column, group2_column, data_column)

# Two way ANOVA, repeated measures
result_anova2way_rm = anova2way(data_path, group1_column, group2_column, data_column, repeated_measure_column)

# Three way ANOVA, independent samples
result_anova3way = anova3way(data_path, group1_column, group2_column, group3_column, data_column)

# Three way ANOVA, repeated measures
result_anova3way_rm = anova3way(data_path, group1_column, group2_column, group3_column, data_column, repeated_measure_column)
```

# t-test
Both independent samples (one and two samples) and paired samples t-tests are supported. They include tests of homogeneity of variance and normality of residuals.
```python
from csv_stats import ttest_ind, ttest_rel

# Independent samples t-test
result_ttest_ind = ttest_ind(data_path, group1_column, data_column)

# Paired samples t-test
result_ttest_rel = ttest_dep(data_path, group1_column, data_column, repeated_measure_column)

# One sample t-test
result_ttest_1samp = ttest_1samp(data_path, group1_column, data_column) # Tests against a mean of 0
```