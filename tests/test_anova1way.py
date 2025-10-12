

from csv_stats.anova import anova1way
import numpy as np
import pandas as pd

def test_anova1way():

    # Create sample data
    np.random.seed(0)
    group_A = np.random.normal(loc=5.0, scale=1.0,
                            size=30)
    group_B = np.random.normal(loc=6.0, scale=1.0, size=30)
    data = np.concatenate([group_A, group_B])
    groups = np.array(['A'] * 30 + ['B'] * 30)
    data = pd.DataFrame({'group': groups, 'value': data})
    data_column = 'value'
    group_column = 'group'

    # Perform one-way ANOVA using the function
    result = anova1way(data, group_column=group_column, data_column=data_column)  


if __name__ == "__main__":
    test_anova1way()
    print("All tests passed.")