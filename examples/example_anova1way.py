"""One-way ANOVA example using synthetic data.

Generates two groups drawn from normal distributions with different means
and runs an independent-samples one-way ANOVA.  Results are saved to
example_anova1way.pdf in the current working directory.
"""

import numpy as np
import pandas as pd
from csvstats.anova import anova1way

# Reproducible random data
np.random.seed(0)
n = 30

data = pd.DataFrame({
    "group": np.repeat(["Control", "Treatment"], n),
    "score": np.concatenate([
        np.random.normal(loc=50, scale=10, size=n),   # Control
        np.random.normal(loc=58, scale=10, size=n),    # Treatment
    ]),
})

result = anova1way(
    data,
    group_column="group",
    data_column="score",
    filename="example_anova1way.pdf",
)

print("F =", result["F"])
print("p =", result["p"])
