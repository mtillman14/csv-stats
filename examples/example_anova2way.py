"""Two-way ANOVA example using synthetic data.

Generates data with two factors (treatment and sex) and runs an
independent-samples two-way ANOVA.  Results are saved to
example_anova2way.pdf in the current working directory.
"""

import numpy as np
import pandas as pd
from csvstats.anova import anova2way

# Reproducible random data
np.random.seed(42)
n = 20

data = pd.DataFrame({
    "treatment": np.repeat(["A", "B"], n * 2),
    "sex": np.tile(np.repeat(["M", "F"], n), 2),
    "score": np.concatenate([
        np.random.normal(loc=5.0, scale=1.0, size=n),   # A, M
        np.random.normal(loc=5.0, scale=1.0, size=n),   # A, F
        np.random.normal(loc=8.0, scale=1.0, size=n),   # B, M
        np.random.normal(loc=8.0, scale=1.0, size=n),   # B, F
    ]),
})

result = anova2way(
    data,
    group_column1="treatment",
    group_column2="sex",
    data_column="score",
    filename="example_anova2way.pdf",
)

print("Main effect (treatment): F =", result["main_effects"]["treatment"]["F"],
      " p =", result["main_effects"]["treatment"]["p"])
print("Main effect (sex):       F =", result["main_effects"]["sex"]["F"],
      " p =", result["main_effects"]["sex"]["p"])
print("Interaction:             F =", result["interaction"]["F"],
      " p =", result["interaction"]["p"])
