import pingouin as pg                                                                                                                               
import numpy as np                                                          
import pandas as pd

np.random.seed(7)
n = 15
subjects = np.tile(np.arange(n), 4)
factor1 = np.repeat(['low', 'high'], n * 2)
factor2 = np.tile(np.repeat(['pre', 'post'], n), 2)
scores = np.concatenate([
    np.random.normal(10, 2, n),
    np.random.normal(12, 2, n),
    np.random.normal(16, 2, n),
    np.random.normal(18, 2, n),
])
data = pd.DataFrame({'subject': subjects, 'dose': factor1, 'time': factor2, 'score': scores})

# Two within factors
result = pg.sphericity(data, dv='score', subject='subject', within=['dose', 'time'])
print('type:', type(result))
print('value:', result)
print()

# Single within factor (what 1-way RM uses)
data2 = data[data['time'] == 'pre']
result2 = pg.sphericity(data2, dv='score', subject='subject', within='dose')
print('type single:', type(result2))
print('value single:', result2)