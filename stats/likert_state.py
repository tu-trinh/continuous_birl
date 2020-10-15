import pandas as pd
import numpy as np

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.descriptivestats import describe
import scikit_posthocs as sp


"""

* do within subjects anova to check for significance,
* then follow up with post hoc anova between the two algorithms of interest
* all results are statistically significant (ours -> noise, ours -> IRL)
* p < .01 across the board


"""


# learned
df = pd.DataFrame({'subject': np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
                   'algorithm': np.repeat([1, 2, 3], 10),
                   'learned': [3.5, 4.5, 2.5, 1, 1.5, 3, 3.5, 2, 2, 1.5,
                   5, 4, 6, 5.5, 4.5, 3, 2.5, 6, 4.5, 4,
                   7, 7, 7, 7, 6, 7, 6, 6, 6, 6.5]})


df23 = df[10:30]
print("learned")
print(AnovaRM(data=df, depvar='learned', subject='subject', within=['algorithm']).fit())
print(AnovaRM(data=df23, depvar='learned', subject='subject', within=['algorithm']).fit())

# intuitive
df = pd.DataFrame({'subject': np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
                   'algorithm': np.repeat([1, 2, 3], 10),
                   'intuitive': [2.5, 4.5, 3, 1, 1.5, 2, 3, 2.5, 2.5, 4,
                   4.5, 2.5, 2.5, 5, 3, 3, 1, 4.5, 4.5, 4.5,
                   7, 5, 4.5, 6.5, 4.5, 7, 5, 5, 6.5, 4]})


df23 = df[10:30]
print("intuitive")
print(AnovaRM(data=df, depvar='intuitive', subject='subject', within=['algorithm']).fit())
print(AnovaRM(data=df23, depvar='intuitive', subject='subject', within=['algorithm']).fit())

# intent
df = pd.DataFrame({'subject': np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
                   'algorithm': np.repeat([1, 2, 3], 10),
                   'intent': [3, 3, 2.5, 1, 1, 5, 4, 2, 5, 3.5,
                   2.5, 4, 6, 5.5, 1.5, 3, 1.5, 5.5, 4.5, 4.5,
                   7, 3, 7, 7, 4.5, 7, 6.5, 5.5, 6, 6.5]})


df23 = df[10:30]
print("intent")
print(AnovaRM(data=df, depvar='intent', subject='subject', within=['algorithm']).fit())
print(AnovaRM(data=df23, depvar='intent', subject='subject', within=['algorithm']).fit())

# deploy
df = pd.DataFrame({'subject': np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
                   'algorithm': np.repeat([1, 2, 3], 10),
                   'deploy': [1.5, 3, 1, 1, 1, 1, 1.5, 2, 1.5, 1,
                   2, 2, 4, 4.5, 5, 1.5, 1, 3.5, 2.5, 1.5,
                   6, 4.5, 3.5, 7, 4.5, 6, 4, 6.5, 5.5, 6]})


df23 = df[10:30]
print("deploy")
print(AnovaRM(data=df, depvar='deploy', subject='subject', within=['algorithm']).fit())
print(AnovaRM(data=df23, depvar='deploy', subject='subject', within=['algorithm']).fit())

#prefer
df = pd.DataFrame({'subject': np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
                   'algorithm': np.repeat([1, 2, 3], 10),
                   'prefer': [3, 3.5, 2.5, 1, 1, 2, 3, 2.5, 2, 3,
                   4, 3.5, 6.5, 5.5, 5, 2, 1, 5.5, 6, 4,
                   7, 7, 7, 7, 5.5, 4, 6.5, 6, 6.5, 7]})


df23 = df[10:30]
print("prefer")
print(AnovaRM(data=df, depvar='prefer', subject='subject', within=['algorithm']).fit())
print(AnovaRM(data=df23, depvar='prefer', subject='subject', within=['algorithm']).fit())
