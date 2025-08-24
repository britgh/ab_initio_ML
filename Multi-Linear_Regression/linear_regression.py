# Multi-Linear Regression Machine Learning Model
# britgh last updated: 08.20.25

import calculations
import correlation
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo                         # MPG dataset by UCIML
from sklearn.model_selection import KFold                   # help implement K-Fold cross-validation

# DATASET---
auto_mpg = fetch_ucirepo(id=9)
X = auto_mpg.data.features                  # features (car_name not included)
Y = auto_mpg.data.targets                   # labels (MPG)

# PREPROCESSING---
dropping = correlation.dropping(X, Y)       # least correlated + categorical features removed w/ pcc
X = X.drop(dropping, axis=1)
X = X.astype(np.float64)
print(X)

# prevent multi collinearity: if an independent var (col) can be predicted
# accurately by other vars using linear regression, remove current var
    # use tolerance and VIF to measure

folds = 10
iteration = 0
results = np.empty((folds,X.shape[1] + 2))      #  num of folds (rows); features + RMSEs (cols)

for train_index, test_index in KFold(n_splits=10).split(X):             # iterating through folds
    X_training, X_testing = X.iloc[train_index], X.iloc[test_index]     # indices
    Y_training, Y_testing = Y.iloc[train_index], Y.iloc[test_index]

    # MODEL TRAINING--- find m + b s.t. mse is minimized




    # values = np.empty((2, 3))  # stores training sd + mean
    # calculations.z_score(values, True, X, column)







    # MODEL TESTING--- find predicted y according to line









