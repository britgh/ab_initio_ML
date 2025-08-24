# Multi-Linear Regression Machine Learning Model
# britgh last updated: 08.23.25

import calculations
import correlation
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo                         # MPG dataset by UCIML
from sklearn.model_selection import KFold                   # help implement K-Fold cross-validation
import matplotlib.pyplot as plt

# DATASET---
auto_mpg = fetch_ucirepo(id=9)
X = auto_mpg.data.features                  # features (car_name not included)
Y = auto_mpg.data.targets                   # labels (MPG)

# PREPROCESSING---
dropping = correlation.dropping(X, Y)       # least correlated + categorical features removed w/ pcc
X = X.drop(dropping, axis=1)
X = X.astype(np.float64)
# print(X)

# prevent multi collinearity: if an independent var (col) can be predicted
# accurately by other vars using linear regression, remove current var
    # use tolerance and VIF to measure

folds = 10
iteration = 0
results = np.empty((folds,X.shape[1] + 2))      #  num of folds (rows); features + RMSEs (cols)

for train_index, test_index in KFold(n_splits=10).split(X):             # iterating through folds
    X_training, X_testing = X.iloc[train_index], X.iloc[test_index]     # indices
    Y_training, Y_testing = Y.iloc[train_index], Y.iloc[test_index]

    for col in range(X_training.shape[1]):
        x_train_val = X_training.iloc[:, col]                           # z-score training data normalization
        count = x_train_val.shape[0]
        x_mean = np.sum(x_train_val) / count
        x_sd = np.sqrt(np.sum(np.square(x_train_val - x_mean)) / count - 1)
        X_training.iloc[:, col] = (x_train_val - x_mean) / x_sd

        x_test_val = X_testing.iloc[:, col]                             # normalizing testing data w/ training sd + mean
        X_testing.iloc[:, col] = (x_test_val - x_mean) / x_sd

    # MODEL TRAINING---
    learning_rate = 0.001                                               # default stuff
    weights = np.zeros(X_training.shape[1])
    bias = 0

    print("\nBefore GD:", calculations.mse(X_training, weights, bias, Y_training))
    w_deriv, b_deriv = calculations.partial_deriv(X_training, weights, bias, Y_training)

    print("W deriv:", w_deriv)
    print("B deriv:", b_deriv)

    w_new = weights - learning_rate * w_deriv
    b_new = bias - learning_rate * b_deriv

    print("new W:", w_new)
    print("new B:", b_new)

    print("After GD:", calculations.mse(X_training, w_new, b_new, Y_training))





    # do the zscore thing


    # MODEL TESTING--- find predicted y according to line









