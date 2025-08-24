# Multi-Linear Regression Machine Learning Model
# britgh last updated: 08.24.25
# future note: prevent multi collinearity: if indep var is predicted by other vars w/ linear reg -> remove var

import calculations
import correlation
import visualization
import numpy as np
from ucimlrepo import fetch_ucirepo                                     # MPG dataset by UCIML
from sklearn.model_selection import KFold                               # help implement K-Fold cross-validation

# DATASET---
auto_mpg = fetch_ucirepo(id=9)
X = auto_mpg.data.features                                              # features (car_name not included)
Y = auto_mpg.data.targets                                               # labels (MPG)

# PREPROCESSING---
dropping = correlation.dropping(X, Y)                                   # least correlated + categorical removed w/ pcc
X = X.drop(dropping, axis=1)
X = X.astype(np.float64)

folds = 10
iteration = 0
# results = np.empty((folds,X.shape[1] + 2))                              #  num of folds (rows); features + RMSEs (cols)

for train_index, test_index in KFold(n_splits=10).split(X):             # iterating through folds
    X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
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
    print(f"\nFold {iteration}: Training Multi-Linear Regression Model")

    weights = np.zeros(X_training.shape[1])
    turns = 2000
    TEST_cost, TEST_bias, TEST_weights = calculations.gradient_descent(X_training, Y_training, weights, rounds=turns, updates=100)

    print(f"Bias for Testing: {TEST_bias};\t Weights for Testing: {TEST_weights}\n")

    # MODEL TESTING---
    test_results = calculations.predict_y(X_testing, TEST_weights, TEST_bias)
    # print(test_results)
    # print(Y_testing)
    print(f"R^2 Score Results: {calculations.R2_score(Y_testing, test_results)}\n")

    visualization.performance(test_results, Y_testing)

    iteration += 1




