# Functions for adapting line of best fit
# britgh last updated: 08.23.25

import numpy as np
import pandas as pd

# returns vector of labels for all given input cases (Y' = weights (dot prod) input + bias)
def predict_y(given_x, weights, bias) :
    cases = given_x.shape[0]
    predict = np.zeros((cases, 1))

    for case in range(cases) :
        predict[case] = np.dot(weights, given_x.iloc[case]) + bias

    return predict

# mean square error (avg sqrd diff in y vals)
def mse (given_x, weights, bias, real_y) :
    cases = given_x.shape[0]

    predict = predict_y(given_x, weights, bias)

    return (np.sum(np.square(real_y - predict), axis=0)) / cases

# returns partial derivatives w.r.t weights + bias
def partial_deriv(given_x, weights, bias, real_y) :
    cases = given_x.shape[0]

    # features = given_x.shape[1]
    # w_deriv = np.zeros(features)        # set default vals
    # b_deriv = 0

    # partial derivatives w.r.t weights + bias
    w_deriv = np.sum((predict_y(given_x, weights, bias) - real_y).values * given_x * 2, axis=1) / cases
    b_deriv = np.sum((predict_y(given_x, weights, bias) - real_y).values * 2, axis=1) / cases

    return w_deriv, b_deriv

# normalize differing values
def z_score(arr, is_train, group, col):
    x_val = group.iloc[:, col]  # group being normalized

    if is_train :               # store sample mean + sd of training x vals
        count = x_val.shape[0]

        x_mean = np.sum(x_val) / count
        arr[0, col] = x_mean

        x_sd = np.sqrt(np.sum(np.square(x_val - x_mean)) / count - 1)
        arr[1, col] = x_sd

    else :                      # use mean + sd based on training data
        x_mean = arr[0, col]
        x_sd = arr[1, col]

    return (x_val - x_mean) / x_sd