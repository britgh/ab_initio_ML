# Functions for adapting line of best fit
# britgh last updated: 08.20.25

import numpy as np

# mean square error (avg sqrd diff in y vals)
def mse (real, predict) :
    num = real.shape[0]
    return (np.sum(np.square(real - predict), axis=1)) / num

# root mean square error (sqrt of avg sqrd diff in y vals)
def rmse (real, predict) :
    return np.sqrt(mse(real, predict))

# partial derivative of mse w.r.t. m (how mse changes w/ all vars const except m)
def m_deriv(given_x, real, predict) :
    num = - (2 / real.shape[0])
    return num * np.sum(given_x * (real - predict), axis=1)

# partial derivative of mse w.r.t. b
def b_deriv(real, predict) :
    num = - (2 / real.shape[0])
    return num * np.sum(real - predict, axis=1)

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