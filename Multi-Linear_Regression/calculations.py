# Functions for adapting line of best fit
# britgh last updated: 08.23.25

import numpy as np

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

    return (np.sum(np.square(real_y - predict), axis=0)).values / cases

# returns partial derivatives w.r.t weights + bias
def partial_deriv(given_x, weights, bias, real_y) :
    w_deriv = np.zeros(given_x.shape[1])        # set default vals
    b_deriv = 0

    # partial derivatives w.r.t weights + bias
    difference = (predict_y(given_x, weights, bias) - real_y).values

    for row in range(difference.shape[0]):
        b_deriv += difference[row]
        w_deriv += (difference[row] * given_x.iloc[row]).values

    w_deriv /= difference.shape[0]
    b_deriv /= difference.shape[0]

    return w_deriv, b_deriv
