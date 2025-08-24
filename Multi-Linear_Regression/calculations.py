# Functions for adapting line of best fit
# britgh last updated: 08.24.25

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
    w_deriv = np.zeros(given_x.shape[1])
    b_deriv = 0

    difference = (predict_y(given_x, weights, bias) - real_y).values

    for row in range(difference.shape[0]):
        b_deriv += difference[row] * 2
        w_deriv += (difference[row] * given_x.iloc[row]).values * 2

    w_deriv /= difference.shape[0]
    b_deriv /= difference.shape[0]

    return w_deriv, b_deriv

# gradient descent optimization algo to reduce MSE
def gradient_descent (given_x, real_y, weights, bias=0, rounds=500, learning_rate=0.001, updates=50):

    for phase in range(rounds):
        w_deriv, b_deriv = partial_deriv(given_x, weights, bias, real_y)

        weights = weights - learning_rate * w_deriv
        bias = bias - learning_rate * b_deriv

        if phase % updates == 0:
            print(f"Iteration {phase}: \tCost: {mse(given_x, weights, bias, real_y)}, \tWeights: {weights}, \tBias: {bias}")

    return bias, weights