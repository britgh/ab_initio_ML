# File to contain additional vector calculations
import numpy as np

# min-max normalization
def normalize(current, maximum, minimum) :                  # max-min data normalization
    return (current - minimum) / (maximum - minimum)

# Euclidean distance
def euclid_dist(training, testing) :
    return np.sqrt(np.sum(np.square(training - testing), axis = 1))


# finding mode
# def mode (items) :
#

# def accuracy(training, testing) :