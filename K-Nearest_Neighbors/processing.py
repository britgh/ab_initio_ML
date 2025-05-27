# File to contain additional vector calculations
import numpy as np

# min-max normalization
def normalize(current, maxx, minn) :
    return (current - minn) / (maxx - minn)

# Euclidean distance
def euclid_dist(training, testing) :
    return np.sqrt(np.sum(np.square(training - testing), axis = 1))

# finding the max and min
def maximum(current, max) :
    return current if current > max else max

def minimum(current, min) :
    return current if current < min else min

# finding mode
# def mode (items) :
#

# def accuracy(training, testing) :