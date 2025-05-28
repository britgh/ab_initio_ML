# additional vector calculations
# britgh last updated: 05.27.25

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

# finding mode of K neighbor data labels
def mode (items) :
    setosa = 0
    versicolor = 0
    virginica = 0

    for item in items:
        if item == "Iris-setosa":
            setosa += 1
        elif item == "Iris-versicolor":
            versicolor += 1
        elif item == "Iris-virginica":
            virginica += 1

    # return name of max value, not actual value
    name = {setosa: "Iris-setosa", versicolor: "Iris-versicolor", virginica: "Iris-virginica"}
    return name.get(max(setosa, versicolor, virginica))


# determining model accuracy (= correct preds / total test data)
def accuracy(all_elem) :
    correct = 0

    for i in range(len(all_elem)) :
        if all_elem.iloc[i, 2] == 1:   # pred = 1st col, real = 2nd, score = 3rd
            correct += 1

    score = correct / len(all_elem)
    print("Accuracy:", score * 100, "%")