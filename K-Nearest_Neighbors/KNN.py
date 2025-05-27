# K-Nearest Neighbors Machine Learning Model
# britgh last updated: 05.27.25

import processing
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo                     # import iris dataset (given by UCIML)
from sklearn.model_selection import train_test_split    # for consistent shuffle + data split

# DATASET-----
iris = fetch_ucirepo(id=53)
X = iris.data.features.drop('sepal width', axis=1)      # pd dataframes, remove sepal width feature
Y = iris.data.targets

# Y = Y.replace('Iris-setosa', 0)                         # encoding labels into values
# Y = Y.replace('Iris-virginica', 1)
# Y = Y.replace('Iris-versicolor', 2)
# Y = LabelEncoder().fit_transform(Y)                     # turn labels into values (IS=0, IVi=1, IVe=2)

# DATA SPLIT (sklearn library used here)-----
X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = 0.2)

scores = np.empty(shape=len(X_testing))                 # track predictions: correct = 1, else 0
K = 5                                                   # chosen number of neighbors

# NORMALIZING DATA-----
min_max = pd.DataFrame({'min':[], 'max':[]})            # pandas dataframe to store training set's min-max vals

for feature in X.columns:
    min_max.loc[feature] = [X_training[feature].min(), X_training[feature].max()]

# normalizing according to each test case
for testcase in range(len(Y_testing)):
    case = X_testing.iloc[testcase]
    X_temp = X_training.copy()

    for feature in X.columns:
        min_x = min_max.at[feature, 'min']                  # training data min/max
        max_x = min_max.at[feature, 'max']
        # print(min_x, max_x)
        # print(case[feature])

        min_x = processing.minimum(min_x, case[feature])    # min/max including testcase
        max_x = processing.maximum(max_x, case[feature])

        X_temp[feature] = processing.normalize(X_temp[feature], max_x, min_x)   # normalize training
        case[feature] = processing.normalize(case[feature], max_x, min_x)       # normalize test data
        # print(X_temp[feature])

    # MODEL TESTING-----
    distances = processing.euclid_dist(X_temp, case)                            # euclidean dist. (train set + testcase)
    order = distances.sort_values()[0:K].index                                  # closest K neighbors (pd series sort)
    print(distances[order])

    #print(order)

    # order2=distances.argsort()[0:K]
    # print(order2)
    # print(distances[order2])
    # order = np.argsort(distances)[0:K]                                          # closest K neighbors
    # print(order)
    # print(distances[order]) # only works for first testcase??

    # processing.mode(Y_testing[testcase])

    #print(Y_testing[order])