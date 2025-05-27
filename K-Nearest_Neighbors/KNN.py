# K-Nearest Neighbors Machine Learning Model
# britgh last updated: 05.18.25

import processing
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo                     # import iris dataset (given by UCIML)
from sklearn.model_selection import train_test_split    # for consistent shuffle + data split

# DATASET-----
iris = fetch_ucirepo(id=53)
X = iris.data.features.drop('sepal width', axis=1)      # pd dataframes, remove sepal width feature
Y = iris.data.targets

Y = Y.replace('Iris-setosa', 0)                         # turn labels into values
Y = Y.replace('Iris-virginica', 1)
Y = Y.replace('Iris-versicolor', 2)

# DATA SPLIT (sklearn library used here)-----
X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = 0.2)

scores = np.empty(shape=len(X_testing))                 # track predictions: correct = 1, else 0
K = 5                                                   # chosen number of neighbors

# # NORMALIZING DATA-----
# for feature in X.columns:
#     for case in X_testing[feature]:
#         # X_max = np.max(X[feature].max(), X_testing[feature])
#         print(X[feature].max())
#         print(X_testing[feature])
#         # print(X_max)
    # X[feature] = processing.normalize(X[feature], X[feature].max(), X[feature].min())
    # print(X)


# pandas dataframe to store training set's min-max vals
min_max = pd.DataFrame({'min':[], 'max':[]})

for feature in X.columns:
    min_max.loc[feature] = [X_training[feature].min(), X_training[feature].max()]

# iterating through each test case
# for testcase in range(len(X_testing)):
#     X_temp = X_training
#
#     # normalize data
#     for feature in X.columns:
#         print(max_X[feature])
#         upper = max(max_X[feature], X_testing[feature][testcase])
#         print(upper)
#         # min_X = min (X_temp[feature].min)



    # X_testing[testcase] = processing.normalize(X_testing[testcase], X_training[testcase])


# max_trainData = max(X_training)
#
#
# processing.normalize()
#
#
# # MODEL-----
# for testcase in range(len(X_testing)):
#
#
#     # TESTING MODEL-----
#     distances = processing.euclid_dist(X_training, X_testing.iloc[testcase])
#
#     order = np.argsort(distances)[0:K]                  # indices of closest K training samples (ascending order)
#
#

#
#
# for feature in X.columns:
#     X[feature] = processing.normalize(X[feature], X[feature].max(), X[feature].min())