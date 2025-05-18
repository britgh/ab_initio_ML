# K-Nearest Neighbors Machine Learning Model
# britgh last updated: 05.18.25

import processing
import numpy as np
from ucimlrepo import fetch_ucirepo                     # import iris dataset (given by UCIML)
from sklearn.model_selection import train_test_split    # for consistent shuffle + data split

# DATASET-----
iris = fetch_ucirepo(id=53)
X = iris.data.features.drop('sepal width', axis=1)      # pd dataframes, remove sepal width feature
Y = iris.data.targets

# DATA SPLIT (sklearn library used here)-----
X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = 0.2)

scores = np.empty(shape=len(X_testing))                 # track predictions: correct = 1, else 0
K = 5                                                   # chosen number of neighbors

# NORMALIZING DATA-----
processing.normalize()


# MODEL-----
for testcase in range(len(X_testing)):


    # TESTING MODEL-----
    distances = processing.euclid_dist(X_training, X_testing.iloc[testcase])

    order = np.argsort(distances)[0:K]                  # indices of closest K training samples (ascending order)





for feature in X.columns:
    X[feature] = processing.normalize(X[feature], X[feature].max(), X[feature].min())