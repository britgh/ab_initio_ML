# K-Nearest Neighbors Machine Learning Model
# britgh last updated: 05.27.25

import processing
import visualization
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo                         # iris dataset (given by UCIML)
from sklearn.model_selection import train_test_split        # consistent shuffle + data split

# DATASET-----
iris = fetch_ucirepo(id=53)
X = iris.data.features.drop('sepal width', axis=1)          # pd dataframes, remove sepal width feature
Y = iris.data.targets

# DATA SPLIT (sklearn library used here)-----
X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = 0.2)

scores = np.empty(shape=len(X_testing))                     # track predictions: correct = 1, else 0
K = 5                                                       # chosen number of neighbors

# NORMALIZING DATA-----
min_max = pd.DataFrame({'min':[], 'max':[]})                # pandas df to store train set's min + max vals

for feature in X.columns:
    min_max.loc[feature] = [X_training[feature].min(), X_training[feature].max()]

# iterating through test cases
all_preds = pd.DataFrame({'pred':[], 'real':[], 'score':[]})
predictions = np.empty(shape=len(Y_testing), dtype='U15')
X_testing_copy = X_testing.copy()                           # unnormalized features for data viz

for testcase in range(len(Y_testing)):
    case = X_testing.iloc[testcase]
    X_temp = X_training.copy()

    for feature in X.columns:
        min_x = min_max.at[feature, 'min']                  # training data min/max
        max_x = min_max.at[feature, 'max']

        min_x = processing.minimum(min_x, case[feature])    # min/max including testcase
        max_x = processing.maximum(max_x, case[feature])

        X_temp[feature] = processing.normalize(X_temp[feature], max_x, min_x)   # normalize training + testing
        case[feature] = processing.normalize(case[feature], max_x, min_x)

    # MODEL TESTING-----
    distances = processing.euclid_dist(X_temp, case)        # euclidean dist. (train set + testcase)
    order = distances.sort_values()[0:K].index              # indices of closest K neighbors

    predictions[testcase] = processing.mode(np.array(Y_training.loc[order]))    # calculate + store predicted class

    # PERFORMANCE EVALUATION-----
    if predictions[testcase] == Y_testing.iloc[testcase].get('class') :
        all_preds.loc[testcase] = [predictions[testcase], Y_testing.iloc[testcase].get('class'), 1]
    else:
        print("MISS (case", testcase,")! PREDICT:", predictions[testcase], "ACTUAL:", Y_testing.iloc[testcase].get('class'))
        print(X_testing.iloc[testcase])
        all_preds.loc[testcase] = [predictions[testcase], Y_testing.iloc[testcase].get('class'), 0]

processing.accuracy(all_preds)
visualization.plot(X_training, X_testing_copy, Y_training, all_preds)           # plots all training + testing data