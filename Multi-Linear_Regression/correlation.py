# Preprocessing! Chose features using correlation coefficient + data visualization
# britgh last updated: 08.20.25

from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

auto_mpg = fetch_ucirepo(id=9)
X = auto_mpg.data.features
Y = auto_mpg.data.targets

# returns arr of rows w/ null values
def value_check (col, x) :
    feature = np.zeros(shape=len(x))

    count = 0
    for row in range(len(x)) :
        if pd.isnull(x.iloc[row, col]) :
            feature[count] = row
            count += 1

    return feature

# removing rows w/ null vals (for horsepower)
def remove_null (X, Y) :
    returned_arr = value_check(2, X)
    for value in returned_arr:
        if value != 0:
            Y = Y.drop(value)
            X = X.drop(value)

# comparing given feature column (x_value) with y column
def comparison (x_val, y_val, col) :        # find features w/ stronger correlations
    plt.scatter(x_val.iloc[:, col], y_val)
    plt.xlabel(X.columns[col])              # X-value column name
    plt.ylabel('MPG')
    plt.show()

# pearson correlation coefficient
def pcc (givenx, giveny, col) :
    x_val = givenx.iloc[:,col]
    y_val = giveny.iloc[:,0]
    count = x_val.shape[0]                      # number of vars

    x_mean = np.sum(x_val) / count              # mean of given feature column
    y_mean = np.sum(y_val) / count

    x_sd = np.sqrt(np.sum(np.square(x_val - x_mean)) / count-1)
    y_sd = np.sqrt(np.sum(np.square(y_val - y_mean)) / count-1)

    x_calc = (x_val-x_mean)/ x_sd
    y_calc = (y_val-y_mean)/ y_sd

    return (1/(count-1)) * np.sum(x_calc * y_calc)

# returns features correlated w/ >0.8 pcc w/ mpg
def dropping (X, Y) :
    arr = ['origin']                            # remove 'origin' b/c categorical

    for column in range(X.shape[1] - 1):
        # comparison(X, Y, column)
        pccVal = pcc(X, Y, column)
        print(X.columns[column], " pcc: ", pccVal)

        if abs(pccVal) < 0.8 :
            arr.append(X.columns[column])

    return arr