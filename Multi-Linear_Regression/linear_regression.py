# Multi-Linear Regression Machine Learning Model
# britgh last updated: 05.27.25

import preprocessing
import correlation
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo                         # MPG dataset by UCIML
from sklearn.model_selection import KFold                   # help implement K-Fold cross-validation