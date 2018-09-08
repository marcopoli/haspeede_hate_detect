# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd

# Common imports
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib

HOUSING_PATH = os.path.join("datasets", "fb")

#loading data
def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "embeddingstraining.csv")
    return pd.read_csv(csv_path)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

fb = clean_dataset(load_data())

X = fb.drop("class", axis=1).drop("link", axis=1)
y = fb["class"]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=46558)

for train_index, test_index in split.split(fb, fb["class"]):
    strat_train_set = fb.loc[train_index]
    strat_train_set_y = y.loc[train_index]

    strat_test_set = fb.loc[test_index]
    strat_test_y = y.loc[test_index]

strat_train_set_b = strat_train_set.copy()
strat_train_set_y_b = strat_train_set_y.copy()

strat_test_set_b = strat_test_set.copy()
strat_test_y_b =     strat_test_y.copy()

strat_train_set = strat_train_set.drop("class", axis=1).drop("link", axis=1)
strat_test_set = strat_test_set.drop("class", axis=1).drop("link", axis=1)


#GRID RANDOM FOREST
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

from scipy.stats import expon, reciprocal, uniform, randint

random_grid = {
        'alpha': reciprocal(0.001, 1),
        'hidden_layer_sizes': randint(500, 10000),
        'max_iter': randint(100, 1000),
    }

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

rf = MLPClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=34456, n_jobs = -1, scoring='f1')
# Fit the random search model

rf_random.fit(strat_train_set, strat_train_set_y)
y_pred = rf_random.predict(strat_test_set)
from sklearn.metrics import f1_score
scor = f1_score(y_pred,strat_test_y)
print (rf_random.best_estimator_)
print (scor)


joblib.dump(rf_random.best_estimator_, "04.MLP_embeddings_3.pkl")