# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd

# Common imports
import numpy as np
from sklearn.externals import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit


HOUSING_PATH = os.path.join("datasets", "tw")

#loading data
def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "embeddingstraining2.csv")
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

def load_data_text(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "textTrainingFull.csv")
    return pd.read_csv(csv_path)

fb2 = (load_data_text())

X2 = fb2["message"]
#X2_c = X2.copy()

#import preprocessor as p
#for index, message in enumerate(X2):
#    newmex = p.clean(message)
 #   X2_c[index] = newmex

#X2 = X2_c
#print(X2)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=46558)

for train_index, test_index in split.split(fb, fb["class"]):

    strat_train_set = fb.loc[train_index]
    strat_train_set_y = y.loc[train_index]
    strat_train_set_text = X2.loc[train_index]

    strat_test_set = fb.loc[test_index]
    strat_test_y = y.loc[test_index]
    strat_test_set_text = X2.loc[test_index]


strat_train_set_b = strat_train_set.copy()
strat_train_set_y_b = strat_train_set_y.copy()

strat_test_set_b = strat_test_set.copy()
strat_test_y_b =     strat_test_y.copy()

strat_train_set = strat_train_set.drop("class", axis=1).drop("link", axis=1)
strat_test_set = strat_test_set.drop("class", axis=1).drop("link", axis=1)


########################### SPAM FILTER USING TF ####################################
import os


from sklearn.feature_extraction.text import TfidfVectorizer


tf = TfidfVectorizer(tokenizer=lambda doc: doc,lowercase=True, analyzer='word', stop_words = None, ngram_range = [1, 4])
 #mydoclist is a list having my **Text** which im reading from a CSV
X2_c = X2.copy()

tfidf_matrix =  tf.fit_transform(X2)


import scipy.sparse as sp


tf_strat_train_set= tf.transform(strat_train_set_text)
tf_strat_test_set= tf.transform(strat_test_set_text)


#X2_tf = tf.transform(X2_c)

x_embeddings_tf_train = sp.hstack((X, tfidf_matrix))
x_embeddings_tf_test = sp.hstack((strat_test_set, tf_strat_test_set))
from scipy.stats import expon, reciprocal, uniform, randint


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
random_grid = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(1, 50),
        'gamma': expon(scale=1.0)
    }

from sklearn.svm import SVC

rf = SVC()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 70, cv = 3, verbose=3, random_state=34456, n_jobs = -1, scoring='f1')
# Fit the random search model

rf_random.fit(x_embeddings_tf_train, y)
#rf_random = joblib.load("01.MLP_embeddingsText.pkl")
joblib.dump(rf_random.best_estimator_, "02.SVM_embeddingsText_TWITTER.pkl")

print ("F1 best",rf_random.best_score_)


y_pred = rf_random.predict(x_embeddings_tf_test)
from sklearn.metrics import f1_score, precision_score, recall_score
scor = f1_score(y_pred,strat_test_y)
print (rf_random)
print ("F1",scor)

scor = precision_score(y_pred,strat_test_y)
print ("Precision",scor)

scor = recall_score(y_pred,strat_test_y)
print ("Recall",scor)






