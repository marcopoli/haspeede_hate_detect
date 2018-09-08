# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import pandas as pd
from sklearn.externals import joblib
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import VotingClassifier

####################### Data LOADING ########################################
HOUSING_PATH = os.path.join("datasets", "tw")
HOUSING_PATH2 = os.path.join("datasets", "fb")

#CLEANER
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#LOAD training TESTUALE per tfIdf
def load_data_text(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "textTrainingFull.csv")
    return pd.read_csv(csv_path)

fb2 = load_data_text()
X2 = fb2["message"]
y = fb2["class"]

#load test text
def load_data_text_test(housing_path2=HOUSING_PATH2):
    csv_path = os.path.join(housing_path2, "textTestFull.csv")
    return pd.read_csv(csv_path)

fb = load_data_text_test()
X_test_messages = fb["message"]
labels = fb["id"]

###########################  TFIDF ####################################

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


tf = TfidfVectorizer(tokenizer=lambda doc: doc,lowercase=True, analyzer='word', stop_words = None, ngram_range = [1, 3])
 #mydoclist is a list having my **Text** which im reading from a CSV

X_tf = tf.fit_transform(X2)

X_test_messages_tf = tf.transform(X_test_messages)

from sklearn.svm import SVC
svc_clf = SVC(C=1.76800710431488, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=5, gamma=0.1949764030136127,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)




#joblib.load("final_models/FB/04.SVM_FULLText.pkl")

#FIT voting classifier on training
svc_clf.fit(X_tf, y)

#Save the model
#joblib.dump(voting_clf, "00.VotingClassifier.pkl")

#Test on training
y_pred = svc_clf.predict(X_tf)
sc = f1_score(y_pred, y)

print("F1 score on training",sc)

y_pred_test = svc_clf.predict(X_test_messages_tf)
print(y_pred_test)

file = open("final_models/TW/TASK3.2_2_twfb.csv","w")

for index, item in enumerate(y_pred_test):
    print(index, labels[index])
    print ( index , item )
    file.write(str(labels[index]) +","+str(int(item))+"\n")

file.close()

