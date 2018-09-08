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
#HOUSING_PATH2 = os.path.join("datasets", "tw")



#CLEANER
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#loading data
def load_data_test_emb(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "test_emb_and_messages2.csv")
    return pd.read_csv(csv_path)

#load embeddings training def load_data(housing_path=HOUSING_PATH):
def load_data ( housing_path=HOUSING_PATH ):
    csv_path = os.path.join(housing_path, "embeddingstraining2.csv")
    return pd.read_csv(csv_path)

fb = clean_dataset(load_data())
X_training_embeddings = fb.drop("class", axis=1).drop("link", axis=1)
y = fb["class"]


#LOAD TEST embeddings
test_embeddings = load_data_test_emb()
X_test_embeddings = test_embeddings.drop("link", axis=1).drop("message", axis=1)
X_test_messages = test_embeddings["message"]
labels = test_embeddings["link"]


#LOAD training TESTUALE per tfIdf
def load_data_text(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "textTrainingFull.csv")
    return pd.read_csv(csv_path)

fb2 = (load_data_text())
X2 = fb2["message"]



###########################  TFIDF ####################################

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


tf = TfidfVectorizer(tokenizer=lambda doc: doc,lowercase=True, analyzer='word', stop_words = None, ngram_range = [1, 3])
 #mydoclist is a list having my **Text** which im reading from a CSV

X_tf = tf.fit_transform(X2)

X_test_messages_tf = tf.transform(X_test_messages)
X_embeddings_tf_test = sp.hstack((X_test_embeddings, X_test_messages_tf))

###########################################################################
#Transform training for VotingClassifier fitting

x_embeddings_tf_train = sp.hstack((X_training_embeddings, X_tf))


###########################################################################

mlp_clf = joblib.load("final_models/TW/01.MLP_embeddingsText.pkl")
svm_clf = joblib.load("final_models/TW/02.SVM_embeddingsText.pkl")
rand_clf = joblib.load("final_models/TW/03.RandomForest_embeddingsText.pkl")

named_estimators = [
    ("mlp_clf", mlp_clf),
    ("svm_clf", svm_clf),
    ("rand_clf" , rand_clf)
]


voting_clf = joblib.load("final_models/TW/00.VotingClassifier.pkl") #VotingClassifier(named_estimators)


#FIT voting classifier on training
#voting_clf.fit(x_embeddings_tf_train, y)

#Save the model
#joblib.dump(voting_clf, "00.VotingClassifier.pkl")

#Test on training
y_pred = voting_clf.predict(x_embeddings_tf_train)
sc = f1_score(y_pred, y)

print("F1 score on training voting",sc)

y_pred_test = voting_clf.predict(X_embeddings_tf_test)


file = open("final_models/TW/TASK02_final_tw.tsv","w")

for index, item in enumerate(y_pred_test):
    print(index, labels[index])
    print ( index , item )
    file.write(str(labels[index]) +"	"+str(int(item))+"\n")

file.close()

