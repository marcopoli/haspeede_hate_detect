# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import pandas as pd

# Common imports
import numpy as np
from sklearn.externals import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

#X = fb.loc["f001":"f500"]
y = fb["class"]

def load_data_text(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "textTraining.csv")
    return pd.read_csv(csv_path)

fb2 = (load_data_text())

X = fb2["message"]


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=46558)

for train_index, test_index in split.split(fb, fb["class"]):

    strat_train_set = fb.loc[train_index]
    strat_train_set_y = y.loc[train_index]
    strat_train_set_text = X.loc[train_index]

    strat_test_set = fb.loc[test_index]
    strat_test_y = y.loc[test_index]
    strat_test_set_text = X.loc[test_index]


strat_train_set_b = strat_train_set.copy()
strat_train_set_y_b = strat_train_set_y.copy()

strat_test_set_b = strat_test_set.copy()
strat_test_y_b =     strat_test_y.copy()

strat_train_set = strat_train_set.drop("class", axis=1).drop("link", axis=1)
strat_test_set = strat_test_set.drop("class", axis=1).drop("link", axis=1)


########################### SPAM FILTER USING TF ####################################
import os
import tarfile
from six.moves import urllib
import email
import email.policy
from sklearn.model_selection import train_test_split
import re
from html import unescape
from collections import Counter
#Import stemmer
import nltk
#Import urlextractor
import urlextract
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Data Trsnsformer
class TextToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
        self.url_extractor = urlextract.URLExtract()
        self.stemmer = nltk.PorterStemmer()

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        from nltk import ngrams
        X_transformed = []
        for email in X:
            text = email or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and self.url_extractor is not None:
                urls = list(set(self.url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " <URL> ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', '<NUMBER>', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)

            bigrams_count = Counter(ngrams(text.split(), 2))
            word_counts = Counter(text.split())

            if self.stemming and self.stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            #X_transformed.append(word_counts)

            #bigram_counts = Counter()
            #for words, count in bigrams_count.items():
            #    bigram_counts[words] += count
            #bigram_counts = bigram_counts
            X_transformed.append((word_counts +bigrams_count))

        return np.array(X_transformed)

X_c = X.copy()
#Transformation
X_wordcounts = TextToWordCounterTransformer().fit_transform(X_c)
print(X_wordcounts)
#OUTPUT = array([Counter({'wrote': 1, 'chuck': 1, 'murcko': 1, 'stuff': 1, 'yawn': 1, 'r': 1}),...

#Transform to vectors
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 1)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=100000)
#Transform in vectors
X_vectors = vocab_transformer.fit_transform(X_wordcounts)

#We are now ready to train our first spam classifier! Let's transform the whole dataset
preprocess_pipeline = Pipeline([
    ("text_to_wordcount", TextToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X)
print(X_train_transformed.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

tf = TfidfVectorizer(tokenizer=lambda doc: doc,lowercase=True, analyzer='word', stop_words = None, ngram_range = [1, 3])
 #mydoclist is a list having my **Text** which im reading from a CSV
tfidf_matrix =  tf.fit_transform(X)
print(tfidf_matrix.shape)

import scipy.sparse as sp
x_final = tfidf_matrix #sp.hstack((X_train_transformed, tfidf_matrix))



print(x_final.shape)

tf_strat_train_set= tf.transform(strat_train_set_text)
tf_strat_test_set= tf.transform(strat_test_set_text)

x_embeddings_tf_train = sp.hstack((strat_train_set, tf_strat_train_set))
x_embeddings_tf_test = sp.hstack((strat_test_set, tf_strat_test_set))

from sklearn.svm import SVC
svc_clf_tt = SVC(C=1.76800710431488, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=5, gamma=0.1949764030136127,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

svc_clf_tt.fit(x_embeddings_tf_train,strat_train_set_y)
y_pred = svc_clf_tt.predict(x_embeddings_tf_test)
from sklearn.metrics import f1_score
scor = f1_score(y_pred,strat_test_y)
print ("F1 merge SVC",scor)

from sklearn.neural_network import MLPClassifier
mlp_clf_tt =MLPClassifier(activation='relu', alpha=0.5521952082781035, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=2220, learning_rate='constant',
       learning_rate_init=0.001, max_iter=184, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

mlp_clf_tt.fit(x_embeddings_tf_train,strat_train_set_y)
y_pred = mlp_clf_tt.predict(x_embeddings_tf_test)
from sklearn.metrics import f1_score
scor = f1_score(y_pred,strat_test_y)
print ("F1 merge",scor)

from sklearn.neural_network import MLPClassifier
rand_clf_tt =RandomForestClassifier(n_estimators=500)

rand_clf_tt.fit(x_embeddings_tf_train,strat_train_set_y)
y_pred = rand_clf_tt.predict(x_embeddings_tf_test)
from sklearn.metrics import f1_score
scor = f1_score(y_pred,strat_test_y)
print ("F1 merge rand",scor)



log_clf_text =joblib.load("01.Logistic_text.pkl") #LogisticRegression(random_state=53378)
#score = cross_val_score(log_clf_text, x_final, y, cv=2, verbose=3)

random_forest_clf_text = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
#score = cross_val_score(random_forest_clf_text, x_final, y, cv=2, verbose=3)

from sklearn.svm import LinearSVC
svm_clf_text = joblib.load("02.LinearSVC_text.pkl")#
#score = cross_val_score(svm_clf_text, x_final, y, cv=2, verbose=3)

#train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM.
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC , SVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = joblib.load("01.RandomForest_embeddings.pkl")

#RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#            max_depth=30, max_features='sqrt', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=2, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

#extra_trees_clf = ExtraTreesClassifier(random_state=985,n_estimators=3000)

#log_clf = joblib.load("03.Logistic_embeddings_2.pkl")

#LogisticRegression(C=3.531459626821106, class_weight=None, dual=False,
#             fit_intercept=True, intercept_scaling=1, max_iter=100,
#             multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#             solver='liblinear', tol=0.0001, verbose=0, warm_start=False)



svm_clf = joblib.load("02.SVM_embeddings_3.pkl")
#LinearSVC(random_state=6837,C=0.1)
from sklearn.linear_model import SGDClassifier
sgd_clf = joblib.load("05.SGD_embeddings_2.pkl")

mlp_clf =joblib.load("04.MLP_embeddings.pkl")
#MLPClassifier(activation='relu', alpha=0.22785847125610648, batch_size='auto',
#       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=1254, learning_rate='constant',
#       learning_rate_init=0.001, max_iter=862, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=None,
#       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#       verbose=False, warm_start=False)

#
estimators = [random_forest_clf,sgd_clf,svm_clf,mlp_clf]
estimators2 = []

for estimator in estimators2:
    print("Training the", estimator)
    estimator.fit(strat_train_set, strat_train_set_y)



log_clf_text.fit(tf_strat_train_set,strat_train_set_y)
random_forest_clf_text.fit(tf_strat_train_set,strat_train_set_y)
svm_clf_text.fit(tf_strat_train_set,strat_train_set_y)

print([estimator.score(strat_test_set, strat_test_y) for estimator in estimators])

#Implementing the voting function
from sklearn.ensemble import VotingClassifier

#named_estimators = [
#    ("random_forest_clf", random_forest_clf),
#   # ("extra_trees_clf", extra_trees_clf),
#    ("sgd_clf", sgd_clf),
#   # ("log_clf", log_clf),
#    ("svm_clf", svm_clf),
#    ("mlp_clf", mlp_clf)
#]

named_estimators = [
    ("rand_clf_tt", rand_clf_tt),
    ("mlp_clf_tt", mlp_clf_tt),
    ("svc_clf_tt", mlp_clf_tt),
]


voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(x_embeddings_tf_train, strat_train_set_y)
sc = voting_clf.f1_score(x_embeddings_tf_test, strat_test_y)
print("F1 score voting",sc)

y_pred = voting_clf.predict(x_embeddings_tf_test)

from sklearn.metrics import f1_score
scor = f1_score(y_pred,strat_test_y)
print ("F1 voting",scor)


#proviamo ad eliminare l'SVM
#del voting_clf.estimators_[2]
#del voting_clf.estimators_[3]
#sc = voting_clf.score(strat_test_set, strat_test_y)
#print("Score voting senza SVM",sc)

#Settiamo il voting a soft
#voting_clf.voting = "soft"
#sc = voting_clf.score(strat_test_set, strat_test_y)
#print("Score voting soft",sc)

#Stacking Ensemble using previous estimators
svc_clf_tt.fit(x_embeddings_tf_train,strat_train_set_y)
y_pred = svc_clf_tt.predict(x_embeddings_tf_test)


X_val_predictions = np.empty((len(strat_train_set), len(estimators)+3), dtype=np.float32)


for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(strat_train_set)

X_val_predictions[ : , 4 ] = rand_clf_tt.predict(x_embeddings_tf_train)
X_val_predictions[ : , 5 ] = mlp_clf_tt.predict (x_embeddings_tf_train)
X_val_predictions[ : , 6 ] = svc_clf_tt.predict(x_embeddings_tf_train)


rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, strat_train_set_y)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Get the predictions
X_test_predictions = np.empty((len(strat_test_set), len(estimators)+3), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(strat_test_set)

X_test_predictions[ : , 4 ] = rand_clf_tt.predict(x_embeddings_tf_test)
X_test_predictions[ : , 5 ] = mlp_clf_tt.predict (x_embeddings_tf_test)
X_test_predictions[ : , 6 ] = svc_clf_tt.predict(x_embeddings_tf_test)

y_pred = rnd_forest_blender.predict(X_test_predictions)
acc = accuracy_score(strat_test_y, y_pred)
print("Blender Acc:", acc)

pre = precision_score(strat_test_y, y_pred)
print("Blender Pre:", pre)

re = recall_score(strat_test_y, y_pred)
print("Blender Rec:", re)

f1 = f1_score(strat_test_y, y_pred)
print("Blender F1:", f1)




