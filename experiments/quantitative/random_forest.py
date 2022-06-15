"""
Jaccard similarity between Anchors explanations and first len(A) words by LIME.
"""

import os
import sys
import pickle
import dill
import time

# import
import numpy as np
import spacy  # nlp for Anchors
from anchor import anchor_text
from lime import lime_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sys.path.append(os.getcwd().replace('anchors_text', ''))
from dataset.dataset import Dataset
from experiments import utils

np.random.seed(42)

# DATA
path = os.getcwd().replace('anchors_text', 'dataset')
DATASET = 'yelp'
# data = Dataset(DATASET, path)
data = Dataset(DATASET, os.path.join(os.getcwd().replace('quantitative', '').replace('experiments', ''), 'dataset'))
# data = Dataset(DATASET, os.path.join(os.getcwd(), 'dataset'))
df, X, y = data.df, data.X, data.y

X_train, X_test, y_train, y_test = train_test_split(X, y)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# CLASSIFIER
model = RandomForestClassifier()
model.fit(train_vectors, y_train)

# pipeline: Vectorizer + Model
clf = make_pipeline(vectorizer, model)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# initialize the explainers
class_names = ["Negative", "Positive"]
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names)
lime_explainer = lime_text.LimeTextExplainer(class_names=class_names)

# NOTE: LIME requires predict_proba, Anchors use predict
# We explain positive predictions
corpus = np.asarray(X_test)[clf.predict(X_test) == 1]

results_path = os.getcwd()  # os.path.join(os.getcwd().replace('anchors_text', 'results'), 'anchors_text')
exp_path = os.path.join(results_path, str('exp_rf_' + DATASET + '.p'))
time_path = os.path.join(results_path, str('time_rf_' + DATASET + '.p'))
corpus_path = os.path.join(results_path, str('corpus_rf_' + DATASET + '.p'))
probs_path = os.path.join(results_path, str('probs_rf_' + DATASET + '.p'))
pickle.dump(corpus, open(corpus_path, 'wb'))

times, exps, probs = [], [], []
for i, example in np.ndenumerate(corpus):
    print('\nExample {} / {}: {}'.format(str(i[0] + 1), str(len(corpus)), example))
    proba = clf.predict_proba([example])
    print(proba)
    anchor_t0 = time.time()
    anchor_exp = anchor_explainer.explain_instance(str(example), clf.predict).names()
    anchor_tf = time.time() - anchor_t0
    print(anchor_exp)
    lime_t0 = time.time()
    lime_exp = utils.lime_dict(lime_explainer.explain_instance(str(example), clf.predict_proba, num_features=len(str(example))))
    lime_tf = time.time() - lime_t0
    print(lime_exp)
    times.append([anchor_tf, lime_tf])
    exps.append([anchor_exp, lime_exp])
    probs.append(proba)
    dill.dump(exps, open(exp_path, 'wb'))
    dill.dump(times, open(time_path, 'wb'))
    dill.dump(probs, open(probs_path, 'wb'))
