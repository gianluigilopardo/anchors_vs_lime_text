"""
    Comparing Anchors and LIME on text data.
"""

# import
import numpy as np
import pandas as pd

import spacy  # nlp for Anchors

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from anchor import anchor_text  # official Anchors

import os

from experiments import utils
from dataset.dataset import Dataset


# output
import pickle
np.random.seed(42)

description = 'Impact of the intercept on Anchors explanations'

DATASET = 'yelp'
dir = 'experiments'
model_class = 'logistic'
MODEL = 'intercept'

# DATA
path = os.getcwd().replace(model_class, '').replace(dir, 'dataset')
data = Dataset(DATASET, path)
df, X, y = data.df, data.X, data.y

X_train, X_test, y_train, y_test = train_test_split(X, y)

# CountVectorizer transformation
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# CLASSIFIER
model = LogisticRegression()
model.fit(train_vectors, y_train)

# pipeline: Vectorizer + Model
c = make_pipeline(vectorizer, model)

y_pred = c.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

doc = df.text[13]
print(doc)
print(c.predict_proba([doc]))

N_runs = 100

example = data.preprocess(doc)
cols = utils.unique(example.split())

anchors_data = pd.DataFrame(columns=['Anchor', 'Run', 'Shift'])
runs, shifts_, anchors = [], [], []

# classes of the model
class_names = ["Dislike", "Like"]

# initialize the explainers
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)

shifts = np.linspace(-2, 3, 51)  # 3
intercept = model.intercept_[0]

multiplicities = utils.count_multiplicities(data.preprocess(doc))
coefficients = utils.coefficients(vectorizer, model, doc)

print(multiplicities)
print(coefficients)

for shift in shifts:
    model.intercept_ = [intercept - shift]
    print(model.intercept_)
    anchors_res = []
    for i in range(N_runs):
        print('> Run: {} / {} - Shift: {}'.format(str(i+1), str(N_runs), shift))
        print(c.predict_proba([example]))
        anchors_exp = anchor_explainer.explain_instance(example, c.predict).names()
        print(anchors_exp)
        anchors_exp.sort()
        anchor_as_string = ', '.join(w for w in anchors_exp)
        anchors_res.append(anchor_as_string)
        for ele in set(anchors_res):
            anchors.append(ele)
            shifts_.append(shift)
            runs.append(i)

anchors_data.Anchor = anchors
anchors_data.Run = runs
anchors_data.Shift = shifts_


info = {'Description': description,
        'Model': MODEL, 'Dataset': DATASET, 'Example': doc, 'N_runs': N_runs,
        'Coefficients': coefficients, 'Intercept': intercept, 'Shifts': shifts, 'Multiplicities': multiplicities,
        'Anchors': anchors_data,
        }

res = os.getcwd().replace(dir, 'results')
pickle.dump(info, open(os.path.join(res, str(MODEL + '.p')), 'wb'))

