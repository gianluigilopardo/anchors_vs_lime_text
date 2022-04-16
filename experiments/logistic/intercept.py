"""
    Comparing Anchors and LIME on text data.
"""

# import
import numpy as np
import pandas as pd

import pickle

import spacy  # nlp for Anchors

from sklearn.feature_extraction.text import CountVectorizer

from anchor import anchor_text  # official Anchors
from lime import lime_text  # official LIME

import os

from experiments import utils
from dataset.dataset import Dataset

from experiments.models import Logistic

# output
import pickle
import logging
from matplotlib import pyplot as plt
import seaborn as sns

# specific parameters
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.set_theme(style='darkgrid')
lw = 3  # linewidth
ds = 15  # dot size
fs = 25  # fontsize
sns.set(font_scale=2)

description = 'Impact of the intercept on explanations'

DATASET = 'yelp'
dir = 'experiments'
model_class = 'logistic'
MODEL = 'intercept'

# DATA
path = os.getcwd().replace(model_class, '').replace(dir, 'dataset')
data = Dataset(DATASET, path)
df, X, y = data.df, data.X, data.y

doc = df.text[6]
print(doc)

N_runs = 5

# TF-IDF transformation
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(X)

example = data.preprocess(doc)
cols = utils.unique(example.split())

# CLASSIFIER
w_dict = {}
for w in cols:
    if len(w) >= 2:
        w_dict[w] = np.random.normal(loc=0, scale=0.3)
w_1 = 'haircut'
w_2 = 'daughter'
w_dict[w_1] = +1
w_dict[w_2] = +0.5

lambda_0s = - np.linspace(2, 12, 5)

lime_res = pd.DataFrame(columns=cols)
lime_data = pd.DataFrame(columns=[w_1, w_2, 'Run', 'Intercept'])
anchors_data = pd.DataFrame(columns=['Anchor', 'Run', 'Intercept'])
runs, intercepts, anchors = [], [], []
lime_wj, lime_wl, lime_intercepts, lime_runs = [], [], [], []

# classes of the model
class_names = ["Dislike", "Like"]

# initialize the explainers
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
lime_explainer = lime_text.LimeTextExplainer(class_names=class_names)

for lambda_0 in lambda_0s:
    print('\n' + str(lambda_0))
    model = Logistic(vectorizer, w_dict, lambda_0)

    anchors_res = []
    for i in range(N_runs):
        print('\t Run ' + str(i + 1) + '/' + str(N_runs))
        print(model.predict_proba([example]))
        lime_exp = lime_explainer.explain_instance(example, model.predict_proba, num_features=len(cols))
        anchors_exp = anchor_explainer.explain_instance(example, model.predict).names()
        print(anchors_exp)
        lime_dict = utils.lime_dict(lime_exp)
        lime_wj.append(lime_dict[w_1])
        lime_wl.append(lime_dict[w_2])
        lime_intercepts.append(lambda_0)
        lime_runs.append(i)
        anchors_exp.sort()
        anchor_as_string = ', '.join(w for w in anchors_exp)
        anchors_res.append(anchor_as_string)

        for ele in set(anchors_res):
            anchors.append(ele)
            intercepts.append(lambda_0)
            runs.append(i)

anchors_data.Anchor = anchors
anchors_data.Run = runs
anchors_data.Intercept = intercepts

lime_data[w_1] = lime_wj
lime_data[w_2] = lime_wl
lime_data.Run = lime_runs
lime_data.Intercept = - lime_intercepts

multiplicities = utils.count_multiplicities(X, data.preprocess(doc), [list(w_dict.keys())])
info = {'Description': description,
        'Model': MODEL, 'Dataset': DATASET, 'Example': doc, 'N_runs': N_runs,
        'Coefficients': w_dict, 'Lambda_0s': lambda_0s, 'Multiplicities': multiplicities,
        'Anchors': anchors_data,
        'LIME': lime_data}

res = os.getcwd().replace(dir, 'results')
pickle.dump(info, open(os.path.join(res, str(MODEL + '.p')), 'wb'))

# Anchors
runs, anchors, a_1, a_2, intercept = [], [], [], [], []
for i, row in anchors_data.iterrows():
    a_1.append(row.Anchor.count(w_1))
    a_2.append(row.Anchor.count(w_2))
    runs.append(row.Run)
    intercept.append(row.Intercept)

anchors_out = pd.DataFrame(columns=['Run', 'Intercept', 'Anchor', w_1, w_2])
anchors_out.Anchor = anchors
anchors_out.Run = runs
anchors_out[w_1] = a_1
anchors_out[w_2] = a_2
anchors_out.Intercept = - intercept

sns.lineplot(data=anchors_out, x='Intercept', y=a_1, label=w_1, drawstyle='steps-post', linewidth=lw, marker='o', markersize=ds)
sns.lineplot(data=anchors_out, x='Intercept', y=a_2, label=w_2, drawstyle='steps-post', linewidth=lw, marker='o', markersize=ds)
plt.legend()
plt.xlabel('- Intercept')
plt.ylabel('Occurrences')
plt.title('Anchors')

res = os.getcwd().replace(dir, 'results')
filename = os.path.join(res, str(MODEL))
plt.savefig(fname=str(filename + '_anchors.pdf'), bbox_inches='tight', pad_inches=0)

# LIME
lime_data.Intercept = - lime_data.Intercept

sns.lineplot(data=lime_data, x='Intercept', y=w_1, label=w_1, linewidth=lw, ci='sd')  # , marker='o', markersize=ds)
sns.lineplot(data=lime_data, x='Intercept', y=w_2, label=w_2, linewidth=lw, ci='sd')  # , marker='o', markersize=ds)
plt.legend()
plt.xlabel('- Intercept')
plt.ylabel('Coefficients')
plt.title('LIME')

filename = os.path.join(res, str(MODEL))
plt.savefig(fname=str(filename + '_lime.pdf'), bbox_inches='tight', pad_inches=0)
