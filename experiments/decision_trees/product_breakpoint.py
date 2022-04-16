"""
    Comparing Anchors and LIME on text data.
"""

# import
import numpy as np
import pandas as pd

import spacy  # nlp for Anchors

from sklearn.feature_extraction.text import CountVectorizer

from anchor import anchor_text  # official Anchors
from lime import lime_text  # official LIME

import os

from experiments import utils
from dataset.dataset import Dataset
from experiments.models import Product

# output
import pickle
import logging
from matplotlib import pyplot as plt
import seaborn as sns

# specific parameters
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.set_theme(style='darkgrid')
lw = 3  # linewidth
fs = 25  # fontsize
sns.set(font_scale=2)

description = 'Product of indicator functions, returning 1 if "very" and "good" are present. Breakpoint case.'

DATASET = 'restaurants'
dir = 'experiments'
model_class = 'decision_trees'
MODEL = 'product_breakpoint'
doc = "Food is very very very very very good!"

# DATA
path = os.getcwd().replace(model_class, '').replace(dir, 'dataset')
data = Dataset(DATASET, path)
df, X, y = data.df, data.X, data.y

N_runs = 100

# Vectorization
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(X)

# CLASSIFIER
w_list = ['very', 'good']
model = Product(vectorizer, w_list)

# classes of the model
class_names = ["Dislike", "Like"]

# initialize the explainers
nlp = spacy.load('en_core_web_sm')  # anchors need it
anchor_explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
lime_explainer = lime_text.LimeTextExplainer(class_names=class_names)

example = data.preprocess(doc)

cols = utils.unique(example.split())
lime_res = pd.DataFrame(columns=cols)
anchors_res = []

# NOTE: LIME requires predict_proba, Anchors use predict
for i in range(N_runs):
    print('> Run ' + str(i + 1) + '/' + str(N_runs))
    lime_exp = lime_explainer.explain_instance(example, model.predict_proba)
    anchors_exp = anchor_explainer.explain_instance(example, model.predict).names()
    print(anchors_exp)
    print(utils.lime_dict(lime_exp))
    lime_res.loc[i] = utils.lime_mapper(lime_exp)
    anchors_exp.sort()
    anchor_as_string = ', '.join(w for w in anchors_exp)
    anchors_res.append(anchor_as_string)

multiplicities = utils.count_multiplicities(X, data.preprocess(doc))
info = {'Description': description,
        'Model': MODEL, 'Words': w_list, 'Dataset': DATASET, 'Example': doc, 'N_runs': N_runs,
        'Multiplicities': multiplicities, 'Anchors': anchors_res, 'LIME': lime_res}

res = os.getcwd().replace(dir, 'results')
pickle.dump(info, open(os.path.join(res, str(MODEL + '.p')), 'wb'))

# Figure
fig, axes = plt.subplots(nrows=1, ncols=2,
                         gridspec_kw={'width_ratios': [2.5, 1.5]},
                         )
fig.tight_layout()
fig.suptitle(info['Example'], fontsize=fs)

# LIME
sorted_index = lime_res.median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=lime_res[sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')

# Anchors
anchors_res.sort()
sns.countplot(anchors_res, ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
filename = os.path.join(res, str(MODEL))
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)

# log
filename = os.path.join(res, str(MODEL) + '.log')
with open(filename, 'w'):
    pass

logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info.keys():
    logging.info(str(key) + ': ' + str(info[key]))
