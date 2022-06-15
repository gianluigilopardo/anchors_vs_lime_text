"""
    Interpretability Comparison for Text Data: Anchors vs LIME on simple models for sentiment analysis.
"""

import os
import dill

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

import operator

from experiments.utils import compute_similarity
from experiments.utils import get_tfidf
from experiments.utils import get_corpus_coefficients
from experiments.utils import rank_by_coefs
from experiments.quantitative.ell_index import Evaluation

DATASET = 'yelp'
MODEL = 'logistic'

path = os.path.join(os.getcwd(), 'results')

times = np.array(dill.load(open(os.path.join(path, str('time_{}_{}.p'.format(MODEL, DATASET))), 'rb')))
print(times)
corpus = np.array(dill.load(open(os.path.join(path, str('corpus_{}_{}.p'.format(MODEL, DATASET))), 'rb')))
print(corpus)
probs = np.array(dill.load(open(os.path.join(path, str('probs_{}_{}.p'.format(MODEL, DATASET))), 'rb')))
print(probs)
exps = np.array(dill.load(open(os.path.join(path, str('exp_{}_{}.p'.format(MODEL, DATASET))), 'rb')), dtype=object)
exps[:, 1] = [dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True)) for dic in exps[:, 1]]
print(exps)
vectorizer = dill.load(open(os.path.join(path, str('vectorizer_{}_{}.p'.format(MODEL, DATASET))), 'rb'))
logistic = dill.load(open(os.path.join(path, str('model_{}_{}.p'.format(MODEL, DATASET))), 'rb'))

explanations = pd.DataFrame(columns=['Example', 'Proba', 'Anchors', 'LIME', 'Anchors_time', 'LIME_time', 'Similarity',
                                     'ell_Anchors', 'ell_LIME'])
explanations.Example = corpus
explanations.Proba = probs[:, 0, 1]
explanations.Anchors = exps[:, 0]
explanations.LIME = exps[:, 1]
explanations.Anchors_time = times[:, 0]
explanations.LIME_time = times[:, 1]
explanations.Similarity = compute_similarity(exps[:, 0], exps[:, 1])
eval = Evaluation(corpus, logistic, vectorizer)
ell_anchors, ell_lime = eval.ell_index(exps[:, 0], exps[:, 1])
explanations.ell_Anchors = ell_anchors
explanations.ell_LIME = ell_lime

print(explanations)
print(explanations.describe())
