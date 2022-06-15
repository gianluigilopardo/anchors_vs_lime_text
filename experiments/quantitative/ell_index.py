from experiments.utils import jaccard_similarity
import operator
import numpy as np


class Evaluation:
    def __init__(self, corpus, model, vectorizer):
        self.ell_lime = None
        self.ell_anchors = None
        self.corpus = corpus
        self.model = model
        self.vectorizer = vectorizer
        self.coef_dicts = self.get_coefs()

    def ell_index(self, anchor_lists, lime_dicts):
        ell_anchors, ell_lime = [], []
        for i in range(len(self.corpus)):
            anchor_list = list(set(anchor_lists[i]))
            print(self.coef_dicts[i])
            coefs_list = list(self.coef_dicts[i].keys())[:len(anchor_list)]
            print(dict(sorted(lime_dicts[i].items(), key=operator.itemgetter(1), reverse=True)))
            lime_list = list(dict(sorted(lime_dicts[i].items(), key=operator.itemgetter(1), reverse=True)).keys())[:len(anchor_list)]
            ell_anchors.append(jaccard_similarity(anchor_list, coefs_list))
            ell_lime.append(jaccard_similarity(lime_list, coefs_list))
            print(self.corpus[i])
            print(coefs_list)
            print(anchor_list)
            print(ell_anchors[i])
            print(lime_list)
            print(ell_lime[i])
            print('\n')
        self.ell_anchors = ell_anchors
        self.ell_lime = ell_lime
        return ell_anchors, ell_lime

    def get_coefs(self):
        corpus = self.corpus
        model = self.model
        vectorizer = self.vectorizer
        tfidf_matrix = vectorizer.transform(corpus)
        words = vectorizer.get_feature_names()
        tfidf_dicts, linear_dicts, coef_dicts = [], [], []
        for doc in range(len(corpus)):
            feature_index = tfidf_matrix[doc, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
            tfidf = {w: s for w, s in [(words[i], s) for (i, s) in tfidf_scores]}
            tfidf_dicts.append(tfidf)
            linear_coefs = {word: model.coef_[0, words.index(word)] for word in corpus[doc].split() if word in words}
            linear_dicts.append(linear_coefs)
            coefs = {w: linear_coefs[w] * tfidf[w] for w in list(linear_coefs.keys())}
            coefs = dict(sorted(coefs.items(), key=operator.itemgetter(1), reverse=True))
            coef_dicts.append(coefs)
        return np.array(coef_dicts)
