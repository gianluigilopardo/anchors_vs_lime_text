import numpy as np


class Product:
    def __init__(self, vectorizer, w_list):
        self.vectorizer = vectorizer
        self.w_list = w_list
        self.ids = [vectorizer.vocabulary_[w] for w in w_list]

    # CLASSIFIER
    def predict_proba(self, docs):
        outs = np.zeros((len(docs), 2))
        vect = self.vectorizer.transform(docs)
        for i, x in enumerate(docs):
            if all([vect[i, j] > 0 for j in self.ids]):
                outs[i, 1] = 1
        outs[:, 0] = 1 - outs[:, 1]
        return outs.astype(int)

    def predict(self, docs):
        return self.predict_proba(docs)[:, 1]


class DTree:
    def __init__(self, vectorizer, w_lists):
        self.vectorizer = vectorizer
        self.w_lists = w_lists
        self.ids = {}
        for i in range(len(w_lists)):
            self.ids[i] = [vectorizer.vocabulary_[w] for w in w_lists[i]]

    # CLASSIFIER
    def predict_proba(self, docs):
        outs = np.zeros((len(docs), 2))
        vect = self.vectorizer.transform(docs)
        for i, x in enumerate(docs):
            if any([all([vect[i, j] > 0 for j in self.ids[l]]) for l in range(len(self.ids))]):
                outs[i, 1] = 1
        outs[:, 0] = 1 - outs[:, 1]
        return outs.astype(int)

    def predict(self, docs):
        return self.predict_proba(docs)[:, 1]


class Logistic:
    def __init__(self, vectorizer, w_dict, lambda_0):
        self.vectorizer = vectorizer
        self.w_dict = w_dict
        self.lambdas = {}
        for w in w_dict.keys():
            self.lambdas[vectorizer.vocabulary_[w]] = w_dict[w]
        print(self.lambdas)
        self.lambda_0 = lambda_0

    # CLASSIFIER
    def predict_proba(self, docs):
        outs = np.zeros((len(docs), 2))
        vect = self.vectorizer.transform(docs)
        for i, x in enumerate(docs):
            t = self.lambda_0 + np.sum(vect[i, j] * self.lambdas[j] for j in self.lambdas.keys())
            outs[i, 1] = 1 / (1 + np.exp(-t))
        outs[:, 0] = 1 - outs[:, 1]
        return outs

    def predict(self, docs):
        return np.round(self.predict_proba(docs)[:, 1])

