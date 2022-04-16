# dataset manager
import pandas as pd

import re

import os

datasets = ['restaurants', 'yelp']


class Dataset:
    def __init__(self, name, path):
        assert name in datasets, 'DATASET ' + str(name) + ' not available! Chose among ' + str(datasets)
        # light preprocessing: remove symbols, keep spaces, lowercase
        if name == 'restaurants':
            dataset = 'restaurants.tsv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep='\t')
            self.X = [self.preprocess(x) for x in list(self.df["Review"])]
            self.y = list(self.df["Liked"])
        elif name == 'yelp':
            dataset = 'positive_negative_reviews_yelp.csv'
            self.df = pd.read_csv(os.path.join(path, dataset), sep='|')
            self.X = [self.preprocess(x) for x in list(self.df["text"])]
            self.y = list(self.df["stars"])

    def preprocess(self, x):
        return re.sub('[^a-zA-Z\d\s]', '', x).lower()

