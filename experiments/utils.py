import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


# mapper needed for lime
def lime_mapper(x):
    y = []
    x = x.as_map()[1]
    x.sort()
    y = [x[i][1] for i in range(len(x))]
    return np.array(y)


# mapper needed for lime
def lime_dict(x):
    x = x.as_list()
    return {x[i][0]: x[i][1] for i in range(len(x))}


# function to get unique values, keeping the order
def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


# count multiplicities
def count_multiplicities(data, doc, words=None):
    if not words:
        words = doc.split()
    else:
        words = [item for sublist in words for item in sublist]
    m = {}
    counter = CountVectorizer()
    counter.fit_transform(data)
    D = counter.vocabulary_
    counter.transform([doc])
    tf = counter.transform([doc]).toarray().flatten()
    m = {w: tf[D[w]] for w in words}
    return m