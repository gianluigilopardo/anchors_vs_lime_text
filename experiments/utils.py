import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


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
def count_multiplicities(doc):
    counter = {}
    words = doc.split()
    for word in words:
        if word not in counter:
            counter[word] = 0
        counter[word] += 1
    return dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))


def coefficients(vectorizer, model, doc, n=20):
    dic = vectorizer.get_feature_names()
    p = vectorizer.build_preprocessor()
    p_doc = p(doc)
    t = vectorizer.build_tokenizer()
    words = t(p_doc)
    coefs = {}
    for word in words:
        if word in dic:
            coefs[word] = model.coef_[0, dic.index(word)]
        else:
            coefs[word] = 0
    return dict(sorted(coefs.items(), key=lambda item: item[1], reverse=True))


def varying_eps(proba, eps):
    def clf(docs):
        probs = proba(docs)[:, 1]
        return np.int64(probs >= 1 - eps)

    return clf


def dict_idf(vectorizer, doc):
    dic = vectorizer.get_feature_names()
    p = vectorizer.build_preprocessor()
    p_doc = p(doc)
    t = vectorizer.build_tokenizer()
    words = t(p_doc)
    idx = []
    idf = {}
    for w in words:
        if w in dic:
            idx.append(vectorizer.vocabulary_[w])
        else:
            idf[w] = 0
    global_dic = vectorizer.get_feature_names()
    idf_ = vectorizer.idf_
    for i in idx:
        idf[global_dic[i]] = idf_[i]
    return dict(sorted(idf.items(), key=lambda item: item[1], reverse=True))


# Jaccard similarity between two lists
def jaccard_similarity(list1, list2):
    # list1 = [re.sub(r'[^a-zA-Z]', '', s.lower()) for s in list1]
    # list2 = [re.sub(r'[^a-zA-Z]', '', s.lower()) for s in list2]
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def compute_similarity(anchors, lime_dics):
    lime_words, similarity = [], []
    for i in range(len(lime_dics)):
        lime_words.append(list(lime_dics[i].keys())[:len(anchors[i])])
        similarity.append(jaccard_similarity(anchors[i], lime_words[i]))
    return similarity


# preprocess new words as vectorizer
def get_words(vectorizer, doc):
    if vectorizer:
        p = vectorizer.build_preprocessor()
        p_doc = p(doc)
        t = vectorizer.build_tokenizer()
        words = t(p_doc)
    else:
        words = doc.split()
    return words


# returns linear coefficient for each word in a document
def get_coefficients(model, doc, vectorizer):
    dic = vectorizer.get_feature_names()
    words = get_words(vectorizer, doc)
    coefs = {}
    for word in dic:
        if word in dic:
            coefs[word] = model.coef_[0, dic.index(word)]
        else:
            coefs[word] = 0
    return dict(sorted(coefs.items(), key=lambda item: item[1], reverse=True))


# returns linear coefficients for each word in a corpus
def get_corpus_coefficients(model, corpus, vectorizer):
    tfidf_matrix = vectorizer.transform(corpus)
    linear_coefs = []
    for doc in corpus:
        linear_coefs.append(get_coefficients(model, doc, vectorizer))
    return linear_coefs


# returns TF-IDF term value for each word
def get_tfidf(corpus, vectorizer):
    tfidf_dicts = []
    tfidf_matrix = vectorizer.transform(corpus)
    feature_names = vectorizer.get_feature_names()
    for doc in range(len(corpus)):
        feature_index = tfidf_matrix[doc, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
        tfidf_dicts.append({w: s for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]})
    return tfidf_dicts


# return \lambda_j*tfidf_j for each document of the corpus
def rank_by_coefs(model, corpus, vectorizer):
    tfidf_matrix = vectorizer.transform(corpus)
    words = vectorizer.get_feature_names()
    coefs_corpus = []
    tfidf_dicts = get_tfidf(corpus, vectorizer)
    linear_coefs = get_corpus_coefficients(model, corpus, vectorizer)
    for i in range(len(corpus)):
        coefs = {w: linear_coefs[i][w] * tfidf_dicts[i][w] for w in words}  # \lambda_j*tfidf_j
        coefs = dict(sorted(coefs.items(), key=lambda item: item[1], reverse=True))
        coefs_corpus.append(coefs)
    return coefs_corpus


# computing \ell index
def ell_index(exps, coefs):
    words, similarity = [], []
    for i in range(len(coefs)):
        words.append(list(coefs[i].keys())[:len(exps[i])])
        similarity.append(jaccard_similarity(exps[i], words[i]))
    return similarity
