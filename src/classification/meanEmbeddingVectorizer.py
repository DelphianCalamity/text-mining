import numpy as np
from preprocess.preprocess import *


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = []

        X = Preprocessor().tokenize_articles(X)
        for words in X:
            article = [np.zeros(self.dim)]
            for w in words:
                if w in self.word2vec:
                    article.append(self.word2vec[w])

            transformed.append(np.mean(article, axis=0))

        return np.array(transformed)