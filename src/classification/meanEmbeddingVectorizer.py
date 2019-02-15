import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = []
        for words in X:
            article = [np.zeros(self.dim)]
            for w in words:
                if w in self.word2vec:
                    article.append(self.word2vec[w])

            transformed.append(np.mean(article, axis=0))

        # print(np.array(transformed))
        return np.array(transformed)

        # return np.array([
        #     np.mean([self.word2vec[w] for w in words if w in self.word2vec]
        #             or [np.zeros(self.dim)], axis=0)
        #     for words in X
        # ])