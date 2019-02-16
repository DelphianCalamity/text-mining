from sklearn.decomposition import TruncatedSVD


class SvdVariancePrinter(object):
    def __init__(self, svd):
        self.svd = svd

    def fit(self, x, y):
        return self

    def transform(self, x):
        print(self.svd.explained_variance_ratio_)
        print(self.svd.explained_variance_ratio_.sum())
        return x
