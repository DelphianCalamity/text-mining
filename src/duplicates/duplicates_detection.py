import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

class DuplicateDetection:

    def __init__(self, path, train_file, dupThreshold):
        self.path = path
        self.csv_train_file = train_file
        self.train_df = pd.read_csv(self.csv_train_file, sep='\t')
        self.content = self.train_df['Content']
        self.threshold = dupThreshold

    # A naive algorithm for detecting duplicate documents
    def detect_duplicates(self):

        corpus = self.content.values
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(corpus).toarray()
        similarities = {}

        for idx_i, i in enumerate(X):
            if i.any(axis=0) :		
                for idx_j in range(idx_i+1, len(X)):
                    j = X[idx_j]
                    if j.any(axis=0):
                        similarity = 1 - spatial.distance.cosine(i, j)
                        if (similarity >= self.threshold):
                            similarities[(idx_i,idx_j)] = similarity
       
        f = open(self.path + "duplicates.txt", "w")
        for x in similarities:
            f.write(str(x[0]) + "	" + str(x[1]) + "	" + str(similarities[x]) + "\n")
        f.close()
   