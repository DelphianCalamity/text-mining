from sklearn.neighbors import LSHForest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from collections import namedtuple


def detect_duplicates(df, theta, path):

	corpus = df.values
	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform(corpus).toarray()
	# print(vectorizer.get_feature_names())
	print(X.shape)

	similarities = {}

	for idx_i, i in enumerate(X):
		if i.any(axis=0) :		
			for idx_j in range(idx_i+1, len(X)):

				j = X[idx_j]
				if j.any(axis=0):
					similarity = 1 - spatial.distance.cosine(i, j)
					if (similarity >= theta):
						similarities[(idx_i,idx_j)] = similarity

	
	f = open(path + "duplicates.txt", "w")
	for x in similarities:
		print(x[0])
		print(x[1])
		print(similarities[x])
		f.write(str(x[0]) + "	" + str(x[1]) + "	" + similarities[x])
		