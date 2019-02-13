import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from classification.classifier import Classifier
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline
from sklearn import metrics


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_file, test_file, kfold):
		Classifier.__init__(self, path, train_file, test_file, kfold)

	def run(self):
		vectorizer = CountVectorizer(stop_words='english')
		transformer = TfidfTransformer()
		svd = TruncatedSVD(n_components=50, random_state=42)
		clf = LinearSVC() 		# TODO: Maybe use a non-linear SVM ?! 

		pipeline = Pipeline([
		('vect', vectorizer),
		('tfidf', transformer),
		('svd', svd),
		('clf', clf)
		])

		if not self.kfold : 
			pipeline.fit(self.X_train, self.Y_train)
			predicted = pipeline.predict(self.X_test)
			# TODO: print predictions
		else:
			kf = KFold(n_splits=10)
			for train_index, test_index in kf.split(self.X_train):
				pipeline.fit(self.X_train[train_index], self.Y_train[train_index])
				predicted = pipeline.predict(self.X_train[test_index])
				# print(self.Y_train[test_index])
				# print(predicted)
				# print(self.le.inverse_transform(predicted))
				Ylabels = self.le.inverse_transform(self.Y_train[test_index])
				Predlabels = self.le.inverse_transform(predicted)
				print(metrics.classification_report(Ylabels, Predlabels))
