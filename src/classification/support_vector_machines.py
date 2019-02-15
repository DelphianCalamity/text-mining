import pandas as pd
import numpy as np

from preprocess.preprocess import *
from classification.meanEmbeddingVectorizer import *

from sklearn.svm import LinearSVC
from classification.classifier import Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline
import gensim


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_df, test_file, kfold, features):
		Classifier.__init__(self, path, train_df, test_file, kfold, features)

	def run(self):

		tasks = []

		if self.features == "W2V":
			dim = 100
			self.X_train = Preprocessor().tokenize_articles(self.X_train)
			model = gensim.models.Word2Vec(self.X_train, size=dim)
			w2v = dict(zip(model.wv.index2word, model.wv.syn0))
			tasks.append(('vect', MeanEmbeddingVectorizer(w2v, dim)))
			# print(self.X_train)

			if not self.kfold : 
				self.X_test = Preprocessor().tokenize_articles(self.X_test)

		else:		# BoW is the default 
			tasks.append(('vect', CountVectorizer(stop_words='english')))
			tasks.append(('tfidf', TfidfTransformer()))

		if self.features == "SVD":
			svd = TruncatedSVD(n_components=50, random_state=42)
			tasks.append(('svd', svd))

		clf = LinearSVC() 		# TODO: Use a non-linear SVM and  experiment with kernels svm.SVC(kernel='linear', C=1)
		tasks.append(('clf', clf))
		
		pipeline = Pipeline(tasks)


		if not self.kfold : 
			pipeline.fit(self.X_train, self.Y_train)
			predicted = pipeline.predict(self.X_test)
			predlabels = self.le.inverse_transform(predicted)
			Classifier.PrintPredictorFile("SupportVectorMachines", predlabels, self.test_ids, self.path)

		else:
			score_array =[]
			accuracy_array = []
			kf = KFold(n_splits=10)

			for train_index, test_index in kf.split(self.X_train):
				pipeline.fit(self.X_train[train_index], self.Y_train[train_index])
				predicted = pipeline.predict(self.X_train[test_index])

				Ylabels = self.le.inverse_transform(self.Y_train[test_index])
				predlabels = self.le.inverse_transform(predicted)
				score_array.append(precision_recall_fscore_support(Ylabels, predlabels, average=None))
				accuracy_array.append(accuracy_score(Ylabels, predlabels))
				# print(metrics.classification_report(Ylabels, predlabels))

			PrintEvaluationFile("SupportVectorMachines", score_array, accuracy_array, path)
