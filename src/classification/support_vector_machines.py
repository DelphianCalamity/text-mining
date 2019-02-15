import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from classification.classifier import Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_df, test_file, kfold, features):
		Classifier.__init__(self, path, train_df, test_file, kfold, features)

	def run(self):


		tasks = []

		if self.features == "BOW":
			vectorizer = CountVectorizer(stop_words='english')
			tasks.append(('vect', vectorizer))

		transformer = TfidfTransformer()
		tasks.append(('tfidf', transformer))

		if self.features == "W2V":
			pass

		if self.features == "SVD":
			svd = TruncatedSVD(n_components=50, random_state=42)
			tasks.append(('svd', svd))

		clf = LinearSVC() 		# TODO: Use a non-linear SVM and  experiment with kernels
		tasks.append(('clf', clf))
		
		pipeline = Pipeline(tasks)

		if not self.kfold : 
			pipeline.fit(self.X_train, self.Y_train)
			predicted = pipeline.predict(self.X_test)
			predlabels = self.le.inverse_transform(predicted)
			Classifier.PrintPredictorFile("SupportVectorMachines", predlabels, self.test_ids, self.path)
		else:
			score_array =[]
			kf = KFold(n_splits=10)
			for train_index, test_index in kf.split(self.X_train):
				pipeline.fit(self.X_train[train_index], self.Y_train[train_index])
				predicted = pipeline.predict(self.X_train[test_index])

				Ylabels = self.le.inverse_transform(self.Y_train[test_index])
				predlabels = self.le.inverse_transform(predicted)
				score_array.append(precision_recall_fscore_support(Ylabels, predlabels, average=None))
				print(metrics.classification_report(Ylabels, predlabels))

			avg_score = np.mean(score_array,axis=0)
			print(avg_score)	

			# PrintEvaluationFile(accuracy_values, path)